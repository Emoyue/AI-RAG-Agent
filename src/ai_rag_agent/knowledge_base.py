"""知识库服务：文本 -> 切分 -> 向量化 -> Chroma 持久化，并维护去重台账。

修改要点：
1. 嵌入模型统一从 config 读取（原来硬编码，config 一改就会出现"入库用 A 模型、检索用 B 模型"的错位）；
2. 去掉"长度 > 1000 才切分"的伪阈值，统一交给分割器判断（短文本它本来也只产 1 块）；
3. 每个片段使用确定性 ID（md5-序号），重复入库变成"覆盖"而不是"新增副本"；
4. 台账改成 JSON：md5 -> {来源文件, 片段数, 时间}，支持按来源删除，并有旧格式（纯 md5 行）兼容迁移；
5. 新增 list_sources / delete_by_source，让知识库可管理（原来只能进、不能出）。
"""

import hashlib
import io
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import config_data as config
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from providers import embedding_signature, get_embeddings
from pypdf import PdfReader

load_dotenv()
logger = logging.getLogger(__name__)


# ====================== 文件解析 ======================
def decode_bytes(raw: bytes) -> str:
    """文本文件解码：UTF-8 优先，失败回退 GBK / GB18030（中文 Windows 的 txt 常见）。"""
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别的文本编码，请另存为 UTF-8 后重试")


def _load_pdf(raw: bytes) -> list[tuple[int, str]]:
    """按页抽取 PDF 文本，返回 [(页码, 文本)]。

    注意：只对"有文字层"的 PDF 有效。扫描件是图片，抽不出文字，
    需要先做 OCR（那是另一条流水线），这里会明确报错而不是静默入库空内容。
    """
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"PDF 解析失败（文件可能损坏）：{exc}") from exc

    if reader.is_encrypted:
        # 有些 PDF 只是"限制编辑"，用空密码就能打开；真正有密码的才失败
        try:
            if reader.decrypt("") == 0:
                raise ValueError("PDF 已加密，需要密码才能打开")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF 已加密，需要密码才能打开") from exc

    sections: list[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # 个别页损坏不影响整份文档
            logger.warning("第 %s 页解析失败，已跳过：%s", page_no, exc)
            continue
        if text.strip():
            sections.append((page_no, text))
    return sections


def load_sections(filename: str, raw: bytes) -> list[tuple[int | None, str]]:
    """把上传的字节流解析成 [(页码, 文本)]；TXT/MD 没有页码，页码为 None。"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(raw)
    return [(None, decode_bytes(raw))]


# ====================== 去重台账 ======================
def get_string_md5(input_str: str, encoding: str = "utf-8") -> str:
    """把字符串转换成 md5 十六进制串。"""
    return hashlib.md5(input_str.encode(encoding=encoding)).hexdigest()


def _load_records() -> dict:
    """读取台账；兼容旧版"每行一个 md5"的纯文本格式。"""
    path = Path(config.md5_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return {}

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 旧格式迁移：一行一个 md5，没有来源信息
        return {
            line.strip(): {"source": None, "chunks": 0, "time": None}
            for line in text.splitlines()
            if line.strip()
        }

    return data if isinstance(data, dict) else {}


def _save_records(records: dict) -> None:
    """台账原子写，避免记录写坏导致重复入库。"""
    path = Path(config.md5_path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def check_md5(md5_str: str) -> bool:
    """检查内容是否已入库（True 表示已处理过）。"""
    return md5_str in _load_records()


def save_md5(md5_str: str, source: str | None = None, chunks: int = 0) -> None:
    """记录一条入库指纹。"""
    records = _load_records()
    records[md5_str] = {
        "source": source,
        "chunks": chunks,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_records(records)


# ====================== 知识库 ======================
class KnowledgeBaseService:
    def __init__(self):
        Path(config.persist_directory).mkdir(parents=True, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,   # 相当于表名
            embedding_function=get_embeddings(),      # 由 .env 的 EMBEDDING_MODEL_NAME 决定
            persist_directory=config.persist_directory,
            # 记录建库时用的嵌入模型，用于发现"换了模型却没重建库"
            collection_metadata={"embedding": embedding_signature()},
        )
        self._warn_if_embedding_changed()
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def _warn_if_embedding_changed(self) -> None:
        """换嵌入模型但没重建库时给出明确告警——这种情况检索会"看起来正常、结果全错"。"""
        try:
            stored = (self.chroma._collection.metadata or {}).get("embedding")
        except Exception:  # noqa: BLE001 - 元数据读取失败不影响主流程
            return
        if stored and stored != embedding_signature():
            logger.warning(
                "向量库是用 %s 建的，当前配置是 %s。两者向量不可混用，请重建知识库"
                "（删除 %s 后重新上传文档）。",
                stored,
                embedding_signature(),
                config.persist_directory,
            )

    def upload_by_str(self, data: str, filename: str) -> dict:
        """把纯文本切分、向量化并写入向量库（TXT / 命令行 / 评测用）。"""
        return self._ingest([(None, data)], filename)

    def upload_bytes(self, filename: str, raw: bytes) -> dict:
        """按文件名后缀选解析方式：PDF 按页抽取，TXT/MD 解码；再统一入库。"""
        if not raw:
            return {"status": "error", "message": "[失败] 文件为空", "chunks": 0}
        try:
            sections = load_sections(filename, raw)
        except Exception as exc:  # noqa: BLE001 - 解析失败要变成界面提示，而不是 500
            return {"status": "error", "message": f"[失败] {exc}", "chunks": 0}
        return self._ingest(sections, filename)

    def _ingest(self, sections: list[tuple[int | None, str]], filename: str) -> dict:
        """统一入库：sections 是 [(页码, 文本)]，页码为空表示这份文档没有分页。

        为什么先合并再算 md5：去重是按"整份文件内容"判定的，
        而切分和写入是按"页"进行的——这样 PDF 的每个片段才能带上真实页码。
        """
        full_text = "\n\n".join(text for _, text in sections)
        if not full_text.strip():
            return {
                "status": "error",
                "message": "[失败] 没有可入库的文字（扫描版 PDF 需要先做 OCR）",
                "chunks": 0,
            }

        md5_hex = get_string_md5(full_text)
        if check_md5(md5_hex):
            return {"status": "skipped", "message": "[跳过] 相同内容已在知识库中", "chunks": 0}

        chunks: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        for page_no, text in sections:
            for part in self.spliter.split_text(text):
                if not part.strip():
                    continue
                index = len(chunks)
                # 确定性 ID：同一内容重复上传是覆盖，不是新增副本
                ids.append(f"{md5_hex}-{page_no or 0}-{index}")
                meta = {
                    "source": filename,
                    "chunk_index": index,
                    "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                if page_no:
                    meta["page"] = page_no
                metadatas.append(meta)
                chunks.append(part)

        if not chunks:
            return {"status": "error", "message": "[失败] 切分后没有有效片段", "chunks": 0}

        try:
            self.chroma.add_texts(chunks, metadatas=metadatas, ids=ids)
        except Exception as exc:  # 向量库/嵌入接口失败时，不写台账，便于原样重试
            return {"status": "error", "message": f"[失败] 写入向量库异常：{exc}", "chunks": 0}

        save_md5(md5_hex, source=filename, chunks=len(chunks))
        pages = len([1 for page, _ in sections if page])
        suffix = f"，来自 {pages} 页" if pages else ""
        return {
            "status": "success",
            "message": f"[成功] 已入库 {len(chunks)} 个片段{suffix}",
            "chunks": len(chunks),
            "pages": pages,
        }

    # ---------- 管理能力 ----------
    def list_sources(self) -> dict[str, int]:
        """列出知识库中的文件及其片段数。"""
        data = self.chroma.get(include=["metadatas"])
        counter = Counter(
            (meta or {}).get("source") or "未知来源"
            for meta in (data.get("metadatas") or [])
        )
        return dict(counter)

    def count_chunks(self) -> int:
        return len(self.chroma.get(include=[])["ids"])

    def delete_by_source(self, source: str) -> int:
        """按来源文件名删除全部片段，同时清理台账，允许重新入库。"""
        data = self.chroma.get(where={"source": source}, include=[])
        ids = data.get("ids") or []
        if ids:
            self.chroma.delete(ids=ids)

        records = _load_records()
        kept = {k: v for k, v in records.items() if (v or {}).get("source") != source}
        if len(kept) != len(records):
            _save_records(kept)

        return len(ids)


if __name__ == "__main__":
    service = KnowledgeBaseService()
    print(service.upload_by_str("这是一段用于自测的示例文本。", "testfile"))
    print("当前片段数：", service.count_chunks())
    print("来源统计：", service.list_sources())

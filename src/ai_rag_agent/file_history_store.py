"""基于本地 JSON 文件的会话记忆存储。

修改要点：
1. 删掉 `import Sequence`（原本是为了满足这个多余导入，才在 pyproject 里装了无关的 sequence 包）；
2. session_id 不再直接当文件名，改为"可读前缀 + md5 后缀"，避免路径穿越和非法字符；
3. 写入改为"临时文件 + os.replace"的原子操作，并加 fsync，避免写一半崩溃导致 JSON 损坏；
4. 读取时容错：文件损坏按空历史处理，不让整个应用崩掉；
5. 新增会话管理：文件里记录 session_id / updated_at，支持列出与删除（界面的「对话记录」页用）。
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

import config_data as config

logger = logging.getLogger(__name__)


def get_history(session_id: str) -> "FileChatMessageHistory":
    """RunnableWithMessageHistory 的历史工厂：按 session_id 返回一个文件型记忆实例。"""
    return FileChatMessageHistory(session_id, config.chat_history_directory)


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_path: str | os.PathLike):
        self.session_id = session_id          # 会话 id
        self.storage_path = Path(storage_path)  # 存放所有会话文件的目录
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_path / f"{self._safe_name(session_id)}.json"

    @staticmethod
    def _safe_name(session_id: str) -> str:
        """把任意 session_id 映射为安全文件名：保留少量可读字符，其余用哈希兜底。"""
        raw = str(session_id)
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
        readable = "".join(c for c in raw if c.isalnum() or c in "-_")[:24]
        return f"{readable}_{digest}" if readable else digest

    def _read_payload(self) -> dict:
        """读文件，兼容两种格式：
        - 新格式 {"session_id", "updated_at", "messages": [...]}
        - 旧格式 [...]（只有消息列表）
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"messages": []}
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("会话文件 %s 解析失败，按空历史处理：%s", self.file_path, exc)
            return {"messages": []}

        if isinstance(data, dict):
            return data
        return {"messages": data}

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """追加消息并整体落盘（原子写，避免文件损坏）。"""
        all_messages = list(self.messages) + list(messages)
        payload = {
            "session_id": self.session_id,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [message_to_dict(message) for message in all_messages],
        }

        tmp_path = self.file_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.file_path)  # 原子替换，读方永远看不到半个文件

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            return messages_from_dict(self._read_payload().get("messages") or [])
        except (ValueError, TypeError) as exc:
            logger.warning("会话文件 %s 消息格式异常：%s", self.file_path, exc)
            return []

    def clear(self) -> None:
        """清空当前会话（直接删除文件，比写空列表更干净）。"""
        if self.file_path.exists():
            self.file_path.unlink()


# ====================== 会话管理（界面「对话记录」页） ======================
def _recover_session_id(path: Path) -> str | None:
    """从文件名反推 session_id：去掉哈希后缀后，用同一套规则重算校验。

    这样旧格式文件（没有在内容里记 session_id）也能被正确还原并继续对话，
    而不是仅凭文件名去猜——猜错会让用户"打开了一个空会话"。
    """
    stem = path.stem
    if "_" not in stem:
        return None
    candidate = stem.rsplit("_", 1)[0]
    if FileChatMessageHistory._safe_name(candidate) == stem:
        return candidate
    return None


def list_sessions() -> list[dict]:
    """列出全部会话文件，按最后更新时间倒序。

    返回字段：file / session_id / messages / updated_at / preview
    """
    directory = Path(config.chat_history_directory)
    if not directory.exists():
        return []

    items: list[dict] = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("会话文件 %s 无法解析，列表中以 0 条消息展示", path)
            data = {}

        if isinstance(data, dict):
            raw_messages = data.get("messages") or []
            session_id = data.get("session_id") or _recover_session_id(path)
            updated_at = data.get("updated_at")
        else:  # 旧格式：整个文件就是消息列表
            raw_messages = data
            session_id = _recover_session_id(path)
            updated_at = None

        try:
            messages = messages_from_dict(raw_messages)
        except (ValueError, TypeError):
            messages = []

        if not updated_at:
            updated_at = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        items.append(
            {
                "file": path.name,
                "session_id": session_id,
                "messages": len(messages),
                "updated_at": updated_at,
                "preview": _preview(messages),
            }
        )

    items.sort(key=lambda item: item["updated_at"], reverse=True)
    return items


def _preview(messages: list[BaseMessage]) -> str:
    """取第一条用户消息作为摘要，便于在列表里认出这是哪次对话。"""
    for message in messages:
        if message.type == "human":
            text = str(message.content).replace("\n", " ").strip()
            return text[:36] + ("…" if len(text) > 36 else "")
    return "（没有用户消息）"


def delete_session_file(file_name: str) -> bool:
    """按文件名删除会话；只允许删除会话目录下的 .json，杜绝路径穿越。"""
    directory = Path(config.chat_history_directory).resolve()
    path = (directory / file_name).resolve()
    if path.parent != directory or path.suffix != ".json" or not path.is_file():
        return False
    path.unlink()
    return True

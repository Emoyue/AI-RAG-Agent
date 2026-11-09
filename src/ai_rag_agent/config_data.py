"""全局配置（唯一配置源）。

修改要点：
1. 所有路径都由本文件位置推导（BASE_DIR），不再依赖"你在哪个目录下运行"；
2. 密钥与模型名交给 .env / 环境变量，代码里只留可运行的默认值；
3. 切块参数、检索参数分开命名，避免"名字叫阈值、实际是条数"的误导。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---- 基础路径 ----------------------------------------------------------
# config_data.py -> ai_rag_agent -> src -> 项目根
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# 优先读包内 .env，其次读项目根 .env（谁存在用谁）
for _env_file in (BASE_DIR / ".env", PROJECT_ROOT / ".env"):
    if _env_file.exists():
        load_dotenv(_env_file)
        break

# ---- 存储路径（全部转为绝对路径） --------------------------------------
md5_path = str(BASE_DIR / "md5")  # 已入库内容的指纹台账
persist_directory = str(BASE_DIR / "chroma_db")  # Chroma 持久化目录
chat_history_directory = str(BASE_DIR / "chat_history")  # 会话记忆目录

# ---- 向量库 ------------------------------------------------------------
collection_name = "rag"

# ---- 文本分割 ----------------------------------------------------------
# 500 / 80（约 16% 重叠）：中文段落语义更完整，也不会产生大量近似重复片段
chunk_size = 500
chunk_overlap = 80
# 按"先段落、后句子、最后字符"的顺序尝试切分，"" 必须放最后作为兜底
separators = [
    "\n\n", "\n",
    "。", "！", "？", "；",
    ".", "!", "?", ";",
    "，", ",",
    " ",
    "",
]

# ---- 检索 --------------------------------------------------------------
top_k = 4          # 最终交给大模型的片段数
fetch_k = 20       # MMR 候选池大小（需 >= top_k）
mmr_lambda = 0.5   # 0=更看重多样性，1=更看重相似度

# ---- 模型（只对接阿里云百炼 / DashScope）--------------------------------
# 全部取自 .env：想换模型只改 .env 里的模型名，代码一行都不用动。
# 换嵌入模型后必须重建知识库（不同嵌入模型的向量不在同一空间），详见 README。


def _env_str(name: str, default: str = "") -> str:
    """读取字符串配置：去掉首尾空格，空串按默认值处理（防止 .env 里留个空键）。"""
    return (os.getenv(name) or "").strip() or default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


# API 配置
dashscope_api_key = _env_str("DASHSCOPE_API_KEY")
# 一般留空，走百炼公网默认网关；只有自建/专属网关才需要填（原生 API 根地址）
dashscope_base_url = _env_str("DASHSCOPE_BASE_URL")

# 模型配置
chat_model_name = _env_str("CHAT_MODEL_NAME", "qwen-plus")
# 对话模型挂在哪个端点上：
#   text       —— 文本生成端点（ChatTongyi），适用于 qwen-plus / qwen-max 等
#   multimodal —— 多模态端点（MultiModalConversation），适用于 gui-plus / qwen3.7-plus 等
# 填错的典型症状是 400 "url error, please check url!"（它说的不是地址错，是模型与端点不匹配）
chat_model_mode = _env_str("CHAT_MODEL_MODE", "text").lower()
if chat_model_mode not in {"text", "multimodal"}:
    chat_model_mode = "text"
embedding_model_name = _env_str("EMBEDDING_MODEL_NAME", "text-embedding-v4")
temperature = _env_float("TEMPERATURE", 0.1)  # 客服场景偏低，减少发挥
top_p = _env_float("TOP_P", 0.8)
max_retries = _env_int("MAX_RETRIES", 3)

# ---- 会话 --------------------------------------------------------------
history_max_messages = 10  # 注入提示词的历史消息上限（约 5 轮），防止 prompt 无限膨胀
default_session_id = "user_001"  # 仅命令行自测使用；Web 端为每个浏览器会话生成独立 ID

# ---- 检索增强：问题改写 -------------------------------------------------
# 追问时（"那它多少钱？"）原句检索不到东西，先把问题改写成可独立检索的完整句子再检索。
# 关掉它（ENABLE_QUERY_REWRITE=0）就能直观对比"有改写/无改写"的召回差异。
enable_query_rewrite = os.getenv("ENABLE_QUERY_REWRITE", "1") == "1"

# ---- 运行时开关 --------------------------------------------------------
debug_prompt = os.getenv("DEBUG_PROMPT", "0") == "1"  # 打印完整提示词，默认关闭
upload_max_size_mb = 10  # 单文件上传大小上限

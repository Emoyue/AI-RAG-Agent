"""模型工厂：只对接阿里云百炼（DashScope）。

为什么还要这一层？
- 业务代码里不再出现 `ChatTongyi(...)` / `DashScopeEmbeddings(...)`，
  换模型只改 `.env`（CHAT_MODEL_NAME / EMBEDDING_MODEL_NAME），改代码为零；
- 密钥只在 `.env` 里出现一次，由这里统一注入，不在多处散落。

两个签名函数（signature）有两个实际用途：
1. 界面上显示"现在用的是谁"，排障第一眼就有答案；
2. 写进向量库元数据，用来发现"换了嵌入模型却没重建库"——
   不同嵌入模型产出的向量不在同一个空间，混用会"检索看着正常、答案全错"。

自检：uv run python src/ai_rag_agent/providers.py
"""

from __future__ import annotations

import logging
import os

import config_data as config

logger = logging.getLogger(__name__)


# ====================== 内部工具 ======================
def mask_key(key: str) -> str:
    """密钥打码，只露头尾，方便确认"填的是哪一把"又不泄漏。"""
    if not key:
        return "(未配置)"
    if len(key) <= 12:
        return key[:2] + "***"
    return f"{key[:6]}***{key[-4:]}"


def _require_api_key() -> str:
    """取密钥；没配就直接给出可操作的提示，而不是让 SDK 抛一句费解的报错。"""
    key = config.dashscope_api_key
    if not key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY。\n"
            "请在项目根目录的 .env 里填写：\n"
            "    DASHSCOPE_API_KEY=sk-xxxxxxxx\n"
            "密钥在阿里云百炼控制台 → API-KEY 管理 创建。"
        )
    return key


def _prepare(api_key: str) -> None:
    """把配置交给 DashScope SDK。

    SDK 内部有些代码路径直接读环境变量，所以这里同时设置 os.environ，
    避免"传了参数却不生效"的怪现象。
    """
    os.environ.setdefault("DASHSCOPE_API_KEY", api_key)

    if config.dashscope_base_url:
        import dashscope

        dashscope.base_http_api_url = config.dashscope_base_url


# ====================== 对外接口 ======================
def chat_signature() -> str:
    """当前对话模型的标识（界面展示用）。"""
    suffix = "(多模态端点)" if config.chat_model_mode == "multimodal" else ""
    return f"dashscope:{config.chat_model_name}{suffix}"


def embedding_signature() -> str:
    """当前嵌入模型的标识（界面展示 + 向量库元数据用）。"""
    return f"dashscope:{config.embedding_model_name}"


def error_hint(exc: BaseException) -> str:
    """把常见的百炼报错翻译成"下一步该改什么"，界面上直接显示。"""
    message = str(exc)
    if "url error" in message.lower():
        return (
            "该模型与接口端点不匹配：`gui-plus`、`qwen3.7-plus` 这类模型只能走多模态端点，"
            "请在 .env 里设 `CHAT_MODEL_MODE = multimodal`；"
            "`qwen-plus`、`qwen-max` 这类纯文本模型则设 `text`。"
        )
    if "Model not exist" in message:
        return "模型名不存在或当前账号未开通，请到百炼控制台的模型列表核对 `CHAT_MODEL_NAME`。"
    if "Arrearage" in message:
        return "账号欠费，请到阿里云费用中心充值后再试。"
    if "InvalidApiKey" in message or "Unauthorized" in message:
        return "API Key 无效，请检查 .env 里的 `DASHSCOPE_API_KEY`。"
    return ""


def get_chat_model():
    """返回对话模型实例。换模型只需改 .env 的 CHAT_MODEL_NAME / CHAT_MODEL_MODE。

    百炼有两类端点，模型挂在哪一类上决定了用哪个客户端：
    - text       → ChatTongyi（文本生成端点）
    - multimodal → MultiModalChatModel（多模态端点，gui-plus / qwen3.7-plus 这类）
    两者对 LangChain 都是 BaseChatModel，上层链不用关心区别。
    """
    api_key = _require_api_key()
    _prepare(api_key)

    if config.chat_model_mode == "multimodal":
        from chat_model_mm import MultiModalChatModel

        return MultiModalChatModel(
            model=config.chat_model_name,
            dashscope_api_key=api_key,
            temperature=config.temperature,
            top_p=config.top_p,
            max_retries=config.max_retries,
            base_http_api_url=config.dashscope_base_url,
        )

    from langchain_community.chat_models.tongyi import ChatTongyi

    return ChatTongyi(
        model=config.chat_model_name,
        dashscope_api_key=api_key,
        top_p=config.top_p,
        # ChatTongyi 没有 temperature 字段，但 model_kwargs 会原样透传给 DashScope
        model_kwargs={"temperature": config.temperature},
        max_retries=config.max_retries,
    )


def get_embeddings():
    """返回嵌入模型实例。换模型只需改 .env 的 EMBEDDING_MODEL_NAME（换完必须重建库）。"""
    api_key = _require_api_key()
    _prepare(api_key)

    from langchain_community.embeddings import DashScopeEmbeddings

    return DashScopeEmbeddings(
        model=config.embedding_model_name,
        dashscope_api_key=api_key,
        max_retries=config.max_retries,
    )


# ====================== 自检（不发请求、不花钱） ======================
def check() -> None:
    print("后端　　　：阿里云百炼（DashScope）")
    print(f"API 密钥　：{mask_key(config.dashscope_api_key)}")
    print(f"网关　　　：{config.dashscope_base_url or '默认公网（dashscope.aliyuncs.com）'}")
    print(f"对话模型　：{config.chat_model_name}  (端点={config.chat_model_mode}, temperature={config.temperature}, top_p={config.top_p})")
    print(f"嵌入模型　：{config.embedding_model_name}")
    print(f"重试次数　：{config.max_retries}")
    print()

    if not config.dashscope_api_key:
        print("❌ 未配置 DASHSCOPE_API_KEY，模型无法调用。请在 .env 中补上。")
        return

    try:
        import dashscope

        print(f"✅ dashscope SDK {dashscope.__version__} 已安装")
    except ImportError:
        print("❌ 未安装 dashscope：uv add dashscope")
        return

    print("✅ 配置完整。真实调用若返回 400 / Arrearage / AllocationQuota，通常是余额或额度问题")
    print("   注意：换过嵌入模型后要删除 chroma_db 与 md5 再重新入库，否则新旧向量会混在一起")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check()

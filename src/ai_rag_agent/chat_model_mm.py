"""多模态对话模型适配器：把 DashScope 的 MultiModalConversation 包成 LangChain 的 ChatModel。

为什么需要这个文件？

百炼的模型分两类端点：
- 文本生成（`Generation` / `/services/aigc/text-generation/generation`）—— qwen-plus、qwen-max 等；
- 多模态（`MultiModalConversation` / `/services/aigc/multimodal-generation/generation`）—— gui-plus、qwen3.7-plus 等。

`ChatTongyi` 内部固定调用**文本生成**端点，所以一个只挂在多模态端点上的模型，
用它调用会返回 `400 InvalidParameter: url error, please check url!`（这条报错文案极易误导，
它说的不是地址写错，而是"模型和端点不匹配"）。

本类只做一件事：把 LangChain 的 messages 转成 DashScope 多模态格式，
调用 MultiModalConversation，再把返回的 content 拆回文本。
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)

# LangChain 的 message.type -> DashScope 的 role
_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def _to_payload(messages: list[BaseMessage]) -> list[dict]:
    """LangChain 消息 -> DashScope 多模态消息。

    多模态接口的 content 必须是 list（`[{"text": "..."}]`），
    即使只有纯文本也要包一层，直接传字符串会被判参数错误。
    """
    payload: list[dict] = []
    for message in messages:
        content = message.content if isinstance(message.content, str) else str(message.content)
        payload.append(
            {
                "role": _ROLE_MAP.get(message.type, "user"),
                "content": [{"text": content}],
            }
        )
    return payload


def _extract_text(output: Any) -> str:
    """从多模态响应里取出纯文本（content 是 `[{"text": ...}]` 列表）。"""
    try:
        content = output["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def describe_error(response: Any) -> str:
    """把 DashScope 的响应对象转成一句看得懂的报错。"""
    code = getattr(response, "code", None) or "Unknown"
    message = getattr(response, "message", "") or ""
    status = getattr(response, "status_code", "?")

    hint = ""
    if "url error" in str(message).lower():
        hint = (
            "\n提示：这条报错通常表示「模型与接口端点不匹配」。"
            "该模型可能只能走多模态端点，请把 .env 的 CHAT_MODEL_MODE 设为 multimodal；"
            "若它本来是纯文本模型，则设为 text。"
        )
    elif "Model not exist" in str(message):
        hint = "\n提示：模型名不存在或当前账号没有开通，请到百炼控制台核对模型名。"

    return f"HTTP {status} · {code} · {message}{hint}"


def _is_retryable(response: Any) -> bool:
    """只对网络抖动 / 限流 / 服务端 5xx 重试；参数错误重试没有意义。"""
    status = getattr(response, "status_code", 0)
    if status >= 500:
        return True
    code = str(getattr(response, "code", "") or "")
    return code in {"Throttling", "Throttling.RateQuota", "ServiceUnavailable", "InternalError"}


class MultiModalChatModel(BaseChatModel):
    """用 DashScope 多模态端点做纯文本对话的 ChatModel。"""

    model: str
    dashscope_api_key: str = ""
    temperature: float = 0.1
    top_p: float = 0.8
    max_retries: int = 3
    base_http_api_url: str = ""

    @property
    def _llm_type(self) -> str:
        return "dashscope-multimodal"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "mode": "multimodal"}

    # ---------- 内部：一次调用 ----------
    def _call(self, messages: list[BaseMessage], stream: bool = False):
        import dashscope
        from dashscope import MultiModalConversation

        if self.base_http_api_url:
            dashscope.base_http_api_url = self.base_http_api_url

        return MultiModalConversation.call(
            model=self.model,
            messages=_to_payload(messages),
            api_key=self.dashscope_api_key,
            result_format="message",
            temperature=self.temperature,
            top_p=self.top_p,
            stream=stream,
            incremental_output=stream,  # 流式时每次只回增量，避免重复拼接
        )

    # ---------- 非流式 ----------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = None
        for attempt in range(max(1, self.max_retries)):
            response = self._call(messages)
            if getattr(response, "status_code", 0) == 200:
                text = _extract_text(response.output)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

            last = response
            if not _is_retryable(response):
                break
            logger.warning(
                "第 %s 次调用失败（%s），准备重试", attempt + 1, describe_error(response)
            )

        raise RuntimeError(describe_error(last))

    # ---------- 流式 ----------
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        responses = self._call(messages, stream=True)
        for response in responses:
            if getattr(response, "status_code", 0) != 200:
                raise RuntimeError(describe_error(response))

            text = _extract_text(response.output)
            if not text:
                continue
            if run_manager:
                run_manager.on_llm_new_token(text)
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))

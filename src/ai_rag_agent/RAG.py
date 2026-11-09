"""RAG 服务：检索 -> 拼提示词 -> 大模型 -> 解析 + 会话记忆。

修改要点：
1. 链用 RunnablePassthrough.assign 逐步"加料"，不再出现 value["input"]["input"] 这种嵌套；
2. 提示词增加"资料不足必须说不知道"和"标注来源"两条约束，抑制无根据发挥；
3. 历史只注入最近 N 条，防止多轮对话把 prompt 撑爆；
4. 调试打印改为 logger + 开关，默认不再把用户问题和知识库原文打到控制台；
5. 新增"问题改写（condense question）"：追问时先把问题补全成可独立检索的句子，再去检索；
6. 检索与生成拆成两段（_prepare / stream_answer）：检索结果能带出到界面做"引用回显"，
   而生成阶段仍是流式，两者不互相牵制。
"""

import logging

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableWithMessageHistory,
)

import config_data as config
from file_history_store import get_history
from providers import get_chat_model, get_embeddings
from vector_stores import VectorStoreService

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是企业知识库客服，请严格依据【参考资料】回答用户问题：\n"
    "1. 只使用参考资料中的信息作答，不要编造事实或数字；\n"
    "2. 如果参考资料不足以回答，直接说明\"知识库中未找到相关内容\"，并指出还缺什么信息；\n"
    "3. 回答简洁、专业；引用到具体资料时，标注来源文件名。\n\n"
    "【参考资料】\n{context}"
)

# 问题改写专用提示词：只做"补全"，不做"回答"
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是检索问题改写器。请结合对话历史，把用户的最新问题改写成一个"
            "不依赖上下文、可独立检索的完整问题。\n"
            "规则：\n"
            "1. 补全代词所指的对象（它/这个/那个/该产品指的是什么）；\n"
            "2. 保留原问题里的型号、数字、专有名词；\n"
            "3. 不要回答问题，不要解释，不要加引号；\n"
            "4. 如果原问题本身已经完整，就原样输出。\n"
            "只输出改写后的问题本身。",
        ),
        MessagesPlaceholder("history"),
        ("user", "{input}"),
    ]
)


def _format_document(docs: list[Document]) -> str:
    """把检索到的片段拼成一段文本，顺便带上来源与页码，便于模型标注出处。"""
    if not docs:
        return "（未检索到任何资料，请如实告知用户知识库中没有相关内容）"

    parts = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "未知来源")
        page = meta.get("page")
        label = f"{source} 第{page}页" if page else source
        parts.append(f"【来源：{label}】\n{doc.page_content}")
    return "\n\n".join(parts)


def _to_sources(docs: list[Document]) -> list[dict]:
    """把命中的片段整理成界面可渲染的引用列表（同一页同一片段只留一次）。"""
    sources: list[dict] = []
    seen: set[tuple] = set()
    for doc in docs:
        meta = doc.metadata or {}
        key = (meta.get("source"), meta.get("page"), meta.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        text = " ".join(doc.page_content.split())
        sources.append(
            {
                "source": meta.get("source", "未知来源"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "snippet": text[:180] + ("…" if len(text) > 180 else ""),
            }
        )
    return sources


def _trim_history(messages: list[BaseMessage] | None) -> list[BaseMessage]:
    """只保留最近若干条历史，避免 prompt 随对话轮次无限增长。"""
    if not messages:
        return []
    return messages[-config.history_max_messages:]


def _history_of(value: dict) -> list[BaseMessage]:
    return _trim_history(value.get("history"))


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(embedding=get_embeddings())
        self.retriever = self.vector_service.get_retriever()

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("history", optional=True),
                ("user", "{input}"),
            ]
        )

        self.chat_model = get_chat_model()
        self.answer_chain = (
            self.prompt_template
            | RunnableLambda(self._maybe_print_prompt)
            | self.chat_model
            | StrOutputParser()
        )

        # 命令行 / 评测用：输入问题，输出答案字符串（引用只进提示词，不返回）
        self.chain = RunnableWithMessageHistory(
            RunnableLambda(self._prepare)
            | RunnablePassthrough.assign(answer=self.answer_chain)
            | RunnableLambda(lambda value: value["answer"]),
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # 界面用：检索已由 prepare() 提前完成，这里只负责"流式生成 + 落历史"
        self.answer_with_history = RunnableWithMessageHistory(
            RunnablePassthrough.assign(history=RunnableLambda(_history_of))
            | RunnablePassthrough.assign(answer=self.answer_chain)
            | RunnableLambda(lambda value: value["answer"]),
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _maybe_print_prompt(self, prompt):
        """提示词调试：默认只进 debug 日志，需要时才打到控制台。"""
        logger.debug("最终提示词：\n%s", prompt.to_string())
        if config.debug_prompt:
            print("=" * 20)
            print(prompt.to_string())
            print("=" * 20)
        return prompt

    def _condense_question(self, payload: dict) -> str:
        """把"那它多少钱？"这类依赖上下文的问题，改写成可独立检索的完整问题。

        设计要点（也是省钱的要点）：
        - 没有历史 = 没有指代 = 无需改写，直接返回原问题，省掉一次模型调用；
        - 改写失败（网络/模型异常）时回退原问题，宁可少改不可报错；
        - 改写只用最近若干条历史，和最终提示词保持同一窗口。
        """
        question = (payload.get("input") or "").strip()
        history = _trim_history(payload.get("history"))

        if not question or not history or not config.enable_query_rewrite:
            return question

        try:
            message = (CONDENSE_PROMPT | self.chat_model).invoke(
                {"input": question, "history": history}
            )
            rewritten = message.content if isinstance(message.content, str) else str(message.content)
            rewritten = rewritten.strip().strip('"').strip("“”").strip("「」").strip()
        except Exception as exc:
            logger.warning("问题改写失败，回退为原问题：%s", exc)
            return question

        rewritten = rewritten or question
        if rewritten != question:
            logger.info("问题改写：%s -> %s", question, rewritten)
            if config.debug_prompt:
                print(f"[问题改写] {question} -> {rewritten}")
        return rewritten

    def _prepare(self, payload: dict) -> dict:
        """检索阶段：改写问题 → 召回资料 → 拼好上下文。

        单独抽出来是为了让界面拿到 `docs`（引用回显）和 `standalone`（补全后的问题）。
        如果检索和生成揉在一条链里，链的输出只有一段文本，命中了哪些片段就无从得知了。
        """
        question = (payload.get("input") or "").strip()
        history = _trim_history(payload.get("history"))
        standalone = self._condense_question({"input": question, "history": history})
        docs = self.retriever.invoke(standalone) if standalone else []

        return {
            "input": question,
            "history": history,
            "standalone": standalone,
            "docs": docs,
            "context": _format_document(docs),
            "sources": _to_sources(docs),
        }

    # ---------- 供界面调用：先检索、再流式生成 ----------
    def prepare(self, question: str, session_id: str) -> dict:
        """检索阶段。历史从存储里读，和生成阶段用的是同一份会话。"""
        history = get_history(session_id).messages
        return self._prepare({"input": question, "history": history})

    def stream_answer(self, prepared: dict, session_id: str):
        """生成阶段：流式返回答案片段；消息历史由 answer_with_history 负责读写。"""
        return self.answer_with_history.stream(
            prepared, {"configurable": {"session_id": session_id}}
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    session_config = {"configurable": {"session_id": config.default_session_id}}
    rag = RagService()

    # 两轮对话：第二轮故意用指代词，观察 [问题改写] 日志
    print("Q1:", rag.chain.invoke({"input": "知识库里有哪些内容？"}, session_config))
    print("-" * 40)
    print("Q2:", rag.chain.invoke({"input": "那它有没有提到薪资结构？"}, session_config))

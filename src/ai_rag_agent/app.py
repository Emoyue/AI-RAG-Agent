"""ai-rag-agent 单入口应用：侧边栏切换「智能问答 / 知识库管理 / 对话记录」。

为什么合并成一个文件？
1. 两个 Streamlit 进程 = 两套缓存、两份模型实例，端口也要记两个；
2. 合并后 `@st.cache_resource` 在页面间共享，切页不重建向量库和模型；
3. 侧边栏可以放"当前会话""清空对话""知识库统计"这类全局信息，用户体验才像个产品。

本次新增：
- 引用回显：回答下方列出命中的资料（文件 / 页码 / 原文片段），答案不再"无凭无据"；
- PDF 上传：按页抽取文本，片段带上页码，引用能精确到第几页；
- 对话记录：列出本机所有会话，可打开继续聊、可删除。

运行：uv run streamlit run src/ai_rag_agent/app.py
"""

import uuid

import streamlit as st

import config_data as config
from RAG import RagService
from file_history_store import delete_session_file, get_history, list_sessions
from knowledge_base import KnowledgeBaseService
from providers import chat_signature, embedding_signature, error_hint

st.set_page_config(page_title="智能客服 · 知识库", page_icon="🌙")

PAGES = ["💬 智能问答", "📚 知识库管理", "🗂 对话记录"]
GREETING = "你好，有什么可以帮助你？"


# ====================== 共享资源（跨页面只初始化一次） ======================
@st.cache_resource(show_spinner=False)
def get_rag() -> RagService:
    return RagService()


@st.cache_resource(show_spinner=False)
def get_kb() -> KnowledgeBaseService:
    return KnowledgeBaseService()


def goto(page: str) -> None:
    """切换页面：先记下目标页，下一次 rerun 在 radio 实例化之前写入，避免 Streamlit 报错。"""
    st.session_state["pending_page"] = page
    st.rerun()


def load_messages(session_id: str) -> list[dict]:
    """从后端记忆还原对话，让"刷新页面"和"打开历史会话"都不丢上下文。"""
    messages = []
    for message in get_history(session_id).messages:
        if message.type not in {"human", "ai"}:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        messages.append({"role": "user" if message.type == "human" else "assistant", "content": content})
    return messages


# ====================== 引用回显 ======================
def render_sources(sources: list[dict]) -> None:
    """把命中的片段渲染成可展开的引用列表。"""
    if not sources:
        st.caption("📎 本次回答没有引用知识库内容")
        return

    with st.expander(f"📎 参考资料 · 命中 {len(sources)} 段", expanded=False):
        for index, item in enumerate(sources, start=1):
            location = item["source"]
            if item.get("page"):
                location += f" · 第 {item['page']} 页"
            st.markdown(f"**[{index}] {location}**")
            st.caption(item["snippet"])
            if index < len(sources):
                st.divider()


# ====================== 页面一：智能问答 ======================
def render_qa() -> None:
    st.title("智能客服")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"user_{uuid.uuid4().hex[:8]}"
    session_id = st.session_state["session_id"]

    if "message" not in st.session_state:
        restored = load_messages(session_id)
        st.session_state["message"] = restored or [{"role": "assistant", "content": GREETING}]

    with st.sidebar:
        st.caption(f"当前会话：{session_id}")
        if st.button("清空当前对话", use_container_width=True):
            get_history(session_id).clear()  # 后端记忆
            st.session_state["message"] = [{"role": "assistant", "content": "对话已清空，我们重新开始。"}]
            st.rerun()

    for message in st.session_state["message"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # 只有本轮新产生的回答才带引用（历史文件里只存文本，不存引用）
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input()
    if not prompt:
        return

    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    answer_parts: list[str] = []
    sources: list[dict] = []

    def collect(stream):
        """边吐给前端、边把片段攒起来，最后一次性写回会话列表。"""
        for chunk in stream:
            answer_parts.append(chunk)
            yield chunk

    try:
        rag = get_rag()
        # ① 先检索：拿到命中的片段，才能把引用显示出来
        prepared = rag.prepare(prompt, session_id)
        sources = prepared["sources"]

        # ② 再流式生成：历史读写仍由 RunnableWithMessageHistory 负责
        with st.chat_message("assistant"):
            st.write_stream(collect(rag.stream_answer(prepared, session_id)))
            render_sources(sources)
    except Exception as exc:  # 网络、额度、模型名、端点不匹配都在这里兜住
        hint = error_hint(exc)
        st.error(f"回答生成失败：{exc}" + (f"\n\n💡 {hint}" if hint else ""))
    finally:
        if answer_parts:
            st.session_state["message"].append(
                {"role": "assistant", "content": "".join(answer_parts), "sources": sources}
            )


# ====================== 页面二：知识库管理 ======================
def render_kb() -> None:
    st.title("知识库管理")
    service = get_kb()

    uploader_file = st.file_uploader(
        "请上传文档（支持 TXT / MD / PDF）",
        type=["txt", "md", "pdf"],
        accept_multiple_files=False,
    )

    if uploader_file is not None:
        raw = uploader_file.getvalue()
        size_mb = len(raw) / 1024 / 1024

        st.subheader(f"文件名：{uploader_file.name}")
        st.write(f"格式：{uploader_file.type or '未知'} | 大小：{size_mb:.2f} MB")

        if size_mb > config.upload_max_size_mb:
            st.error(f"文件过大（上限 {config.upload_max_size_mb} MB），请拆分后再上传。")
        elif not raw:
            st.warning("文件为空，没有可入库的内容。")
        else:
            with st.spinner("解析并入库中..."):
                result = service.upload_bytes(uploader_file.name, raw)

            if result["status"] == "success":
                st.success(result["message"])
            elif result["status"] == "skipped":
                st.info(result["message"])
            else:
                st.error(result["message"])

    st.divider()
    st.subheader("知识库内容")

    sources = service.list_sources()
    if not sources:
        st.caption("知识库还是空的，先上传一个文档试试（可用 samples/ 下的示例）。")
        return

    st.caption(f"共 {sum(sources.values())} 个片段，来自 {len(sources)} 个文件")
    for name, chunks in sources.items():
        col_name, col_action = st.columns([4, 1])
        col_name.write(f"📄 {name}（{chunks} 个片段）")
        if col_action.button("删除", key=f"delete_{name}", use_container_width=True):
            removed = service.delete_by_source(name)
            st.success(f"已删除 {name} 的 {removed} 个片段")
            st.rerun()


# ====================== 页面三：对话记录 ======================
def render_history() -> None:
    st.title("对话记录")
    st.caption(f"保存在本机：`{config.chat_history_directory}`")

    sessions = list_sessions()
    if not sessions:
        st.info("还没有任何对话记录。去「智能问答」问一句，这里就会出现。")
        return

    st.caption(f"共 {len(sessions)} 个会话，按最后更新时间排序")
    current_id = st.session_state.get("session_id")

    for item in sessions:
        with st.container(border=True):
            col_info, col_open, col_delete = st.columns([5, 1, 1])
            is_current = item["session_id"] == current_id
            title = f"**{item['preview']}**" + ("　`当前会话`" if is_current else "")
            col_info.markdown(title)
            col_info.caption(
                f"{item['updated_at']} · {item['messages']} 条消息 · `{item['session_id'] or '旧格式文件'}`"
            )

            if col_open.button(
                "继续聊",
                key=f"open_{item['file']}",
                use_container_width=True,
                disabled=not item["session_id"],
            ):
                st.session_state["session_id"] = item["session_id"]
                st.session_state.pop("message", None)  # 让问答页从磁盘重新加载
                goto("💬 智能问答")

            if col_delete.button("删除", key=f"del_{item['file']}", use_container_width=True):
                if delete_session_file(item["file"]):
                    # 删掉的正好是当前会话，界面也要跟着清空
                    if item["session_id"] == current_id:
                        st.session_state.pop("message", None)
                    st.rerun()
                else:
                    st.error("删除失败：文件名不合法或文件已不存在")


# ====================== 入口 ======================
def main() -> None:
    pending = st.session_state.pop("pending_page", None)
    if pending:
        st.session_state["nav"] = pending

    with st.sidebar:
        st.title("🌙 智能客服")
        page = st.radio("功能", PAGES, label_visibility="collapsed", key="nav")
        st.divider()
        st.caption(f"对话模型：`{chat_signature()}`")
        st.caption(f"嵌入模型：`{embedding_signature()}`")

    if page == "💬 智能问答":
        render_qa()
    elif page == "📚 知识库管理":
        render_kb()
    else:
        render_history()


if __name__ == "__main__":
    main()

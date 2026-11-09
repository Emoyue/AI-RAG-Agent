"""ai-rag-agent：基于 LangChain + Chroma + 阿里云百炼的企业知识库问答（RAG）应用。"""


def main() -> None:
    print(
        "本项目是 Streamlit 应用，请在项目根目录执行：\n"
        "  uv run streamlit run src/ai_rag_agent/app.py   # 单入口：问答 / 知识库 / 对话记录\n"
        "自检配置（不花钱）：\n"
        "  uv run python src/ai_rag_agent/providers.py\n"
        "命令行两轮对话：\n"
        "  uv run python src/ai_rag_agent/RAG.py"
    )

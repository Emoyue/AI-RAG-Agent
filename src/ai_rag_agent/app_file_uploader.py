"""兼容入口：知识库管理页已合并到 app.py。

旧命令 `streamlit run app_file_uploader.py` 依然可用，但推荐：
    uv run streamlit run src/ai_rag_agent/app.py
完整实现见 app.py 中的 render_kb()。
"""

from app import render_kb

render_kb()

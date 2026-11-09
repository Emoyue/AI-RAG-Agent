"""兼容入口：问答页已合并到 app.py。

旧命令 `streamlit run app_qa.py` 依然可用，但推荐：
    uv run streamlit run src/ai_rag_agent/app.py
完整实现见 app.py 中的 render_qa()。
"""

from app import render_qa

render_qa()

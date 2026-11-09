"""向量库服务：只负责"建 Chroma 实例 + 给检索器"，不掺业务。

修改要点：
1. 检索改为 MMR（最大边际相关性）：k=4 个结果彼此去重，不再是两个"双胞胎片段"；
2. 参数全部来自 config，命名与含义对齐（top_k 就是条数，不再冒充 similarity_threshold）。
"""

import config_data as config
from langchain_chroma import Chroma


class VectorStoreService:
    def __init__(self, embedding):
        """:param embedding: 嵌入模型实例"""
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回可插入 LCEL 链的检索器。"""
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.top_k,
                "fetch_k": config.fetch_k,
                "lambda_mult": config.mmr_lambda,
            },
        )


if __name__ == "__main__":
    from providers import get_embeddings

    retriever = VectorStoreService(get_embeddings()).get_retriever()
    for doc in retriever.invoke("我体重120斤，推荐一下我适合的尺码"):
        print(doc.metadata.get("source"), doc.metadata.get("chunk_index"), "|", doc.page_content[:60])

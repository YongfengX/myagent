import uuid
import os
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")


class LongTermMemory:
    """
    基于向量语义检索的长期记忆。
    使用 sentence-transformers 生成 embedding，ChromaDB 持久化存储。
    """

    def __init__(self, persist_dir: str = _DEFAULT_PERSIST_DIR):
        self._encoder = SentenceTransformer(_MODEL_NAME)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, content: str) -> str:
        """存入一条记忆，返回确认信息。"""
        entry_id = str(uuid.uuid4())[:8]
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embedding = self._encoder.encode(content).tolist()
        self._col.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"created_at": created_at}],
        )
        return f"已存入长期记忆 [id:{entry_id}]: {content}"

    def search(self, query: str, top_k: int = 3) -> str:
        """语义检索，返回最相关的 top_k 条记忆。"""
        if self._col.count() == 0:
            return "长期记忆为空。"
        actual_k = min(top_k, self._col.count())
        embedding = self._encoder.encode(query).tolist()
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=actual_k,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        lines = [f"[{m['created_at']}] {d}" for d, m in zip(docs, metas)]
        return "\n".join(lines)

    def get_all_summary(self, max_entries: int = 5) -> str:
        """返回最近几条记忆，用于注入 system prompt。"""
        if self._col.count() == 0:
            return ""
        all_items = self._col.get(include=["documents", "metadatas"])
        entries = list(zip(all_items["metadatas"], all_items["documents"]))
        entries.sort(key=lambda x: x[0]["created_at"])
        recent = entries[-max_entries:]
        lines = [f"- [{m['created_at']}] {d}" for m, d in recent]
        return "\n".join(lines)

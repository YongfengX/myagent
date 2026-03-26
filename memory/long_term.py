import json
import os
import uuid
from datetime import datetime

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "long_term_store.json")


class LongTermMemory:
    """
    基于 JSON 文件的持久化长期记忆。
    支持关键词检索（无需向量数据库）。
    """

    def __init__(self, filepath: str = MEMORY_FILE):
        self.filepath = filepath
        self._store: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._store = json.load(f)

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    def save(self, content: str) -> str:
        """存入一条记忆，返回确认信息。"""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "content": content,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._store.append(entry)
        self._save()
        return f"已存入长期记忆 [id:{entry['id']}]: {content}"

    def search(self, query: str, top_k: int = 3) -> str:
        """关键词检索，返回最相关的 top_k 条记忆。"""
        if not self._store:
            return "长期记忆为空。"

        keywords = set(query.lower().split())

        def score(entry: dict) -> int:
            text = entry["content"].lower()
            return sum(1 for kw in keywords if kw in text)

        ranked = sorted(self._store, key=score, reverse=True)[:top_k]
        if not any(score(e) > 0 for e in ranked):
            # 无关键词命中时，返回最近的记录
            ranked = self._store[-top_k:]

        lines = [f"[{e['created_at']}] {e['content']}" for e in ranked]
        return "\n".join(lines)

    def get_all_summary(self, max_entries: int = 5) -> str:
        """返回最近几条记忆，用于注入 system prompt。"""
        if not self._store:
            return ""
        recent = self._store[-max_entries:]
        lines = [f"- [{e['created_at']}] {e['content']}" for e in recent]
        return "\n".join(lines)

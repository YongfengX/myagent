from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class MemoryManager:
    """
    统一管理短期记忆和长期记忆。
    - 短期：当前会话滑动窗口
    - 长期：跨会话持久化 JSON 文件
    """

    def __init__(self, max_turns: int = 10):
        self.short = ShortTermMemory(max_turns=max_turns)
        self.long = LongTermMemory()

    def set_system(self, content: str):
        """设置 system prompt，自动附加长期记忆摘要。"""
        summary = self.long.get_all_summary()
        if summary:
            content += f"\n\n【长期记忆（最近）】\n{summary}"
        self.short.set_system(content)

    def add(self, message: dict):
        self.short.add(message)

    def get_messages(self) -> list[dict]:
        return self.short.get_messages()

    def save_to_long_term(self, content: str) -> str:
        return self.long.save(content)

    def search_long_term(self, query: str) -> str:
        return self.long.search(query)

    def clear_session(self):
        self.short.clear()

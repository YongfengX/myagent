class ShortTermMemory:
    """
    滑动窗口短期记忆，保留最近 max_turns 轮对话。
    system prompt 始终保留，不计入窗口。
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._system: dict | None = None
        self._turns: list[dict] = []  # 每个元素是 [user_msg, assistant_msg]

    def set_system(self, content: str):
        self._system = {"role": "system", "content": content}

    def add(self, message: dict):
        role = message["role"]
        if role == "system":
            self._system = message
            return
        if role == "user":
            self._turns.append([message])
        elif role == "assistant" and self._turns:
            self._turns[-1].append(message)
        # 超出窗口时丢弃最旧的一轮
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get_messages(self) -> list[dict]:
        messages = []
        if self._system:
            messages.append(self._system)
        for turn in self._turns:
            messages.extend(turn)
        return messages

    def clear(self):
        self._turns.clear()

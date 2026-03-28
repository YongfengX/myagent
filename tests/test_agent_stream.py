import re
from unittest.mock import MagicMock, patch
from memory.manager import MemoryManager


def make_mock_memory():
    """创建一个不依赖 ChromaDB 的 mock MemoryManager。"""
    mem = MagicMock(spec=MemoryManager)
    mem.get_messages.return_value = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
    ]
    mem.save_to_long_term = MagicMock(return_value="已存入")
    mem.search_long_term = MagicMock(return_value="搜索结果")
    return mem


def test_run_agent_uses_passed_memory():
    """传入 memory 时，run_agent 应使用传入的实例而非全局 _memory。"""
    from agent import run_agent

    mock_mem = make_mock_memory()

    with patch("agent._chat") as mock_chat:
        mock_chat.return_value = "Thought: done\nFinal Answer: 42"
        result = run_agent("test input", memory=mock_mem)

    # 应该在传入的 mock_mem 上调用 get_messages
    mock_mem.get_messages.assert_called()
    assert result == "42"


def test_run_agent_memory_tools_bound_to_passed_memory():
    """传入 memory 时，save_memory 工具应绑定到该实例。"""
    from agent import run_agent

    mock_mem = make_mock_memory()

    with patch("agent._chat") as mock_chat:
        # 第一步调用 save_memory，第二步给出 Final Answer
        mock_chat.side_effect = [
            "Thought: save\nAction: save_memory\nAction Input: test content",
            "Thought: done\nFinal Answer: saved",
        ]
        result = run_agent("save something", memory=mock_mem)

    # save_to_long_term 应在传入的 mock_mem 上被调用
    mock_mem.save_to_long_term.assert_called_once_with("test content")
    assert result == "saved"


def test_run_agent_stream_yields_steps():
    """stream=True 时应 yield thought/action/observation/answer 步骤。"""
    from agent import run_agent

    mock_mem = make_mock_memory()

    with patch("agent._chat") as mock_chat:
        mock_chat.side_effect = [
            "Thought: 需要计算\nAction: calculator\nAction Input: 1+1",
            "Thought: 得到结果\nFinal Answer: 2",
        ]
        steps = list(run_agent("1+1 等于多少", memory=mock_mem, stream=True))

    types = [s["type"] for s in steps]
    assert "thought" in types
    assert "action" in types
    assert "observation" in types
    assert "answer" in types
    # 最后一步是 answer
    assert steps[-1]["type"] == "answer"
    assert steps[-1]["content"] == "2"


def test_run_agent_stream_false_returns_string():
    """stream=False（默认）应返回字符串而非 generator。"""
    from agent import run_agent

    mock_mem = make_mock_memory()

    with patch("agent._chat") as mock_chat:
        mock_chat.return_value = "Thought: simple\nFinal Answer: hello"
        result = run_agent("hello", memory=mock_mem, stream=False)

    assert isinstance(result, str)
    assert result == "hello"

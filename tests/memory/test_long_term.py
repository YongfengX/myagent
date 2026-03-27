import pytest
import tempfile
import os
from memory.long_term import LongTermMemory


@pytest.fixture
def mem(tmp_path):
    """每个测试用独立的临时目录，避免互相污染。"""
    return LongTermMemory(persist_dir=str(tmp_path / "chroma"))


def test_save_returns_confirmation(mem):
    result = mem.save("用户叫张三")
    assert "已存入长期记忆" in result
    assert "用户叫张三" in result


def test_search_empty_returns_message(mem):
    result = mem.search("任何查询")
    assert result == "长期记忆为空。"


def test_search_finds_semantically_related(mem):
    mem.save("用户叫张三")
    mem.save("用户喜欢吃火锅")
    result = mem.search("这个人叫什么名字")
    assert "张三" in result


def test_search_returns_top_k(mem):
    for i in range(5):
        mem.save(f"记忆条目 {i}")
    result = mem.search("记忆条目", top_k=2)
    lines = [l for l in result.strip().split("\n") if l]
    assert len(lines) == 2


def test_get_all_summary_empty(mem):
    result = mem.get_all_summary()
    assert result == ""


def test_get_all_summary_returns_recent(mem):
    mem.save("第一条记忆")
    mem.save("第二条记忆")
    mem.save("第三条记忆")
    result = mem.get_all_summary(max_entries=2)
    assert "第二条记忆" in result
    assert "第三条记忆" in result
    assert "第一条记忆" not in result


def test_get_all_summary_format(mem):
    mem.save("测试内容")
    result = mem.get_all_summary()
    assert result.startswith("- [")
    assert "测试内容" in result

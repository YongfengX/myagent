# RAG Long-Term Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace keyword-based long-term memory retrieval with RAG (vector semantic search) using sentence-transformers + ChromaDB.

**Architecture:** Rewrite `memory/long_term.py` to use a local `SentenceTransformer` model for encoding text into embeddings, and ChromaDB as a persistent vector store. All other files remain unchanged — the public interface (`save`, `search`, `get_all_summary`) stays identical.

**Tech Stack:** `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2), `chromadb` (PersistentClient), `uv` for dependency management, `pytest` for tests.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add new dependencies |
| Rewrite | `memory/long_term.py` | Vector-based save/search/summary |
| Create | `tests/memory/test_long_term.py` | Unit tests for LongTermMemory |

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add chromadb and sentence-transformers to pyproject.toml**

Edit the `dependencies` list in `pyproject.toml` to:

```toml
[project]
name = "myagent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.26.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
]
```

- [ ] **Step 2: Install dependencies**

```bash
uv sync
```

Expected: no errors, `chromadb` and `sentence-transformers` appear in `.venv`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add chromadb and sentence-transformers dependencies"
```

---

### Task 2: Write failing tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/memory/__init__.py`
- Create: `tests/memory/test_long_term.py`

- [ ] **Step 1: Create test directory structure**

```bash
mkdir -p tests/memory
touch tests/__init__.py tests/memory/__init__.py
```

- [ ] **Step 2: Write test file**

Create `tests/memory/test_long_term.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/memory/test_long_term.py -v
```

Expected: `ImportError` or `TypeError` — `LongTermMemory` 的构造函数还不接受 `persist_dir` 参数。

---

### Task 3: Rewrite LongTermMemory

**Files:**
- Rewrite: `memory/long_term.py`

- [ ] **Step 1: Replace the entire file content**

```python
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
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/memory/test_long_term.py -v
```

Expected: 首次运行时 sentence-transformers 会下载模型（约 120MB），之后缓存到 `~/.cache/`。所有 7 个测试应全部 PASS。

- [ ] **Step 3: Commit**

```bash
git add memory/long_term.py tests/memory/test_long_term.py tests/__init__.py tests/memory/__init__.py
git commit -m "feat: replace keyword search with RAG vector memory (ChromaDB + sentence-transformers)"
```

---

### Task 4: Cleanup & smoke test

**Files:**
- Delete: `memory/long_term_store.json`（如存在）

- [ ] **Step 1: Remove old JSON store if it exists**

```bash
rm -f memory/long_term_store.json
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Smoke test the agent**

```bash
uv run python agent.py
```

在 CLI 中输入：
```
你: 我叫李四，记住这个信息
```
期望：agent 调用 `save_memory`，返回确认信息（含"已存入长期记忆"）。

再输入：
```
你: 我叫什么名字？
```
期望：agent 调用 `search_memory`，返回含"李四"的结果。

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: remove legacy long_term_store.json"
```

# RAG 长期记忆设计文档

**日期：** 2026-03-26
**状态：** 待实现

## 背景

当前长期记忆（`memory/long_term.py`）使用 JSON 文件存储，关键词匹配检索。当记忆条目增多时，关键词检索无法捕捉语义相似性（例如"用户叫什么名字"无法匹配"用户叫张三"）。

本设计将长期记忆的检索方式替换为 RAG（向量语义检索），提升记忆召回的相关性。

## 目标

- 用向量语义检索替换关键词匹配
- 对外接口保持不变，`MemoryManager` 和 `agent.py` 零改动
- 旧 JSON 数据丢弃，从空库开始

## 不在范围内

- 短期记忆不变（仍为滑动窗口）
- `get_all_summary()` 行为不变（仍按时间取最近 N 条）
- 不做向量去重、记忆老化、自动压缩

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 本地运行，支持中英文，轻量 |
| 向量数据库 | ChromaDB（PersistentClient） | 轻量本地持久化，API 简单 |

## 架构

### 文件改动

```
memory/
├── long_term.py          ← 完全重写（唯一改动文件）
├── short_term.py         ← 不动
├── manager.py            ← 不动
└── chroma_store/         ← ChromaDB 运行时自动创建
```

### 新依赖

```toml
# pyproject.toml 新增
chromadb
sentence-transformers
```

## 接口（不变）

```python
class LongTermMemory:
    def save(self, content: str) -> str
        # 文本 → embedding → 存入 ChromaDB
        # 返回: "已存入长期记忆 [id:xxxxxxxx]: <content>"

    def search(self, query: str, top_k: int = 3) -> str
        # query → embedding → ChromaDB 语义检索
        # 返回: 最相关的 top_k 条记忆文本（换行分隔）
        # 空库时返回: "长期记忆为空。"

    def get_all_summary(self, max_entries: int = 5) -> str
        # 按写入时间取最近 max_entries 条
        # 返回: "- [时间] 内容" 格式，空库返回 ""
```

## 数据流

```
save("用户叫张三"):
  text → SentenceTransformer.encode() → vector
  → ChromaDB.add(id=uuid, embedding=vector, document=text, metadata={created_at})

search("这个用户叫什么名字"):
  query → SentenceTransformer.encode() → vector
  → ChromaDB.query(query_embeddings=vector, n_results=3)
  → 返回 top-3 文档文本
```

## 错误处理

| 场景 | 处理方式 |
|------|---------|
| 首次启动，模型未缓存 | sentence-transformers 自动下载到 `~/.cache/`，首次慢属正常 |
| `search()` 时库为空 | 返回 `"长期记忆为空。"` |
| ChromaDB 写入失败 | 抛出异常，不静默吞掉 |

## 实现步骤

1. `pyproject.toml` 添加 `chromadb`、`sentence-transformers` 依赖
2. 重写 `memory/long_term.py`：
   - `__init__`：初始化 SentenceTransformer + ChromaDB PersistentClient
   - `save()`：encode → chroma.add
   - `search()`：encode → chroma.query → 格式化返回
   - `get_all_summary()`：chroma.get 按 metadata 时间排序取最近 N 条
3. 删除 `memory/long_term_store.json`（如存在）
4. 运行手动测试验证 save/search 语义匹配正确

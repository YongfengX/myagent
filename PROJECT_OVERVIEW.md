# MyAgent 项目文档

> 一个基于 ReAct 框架的 AI Agent，具备双层记忆系统和可扩展工具链。

---

## 目录

1. [项目概览](#1-项目概览)
2. [已实现功能](#2-已实现功能)
3. [技术栈](#3-技术栈)
4. [核心模块说明](#4-核心模块说明)
5. [问题处理 Pipeline](#5-问题处理-pipeline)
6. [完整示例：一问到底](#6-完整示例一问到底)

---

## 1. 项目概览

```
myagent/
├── agent.py               # 主入口 + ReAct 循环
├── api/
│   ├── engine.py          # API 调度器
│   └── qwen.py            # Qwen 实现
├── memory/
│   ├── manager.py         # 记忆统一入口
│   ├── short_term.py      # 短期记忆（滑动窗口）
│   ├── long_term.py       # 长期记忆（RAG 向量检索）
│   └── chroma_store/      # ChromaDB 持久化数据
├── tools/
│   ├── base_tools.py      # 内置工具（计算器、时间、记忆）
│   └── mcp_loader.py      # MCP 工具加载器
├── tests/
│   └── memory/
│       └── test_long_term.py
├── mcp_servers.json       # MCP 服务器配置
├── pyproject.toml         # 依赖与项目元数据
└── .env                   # API 密钥
```

---

## 2. 已实现功能

### 2.1 ReAct 智能循环

Agent 采用 **ReAct（Reason + Act）** 框架，每轮迭代分四步：

```
Thought → Action → Observation → (重复，直到 Final Answer)
```

- 最多迭代 10 步（`max_steps=10`）
- 每步 Agent 自行决定是否调用工具，或直接给出最终答案
- 通过解析 LLM 输出中的关键词（`Action:`、`Action Input:`、`Final Answer:`）驱动流程

### 2.2 工具系统

| 工具名 | 功能 | 实现位置 |
|--------|------|----------|
| `calculator` | 数学表达式计算 | `tools/base_tools.py` |
| `get_current_time` | 获取当前时间 | `tools/base_tools.py` |
| `save_memory` | 保存信息到长期记忆 | 绑定到 `memory/long_term.py` |
| `search_memory` | 语义检索长期记忆 | 绑定到 `memory/long_term.py` |
| MCP 工具 | 动态加载外部工具 | `tools/mcp_loader.py` |

**工具注册方式**：每个工具以 `{"func": callable, "description": str, "params": str}` 的结构注册到全局 `TOOLS` 字典，LLM 在系统提示中获知所有可用工具及其参数说明。

### 2.3 双层记忆系统

#### 短期记忆（Short-Term Memory）

- **结构**：滑动窗口对话历史
- **容量**：默认保留最近 10 轮（系统提示始终保留）
- **作用**：为当前对话提供上下文，注入 LLM 的 `messages` 列表

#### 长期记忆（Long-Term Memory，RAG）

- **结构**：向量数据库 + 语义检索
- **编码模型**：`paraphrase-multilingual-MiniLM-L12-v2`（本地运行，支持中英文）
- **存储**：ChromaDB 持久化到 `memory/chroma_store/`
- **检索**：余弦相似度语义搜索，返回 top-k 最相关条目
- **注入方式**：每轮对话前，将最近 5 条长期记忆摘要追加到系统提示

### 2.4 MCP 工具集成

通过 `mcp_servers.json` 配置 MCP 服务器，动态加载第三方工具（如文件系统访问）：

```json
[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["@modelcontextprotocol/server-filesystem", "/tmp"]
  }
]
```

MCP 工具在启动时异步加载，自动注册到 `TOOLS` 字典，Agent 可无缝调用。

### 2.5 多 API 抽象

- 通过 `api/engine.py` 统一调度，当前接入 **Qwen（阿里云灵积）**
- 使用 OpenAI 兼容接口（`openai` 库），切换其他 API 只需新增适配文件

### 2.6 CLI 交互界面

`agent.py` 底部的 `main()` 函数提供交互式命令行：

```
你: 今天是几号？
Agent: ...
```

---

## 3. 技术栈

| 层次 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **运行时** | Python | ≥3.11 | 主语言 |
| **包管理** | uv | - | 依赖管理与锁定 |
| **LLM API** | Qwen（DashScope） | qwen-plus | 核心推理模型 |
| **API 客户端** | openai | ≥1.0.0 | OpenAI 兼容客户端 |
| **向量数据库** | ChromaDB | ≥0.5.0 | 长期记忆持久化存储 |
| **语义编码** | sentence-transformers | ≥3.0.0 | 文本向量化（本地） |
| **工具协议** | MCP SDK | ≥1.26.0 | 外部工具动态加载 |
| **环境变量** | python-dotenv | ≥1.0.0 | API 密钥管理 |
| **测试** | pytest | ≥9.0.2 | 单元测试 |

**环境变量**（`.env`）：
```
DASHSCOPE_API_KEY=<阿里云 API 密钥>
```

---

## 4. 核心模块说明

### `agent.py` — ReAct 主循环

```python
API   = "qwen"       # 选择 API 提供商
MODEL = "qwen-plus"  # 模型名称
```

- 构造包含工具描述的系统提示
- 将长期记忆摘要注入系统提示
- 驱动 ReAct 循环，解析 LLM 输出

### `memory/short_term.py` — 滑动窗口

```python
class ShortTermMemory:
    def add_turn(user_msg, assistant_msg)  # 添加一轮对话
    def get_messages(system_prompt)        # 返回 [system, ...最近N轮]
```

### `memory/long_term.py` — RAG 向量记忆

```python
class LongTermMemory:
    def save(content: str) -> str          # 向量化并存入 ChromaDB
    def search(query: str, top_k=3) -> str # 语义检索
    def get_all_summary(max_entries=5) -> str  # 摘要（注入系统提示用）
```

**存储结构**（ChromaDB collection: `long_term_memory`）：
- `documents`: 原始文本内容
- `metadatas`: `{"timestamp": "2026-03-27 13:40:00"}`
- `ids`: UUID

### `tools/mcp_loader.py` — MCP 适配器

异步连接 MCP 服务器，将其工具 schema 转换为 myagent 的 `TOOLS` 格式，并用同步包装器封装异步调用。

---

## 5. 问题处理 Pipeline

以下是用户输入从进入系统到输出答案的完整流程：

```
┌─────────────────────────────────────────────────────────┐
│                      用户输入                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 0: 初始化（每次对话前）                              │
│  • 加载 MCP 工具 → 注册到 TOOLS 字典                      │
│  • 构建系统提示（工具描述 + 格式规范）                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: 记忆注入                                         │
│  • 从 ChromaDB 取最近 5 条长期记忆                        │
│  • 格式化为【长期记忆】段落追加到系统提示                  │
│  • 将用户消息加入短期记忆（滑动窗口）                      │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────────┐
          │         ReAct 循环（最多 10 步）  │
          │                                  │
          │  ┌────────────────────────────┐  │
          │  │  Step 2: 构建 messages      │  │
          │  │  [system, ...最近N轮对话]   │  │
          │  └────────────┬───────────────┘  │
          │               │                  │
          │               ▼                  │
          │  ┌────────────────────────────┐  │
          │  │  Step 3: 调用 LLM API       │  │
          │  │  qwen.chat(messages, model) │  │
          │  │  → DashScope HTTP 请求      │  │
          │  └────────────┬───────────────┘  │
          │               │                  │
          │               ▼                  │
          │  ┌────────────────────────────┐  │
          │  │  Step 4: 解析 LLM 输出      │  │
          │  │                            │  │
          │  │  含 "Final Answer:" ?       │  │
          │  │    ├─ YES → 返回答案，结束  │  │
          │  │    └─ NO  ↓                │  │
          │  │                            │  │
          │  │  含 "Action:" ?             │  │
          │  │    ├─ YES → 提取工具名      │  │
          │  │    └─ NO  → 返回当前内容   │  │
          │  └────────────┬───────────────┘  │
          │               │ (有 Action)       │
          │               ▼                  │
          │  ┌────────────────────────────┐  │
          │  │  Step 5: 执行工具           │  │
          │  │                            │  │
          │  │  TOOLS[tool_name]["func"]  │  │
          │  │  (tool_input)              │  │
          │  │                            │  │
          │  │  可能调用：                 │  │
          │  │  • calculator(expr)        │  │
          │  │  • get_current_time()      │  │
          │  │  • save_memory(content)    │  │
          │  │    └→ SentenceTransformer  │  │
          │  │       encode → ChromaDB    │  │
          │  │  • search_memory(query)    │  │
          │  │    └→ encode → ChromaDB    │  │
          │  │       semantic search      │  │
          │  │  • MCP tool(params)        │  │
          │  │    └→ MCP 服务器           │  │
          │  └────────────┬───────────────┘  │
          │               │                  │
          │               ▼                  │
          │  ┌────────────────────────────┐  │
          │  │  Step 6: 记录 Observation   │  │
          │  │  将工具结果追加到短期记忆   │  │
          │  │  格式："Observation: <结果>"│  │
          │  └────────────┬───────────────┘  │
          │               │                  │
          │               └── 返回 Step 2    │
          │                                  │
          └──────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      输出最终答案                          │
└─────────────────────────────────────────────────────────┘
```

### Pipeline 各阶段详解

| 阶段 | 位置 | 输入 | 输出 |
|------|------|------|------|
| 初始化 | `agent.py:main()` | `mcp_servers.json` | 注册 TOOLS |
| 记忆注入 | `memory/manager.py` | 长期记忆 DB | 带记忆的系统提示 |
| 构建消息 | `memory/short_term.py` | 对话历史 | `messages` 列表 |
| LLM 调用 | `api/qwen.py` | `messages` | 文本响应 |
| 输出解析 | `agent.py:run_agent()` | LLM 文本 | 工具名 + 参数 |
| 工具执行 | `tools/base_tools.py` / MCP | 参数 | 工具结果字符串 |
| 观察记录 | `memory/short_term.py` | 工具结果 | 更新对话历史 |

---

## 6. 完整示例：一问到底

**场景**：用户上次聊天时 Agent 记住了"用户叫爱丽丝"，这次用户问"你记得我叫什么吗？"

```
用户输入: "你记得我叫什么吗？"
         │
         ▼
[Step 1] 记忆注入
  长期记忆摘要 → 系统提示追加：
  【长期记忆】
  - [2026-03-26 10:00] 用户叫爱丽丝

  短期记忆 add: {"role": "user", "content": "你记得我叫什么吗？"}
         │
         ▼
[Step 2] 构建 messages
  [
    {"role": "system", "content": "你是 ReAct 助手...【长期记忆】..."},
    {"role": "user",   "content": "你记得我叫什么吗？"}
  ]
         │
         ▼
[Step 3] 调用 Qwen API
  LLM 输出：
    "Thought: 用户问我是否记得他的名字，我应该搜索长期记忆确认。
     Action: search_memory
     Action Input: 用户的名字"
         │
         ▼
[Step 4] 解析输出
  tool_name  = "search_memory"
  tool_input = "用户的名字"
         │
         ▼
[Step 5] 执行 search_memory("用户的名字")
  → SentenceTransformer.encode("用户的名字") → 向量 [0.12, -0.34, ...]
  → ChromaDB 余弦相似度检索
  → 返回: "- [2026-03-26 10:00] 用户叫爱丽丝"
         │
         ▼
[Step 6] 记录 Observation
  短期记忆 add: {"role": "user", "content": "Observation: - [2026-03-26 10:00] 用户叫爱丽丝"}
         │
         ▼
[Step 2] 再次构建 messages（包含 Observation）
         │
         ▼
[Step 3] 再次调用 Qwen API
  LLM 输出：
    "Final Answer: 记得！你叫爱丽丝。"
         │
         ▼
输出: "记得！你叫爱丽丝。"
```

---

## 附录：扩展指南

| 想要 | 做什么 |
|------|--------|
| 接入 OpenAI | 新建 `api/openai.py`，在 `api/engine.py` 注册 |
| 添加新工具 | 在 `tools/base_tools.py` 定义函数，加入 `TOOLS` 字典 |
| 接入新 MCP 服务 | 在 `mcp_servers.json` 添加服务器配置 |
| 换模型 | 修改 `agent.py` 第 8 行 `MODEL = "..."` |
| 调整记忆窗口 | 修改 `MemoryManager` 初始化参数 `max_turns` |
| 运行测试 | `uv run pytest tests/ -v` |

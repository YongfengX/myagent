# Gradio 前端设计规格

**日期：** 2026-03-27
**状态：** 已批准

---

## Context

MyAgent 目前是纯 CLI 交互，没有图形界面。用户希望为其添加一个类 ChatGPT 风格的本地前端，方便日常使用，同时能查看 Agent 的推理过程和长期记忆。选择 Gradio 是因为用户以 Python 为主，不希望引入 JS 框架。

---

## 技术选型

- **框架：** Gradio（`gr.Blocks`）
- **部署：** 本地运行，无需公网访问
- **入口文件：** 新增 `app.py`（项目根目录）

---

## 功能范围

1. 聊天对话（含 Markdown 渲染）
2. 实时显示 ReAct 推理过程（Thought / Action / Observation）
3. 长期记忆查看与语义搜索
4. 多会话切换（内存级，重启后消失，长期记忆持久）

---

## 架构

### 文件变更

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `app.py` | 新增 | Gradio UI 入口，session 管理 |
| `agent.py` | 改造 | `run_agent()` 支持 `stream=True` generator 模式 |
| `memory/` | 不变 | 复用现有 MemoryManager |
| `tools/` | 不变 | 复用现有工具 |
| `api/` | 不变 | 复用现有 API 层 |

### Session 管理

```python
# app.py 内维护
sessions: dict[str, MemoryManager] = {}  # session_id -> MemoryManager
```

Session 以内存 dict 管理，切换时从 `MemoryManager.short_term` 重建对话历史。长期记忆（ChromaDB）跨 session 持久。

### agent.py 改造

`run_agent()` 增加两个参数：

```python
def run_agent(user_input: str, max_steps: int = 10,
              memory: MemoryManager = None, stream: bool = False):
```

- `memory=None` 时回退到模块级全局 `memory`（CLI 模式不变）
- `stream=False`（默认）：保持现有行为，CLI 正常使用
- `stream=True`：变为 generator，每步 yield：
  ```python
  {"type": "thought" | "action" | "observation" | "answer", "content": str}
  ```

**关键：** 当传入 `memory` 参数时，函数内部需创建一份 TOOLS 的局部副本，将 `save_memory` / `search_memory` 重新绑定到传入的 memory 实例，避免污染全局状态：

```python
local_tools = {**TOOLS}
local_tools["save_memory"]["func"] = memory.save_to_long_term
local_tools["search_memory"]["func"] = memory.search_long_term
```

---

## Gradio 组件结构

```
gr.Blocks
└── gr.Row
    ├── 左列（宽度比 1）：会话管理
    │   ├── gr.Radio        — 会话列表（点击切换）
    │   └── gr.Button       — "+ 新建会话"
    │
    ├── 中列（宽度比 3）：主交互区
    │   ├── gr.Chatbot      — 对话历史，支持 Markdown
    │   ├── gr.Accordion    — "🔍 推理过程"（默认折叠）
    │   │   └── gr.Textbox  — 实时推理步骤文本
    │   └── gr.Row
    │       ├── gr.Textbox  — 消息输入框
    │       └── gr.Button   — "发送"
    │
    └── 右列（宽度比 2）：长期记忆
        ├── gr.Textbox      — 搜索框
        ├── gr.Button       — "搜索"
        ├── gr.Button       — "刷新"
        └── gr.Textbox      — 记忆条目显示（只读，多行）
```

---

## 数据流

### 发送消息

```python
def chat_handler(message, history, session_id):
    reasoning = ""
    for step in run_agent(message, memory=sessions[session_id], stream=True):
        if step["type"] in ("thought", "action", "observation"):
            reasoning += f"[{step['type'].upper()}] {step['content']}\n"
            yield history, reasoning       # 更新推理面板
        elif step["type"] == "answer":
            history.append((message, step["content"]))
            yield history, reasoning       # 更新对话 + 推理面板
```

### 切换 Session

1. 从 `sessions[session_id].short_term.get_history()` 重建 `gr.Chatbot` 历史
2. 清空推理面板
3. 刷新右侧记忆面板

### 记忆搜索

调用 `sessions[session_id].long_term.search(query, top_k=5)`，格式化结果显示在右侧面板。

---

## 不在范围内（刻意保持简单）

- 会话持久化（重启后消失）
- 错误弹窗（错误显示在聊天区）
- 并发支持（单用户本地，Gradio 默认行为即可）
- 删除会话功能
- 导出对话记录

---

## 验证方式

1. `uv run python app.py` 启动，浏览器打开 `http://localhost:7860`
2. 发送消息，确认对话正常显示
3. 展开"推理过程"，确认 Thought/Action/Observation 实时出现
4. 点击"刷新"查看长期记忆，确认 ChromaDB 条目正确显示
5. 新建第二个会话，切换回第一个，确认历史记录正确恢复
6. 在记忆搜索框输入关键词，确认语义搜索返回相关条目
7. 原有 `uv run python agent.py` CLI 模式不受影响

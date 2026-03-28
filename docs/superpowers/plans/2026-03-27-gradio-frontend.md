# Gradio 前端实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 MyAgent 添加 Gradio 本地前端，支持聊天、实时推理过程、长期记忆查看与搜索、多会话切换。

**Architecture:** 新增 `app.py` 作为 Gradio 入口，改造 `agent.py` 的 `run_agent()` 使其支持 generator 流式输出模式。Sessions 以内存 dict 管理，长期记忆通过 ChromaDB 跨 session 持久。

**Tech Stack:** Gradio ≥4.0, Python 3.11+, 复用现有 MemoryManager / LongTermMemory / TOOLS

---

## 文件变更总览

| 文件 | 类型 | 职责 |
|------|------|------|
| `pyproject.toml` | 修改 | 添加 gradio 依赖 |
| `agent.py` | 修改 | `run_agent()` 支持 `memory` 参数和 `stream=True` |
| `app.py` | 新增 | Gradio Blocks UI 入口，session 管理，事件绑定 |
| `tests/test_agent_stream.py` | 新增 | 验证 run_agent stream 模式行为 |

---

## Task 1: 添加 gradio 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 在 pyproject.toml 的 dependencies 中添加 gradio**

将 `pyproject.toml` 中的 `dependencies` 改为：

```toml
dependencies = [
    "mcp>=1.26.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "gradio>=4.0.0",
]
```

- [ ] **Step 2: 安装依赖**

```bash
uv sync
```

Expected: 输出安装进度，最后一行包含 `gradio`，无报错。

- [ ] **Step 3: 验证 gradio 可导入**

```bash
uv run python -c "import gradio; print(gradio.__version__)"
```

Expected: 打印版本号如 `4.x.x`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add gradio dependency"
```

---

## Task 2: 重构 run_agent() 支持 memory 参数

**Files:**
- Modify: `agent.py`
- Create: `tests/test_agent_stream.py`

目标：将模块级 `memory` 全局变量改名为 `_memory`，让 `run_agent()` 接受可选的 `memory` 参数，传入时使用该实例，否则回退到 `_memory`。同时处理 TOOLS 中内存工具的局部绑定。

- [ ] **Step 1: 先写失败的测试**

新建 `tests/test_agent_stream.py`：

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_agent_stream.py::test_run_agent_uses_passed_memory tests/test_agent_stream.py::test_run_agent_memory_tools_bound_to_passed_memory -v
```

Expected: FAILED（`run_agent` 尚不支持 `memory` 参数）

- [ ] **Step 3: 修改 agent.py**

将 `agent.py` 改为以下内容（主要变动：全局 `memory` 改为 `_memory`，`run_agent` 新增 `memory` 参数，内部使用局部 TOOLS 副本）：

```python
import re
from api.engine import get_chat_fn
from tools.base_tools import TOOLS
from tools.mcp_loader import load_mcp_tools
from memory.manager import MemoryManager

# ── 在这里选择 API 和模型 ──────────────────────────────────────────────────────
API   = "qwen"       # 可选: "qwen" | 后续扩展: "openai" ...
MODEL = "qwen-plus"  # 对应 API 下的模型名
# ─────────────────────────────────────────────────────────────────────────────

_chat = get_chat_fn(API)
_memory = MemoryManager(max_turns=10)

# 加载 MCP Server 工具（mcp_servers.json 中配置）
TOOLS.update(load_mcp_tools())

# 将记忆工具的 func 绑定到全局 _memory（CLI 默认）
TOOLS["save_memory"]["func"] = _memory.save_to_long_term
TOOLS["search_memory"]["func"] = _memory.search_long_term


# ── Prompt ─────────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    tool_lines = "\n".join(
        f"- {name}: {info['description']}（参数：{info['params']}）"
        for name, info in TOOLS.items()
    )
    return f"""你是一个 ReAct 式 AI 助手，通过"思考→行动→观察"循环来解决问题。
你拥有长期记忆和短期记忆：
- 短期记忆：当前对话窗口（最近 10 轮）
- 长期记忆：跨会话持久存储，可用 save_memory / search_memory 工具操作

可用工具：
{tool_lines}

严格按照以下格式输出，每次只做一个步骤：

需要使用工具时：
Thought: <你的思考>
Action: <工具名>
Action Input: <工具输入>

得到最终答案时：
Thought: <你的思考>
Final Answer: <最终答案>

规则：
- Action 必须是工具列表中的名称之一
- 每次响应只能包含 Action 或 Final Answer，不能同时出现
- 如果不需要工具，直接给出 Final Answer
- 遇到值得记住的用户信息（姓名、偏好、重要事项等），主动调用 save_memory 保存"""


# ── 内部 generator（核心 ReAct 循环）──────────────────────────────────────────

def _run_agent_gen(user_input: str, max_steps: int, mem: MemoryManager):
    """
    核心 ReAct 循环，始终作为 generator 运行。
    每步 yield {"type": "thought"|"action"|"observation"|"answer", "content": str}
    """
    # 构建局部 TOOLS，将记忆工具绑定到传入的 mem 实例
    local_tools = {**TOOLS}
    local_tools["save_memory"] = {
        **TOOLS["save_memory"],
        "func": mem.save_to_long_term,
    }
    local_tools["search_memory"] = {
        **TOOLS["search_memory"],
        "func": mem.search_long_term,
    }

    mem.set_system(build_system_prompt())
    mem.add({"role": "user", "content": user_input})

    for step in range(max_steps):
        messages = mem.get_messages()
        content = _chat(messages, model=MODEL)
        mem.add({"role": "assistant", "content": content})

        # 提取 Thought
        thought_match = re.search(
            r"Thought[:：](.*?)(?=Action:|Final Answer:|$)", content, re.DOTALL
        )
        if thought_match:
            yield {"type": "thought", "content": thought_match.group(1).strip()}

        # Final Answer
        if "Final Answer:" in content:
            match = re.search(r"Final Answer[:：](.*)", content, re.DOTALL)
            answer = match.group(1).strip() if match else content
            yield {"type": "answer", "content": answer}
            return

        # Action
        action_match = re.search(r"Action[:：]\s*(\w+)", content)
        input_match = re.search(r"Action Input[:：](.*)", content, re.DOTALL)

        if not action_match:
            yield {"type": "answer", "content": content}
            return

        tool_name = action_match.group(1).strip()
        tool_input = input_match.group(1).strip() if input_match else ""

        yield {"type": "action", "content": f"{tool_name}({tool_input})"}

        if tool_name not in local_tools:
            observation = f"错误：工具 '{tool_name}' 不存在，可用工具：{list(local_tools.keys())}"
        else:
            observation = local_tools[tool_name]["func"](tool_input)

        yield {"type": "observation", "content": str(observation)}

        observation_msg = f"Observation: {observation}"
        mem.add({"role": "user", "content": observation_msg})

    yield {"type": "answer", "content": "达到最大步骤数，未能得到最终答案。"}


# ── ReAct Loop（公开接口）──────────────────────────────────────────────────────

def run_agent(user_input: str, max_steps: int = 10,
              memory: MemoryManager = None, stream: bool = False):
    """
    运行 ReAct Agent。
    - memory=None 时使用模块级全局 _memory（CLI 模式）
    - stream=False（默认）：阻塞式，返回最终答案字符串
    - stream=True：返回 generator，逐步 yield ReAct 步骤 dict
    """
    mem = memory if memory is not None else _memory
    gen = _run_agent_gen(user_input, max_steps, mem)

    if stream:
        return gen

    # 非 stream：消费 generator，返回最终答案
    answer = "（无答案）"
    for step in gen:
        if step["type"] == "answer":
            answer = step["content"]
        elif step["type"] in ("thought", "action", "observation"):
            print(f"\n[{step['type'].upper()}] {step['content']}")
    return answer


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"ReAct Agent 已启动 [API: {API} | Model: {MODEL}]，输入 'quit' 退出\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "退出"):
            break
        if not user_input:
            continue
        answer = run_agent(user_input)
        print(f"\n最终答案: {answer}\n{'─' * 50}")
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_agent_stream.py::test_run_agent_uses_passed_memory tests/test_agent_stream.py::test_run_agent_memory_tools_bound_to_passed_memory -v
```

Expected: 2 passed

- [ ] **Step 5: 确认 CLI 仍然正常（smoke test）**

不需要真正运行 agent，只需确认 import 无错误：

```bash
uv run python -c "from agent import run_agent; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent_stream.py
git commit -m "feat: refactor run_agent to accept memory param with local tool binding"
```

---

## Task 3: 为 run_agent 添加 stream=True 测试

**Files:**
- Modify: `tests/test_agent_stream.py`

- [ ] **Step 1: 追加 stream 模式测试**

在 `tests/test_agent_stream.py` 末尾添加：

```python
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
```

- [ ] **Step 2: 运行所有 stream 测试**

```bash
uv run pytest tests/test_agent_stream.py -v
```

Expected: 4 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent_stream.py
git commit -m "test: add stream mode tests for run_agent"
```

---

## Task 4: 创建 app.py —— 基础 Gradio 布局 + 单会话聊天

**Files:**
- Create: `app.py`

目标：先跑通基础聊天功能，不带 session 切换，验证 Gradio + agent 的联通性。

- [ ] **Step 1: 创建 app.py**

```python
import gradio as gr
from memory.manager import MemoryManager
from agent import run_agent, build_system_prompt

# ── Session 存储 ────────────────────────────────────────────────────────────────
# 结构: {session_name: {"memory": MemoryManager, "history": [(user, answer), ...]}}
sessions: dict[str, dict] = {}


def _new_session(name: str) -> dict:
    mem = MemoryManager(max_turns=10)
    return {"memory": mem, "history": []}


def _ensure_default_session():
    if "对话-1" not in sessions:
        sessions["对话-1"] = _new_session("对话-1")


_ensure_default_session()


# ── 事件处理函数 ────────────────────────────────────────────────────────────────

def chat_handler(message: str, history: list, session_name: str, reasoning: str):
    """处理用户发送消息，流式更新对话和推理面板。"""
    if not message.strip():
        yield history, reasoning, ""
        return
    if session_name not in sessions:
        sessions[session_name] = _new_session(session_name)

    mem = sessions[session_name]["memory"]
    current_reasoning = ""

    for step in run_agent(message, memory=mem, stream=True):
        if step["type"] in ("thought", "action", "observation"):
            current_reasoning += f"[{step['type'].upper()}] {step['content']}\n"
            yield history, current_reasoning, message
        elif step["type"] == "answer":
            new_history = history + [[message, step["content"]]]
            sessions[session_name]["history"] = new_history
            yield new_history, current_reasoning, ""


def get_memory_display(session_name: str) -> str:
    """返回当前 session 的长期记忆摘要。"""
    if session_name not in sessions:
        return "（无记忆）"
    mem = sessions[session_name]["memory"]
    summary = mem.long.get_all_summary(max_entries=20)
    return summary if summary else "（无记忆）"


def search_memory(query: str, session_name: str) -> str:
    """在长期记忆中语义搜索。"""
    if not query.strip():
        return get_memory_display(session_name)
    if session_name not in sessions:
        return "（无会话）"
    mem = sessions[session_name]["memory"]
    result = mem.search_long_term(query)
    return result if result else "（无结果）"


def new_session(sessions_list: list[str]) -> tuple:
    """新建会话，返回更新后的会话列表和新会话名。"""
    n = len(sessions_list) + 1
    name = f"对话-{n}"
    while name in sessions:
        n += 1
        name = f"对话-{n}"
    sessions[name] = _new_session(name)
    new_list = sessions_list + [name]
    # 返回顺序对应 outputs: session_radio, sessions_list_state, current_session, chatbot, reasoning_box, memory_display
    return gr.update(choices=new_list, value=name), new_list, name, [], "", "（无记忆）"


def switch_session(session_name: str) -> tuple:
    """切换会话，恢复对话历史和记忆面板，同时更新 current_session。"""
    if session_name not in sessions:
        sessions[session_name] = _new_session(session_name)
    history = sessions[session_name]["history"]
    memory_text = get_memory_display(session_name)
    # 返回顺序对应 outputs: chatbot, reasoning_box, memory_display, current_session
    return history, "", memory_text, session_name


# ── Gradio UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="MyAgent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 MyAgent")

    with gr.Row():
        # 左列：会话管理
        with gr.Column(scale=1, min_width=160):
            gr.Markdown("### 会话")
            session_radio = gr.Radio(
                choices=list(sessions.keys()),
                value="对话-1",
                label="",
                interactive=True,
            )
            new_session_btn = gr.Button("＋ 新建会话", size="sm")

        # 中列：主交互区
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=480, bubble_full_width=False)
            with gr.Accordion("🔍 推理过程", open=False):
                reasoning_box = gr.Textbox(
                    label="",
                    lines=6,
                    max_lines=12,
                    interactive=False,
                    placeholder="发送消息后，此处显示 Agent 的推理步骤...",
                )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="输入消息，按 Enter 发送...",
                    label="",
                    scale=4,
                    container=False,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

        # 右列：长期记忆
        with gr.Column(scale=2):
            gr.Markdown("### 🧠 长期记忆")
            memory_search = gr.Textbox(
                placeholder="搜索记忆...",
                label="",
                container=False,
            )
            with gr.Row():
                search_btn = gr.Button("搜索", size="sm")
                refresh_btn = gr.Button("刷新", size="sm")
            memory_display = gr.Textbox(
                value=get_memory_display("对话-1"),
                label="",
                lines=15,
                max_lines=30,
                interactive=False,
            )

    # 当前 session 名称（隐藏状态）
    current_session = gr.State("对话-1")
    sessions_list_state = gr.State(list(sessions.keys()))

    # ── 事件绑定 ────────────────────────────────────────────────────────────────

    # 发送消息（Enter 或点击按钮）
    send_inputs = [msg_input, chatbot, current_session, reasoning_box]
    send_outputs = [chatbot, reasoning_box, msg_input]

    msg_input.submit(chat_handler, inputs=send_inputs, outputs=send_outputs)
    send_btn.click(chat_handler, inputs=send_inputs, outputs=send_outputs)

    # 新建会话
    new_session_btn.click(
        new_session,
        inputs=[sessions_list_state],
        outputs=[session_radio, sessions_list_state, current_session, chatbot, reasoning_box, memory_display],
    )

    # 切换会话（合并为单个处理器）
    session_radio.change(
        switch_session,
        inputs=[session_radio],
        outputs=[chatbot, reasoning_box, memory_display, current_session],
    )

    # 刷新记忆
    refresh_btn.click(get_memory_display, inputs=[current_session], outputs=[memory_display])

    # 搜索记忆
    search_btn.click(search_memory, inputs=[memory_search, current_session], outputs=[memory_display])
    memory_search.submit(search_memory, inputs=[memory_search, current_session], outputs=[memory_display])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
```

- [ ] **Step 2: 启动 app，手动验证基础聊天**

```bash
uv run python app.py
```

Expected: 输出 `Running on local URL: http://127.0.0.1:7860`，浏览器打开后可看到三栏布局。

在输入框发送 `你好`，确认 chatbot 中出现回复。

- [ ] **Step 3: 验证推理过程面板**

展开"🔍 推理过程"，再发送一条需要工具的消息（如 `现在几点了`），确认推理面板实时出现 `[THOUGHT]`、`[ACTION]`、`[OBSERVATION]` 条目。

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Gradio frontend with 3-column layout and streaming ReAct display"
```

---

## Task 5: 验证多会话与记忆面板

**Files:** 无新文件，手动验证

- [ ] **Step 1: 验证新建会话**

点击"＋ 新建会话"，确认左侧出现"对话-2"且自动切换，chatbot 清空。

- [ ] **Step 2: 验证会话隔离**

在对话-2 发送 `你好`，切换回对话-1，确认对话-1 的历史记录正确恢复，推理面板清空。

- [ ] **Step 3: 验证长期记忆显示**

在任意会话发送 `请记住：我喜欢简洁的回答`，等 Agent 调用 save_memory 后，点击右侧"刷新"，确认记忆条目出现在面板中。

- [ ] **Step 4: 验证记忆搜索**

在搜索框输入 `简洁`，点击搜索，确认返回相关记忆条目。

- [ ] **Step 5: 验证 CLI 模式未受影响**

在另一个终端：

```bash
uv run python agent.py
```

输入 `你好`，确认正常返回答案，输入 `quit` 退出。

- [ ] **Step 6: 运行全部测试确认无回归**

```bash
uv run pytest tests/ -v
```

Expected: 全部 pass（包含 test_agent_stream.py 和已有 memory 测试）

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "feat: complete Gradio frontend with session management and memory panel"
```

---

## 验证清单（对应 spec）

| 验证项 | 对应 Task |
|--------|-----------|
| `uv run python app.py` 启动，浏览器打开 http://localhost:7860 | Task 4 Step 2 |
| 发送消息，对话正常显示 | Task 4 Step 2 |
| 推理过程面板实时显示 Thought/Action/Observation | Task 4 Step 3 |
| 刷新查看长期记忆，ChromaDB 条目正确显示 | Task 5 Step 3 |
| 新建第二个会话，切换回第一个，历史正确恢复 | Task 5 Step 1-2 |
| 记忆搜索框语义搜索返回相关条目 | Task 5 Step 4 |
| 原有 CLI 模式不受影响 | Task 5 Step 5 |

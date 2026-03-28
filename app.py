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

with gr.Blocks(title="MyAgent") as demo:
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
            chatbot = gr.Chatbot(label="对话", height=480)
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

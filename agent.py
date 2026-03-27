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
memory = MemoryManager(max_turns=10)

# 加载 MCP Server 工具（mcp_servers.json 中配置）
TOOLS.update(load_mcp_tools())

# 将记忆工具的 func 绑定到 memory 实例
TOOLS["save_memory"]["func"] = memory.save_to_long_term
TOOLS["search_memory"]["func"] = memory.search_long_term


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


# ── ReAct Loop ─────────────────────────────────────────────────────────────────

def run_agent(user_input: str, max_steps: int = 10) -> str:
    # 每轮对话重新设置 system（会自动附加长期记忆摘要）
    memory.set_system(build_system_prompt())
    memory.add({"role": "user", "content": user_input})

    for step in range(max_steps):
        messages = memory.get_messages()
        content = _chat(messages, model=MODEL)
        memory.add({"role": "assistant", "content": content})

        print(f"\n[Step {step + 1}]\n{content}")

        # Final Answer
        if "Final Answer:" in content:
            match = re.search(r"Final Answer[:：](.*)", content, re.DOTALL)
            return match.group(1).strip() if match else content

        # Action
        action_match = re.search(r"Action[:：]\s*(\w+)", content)
        input_match = re.search(r"Action Input[:：](.*)", content, re.DOTALL)

        if not action_match:
            return content

        tool_name = action_match.group(1).strip()
        tool_input = input_match.group(1).strip() if input_match else ""

        if tool_name not in TOOLS:
            observation = f"错误：工具 '{tool_name}' 不存在，可用工具：{list(TOOLS.keys())}"
        else:
            observation = TOOLS[tool_name]["func"](tool_input)

        observation_msg = f"Observation: {observation}"
        print(observation_msg)
        memory.add({"role": "user", "content": observation_msg})

    return "达到最大步骤数，未能得到最终答案。"


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

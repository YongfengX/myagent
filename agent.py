import re
from api.qwen import chat
from tools.base_tools import TOOLS


# ── Prompt ─────────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    tool_lines = "\n".join(
        f"- {name}: {info['description']}（参数：{info['params']}）"
        for name, info in TOOLS.items()
    )
    return f"""你是一个 ReAct 式 AI 助手，通过"思考→行动→观察"循环来解决问题。

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
- 如果不需要工具，直接给出 Final Answer"""


# ── ReAct Loop ─────────────────────────────────────────────────────────────────

def run_agent(user_input: str, max_steps: int = 10) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_input},
    ]

    for step in range(max_steps):
        content = chat(messages)
        messages.append({"role": "assistant", "content": content})

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
        messages.append({"role": "user", "content": observation_msg})

    return "达到最大步骤数，未能得到最终答案。"


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ReAct Agent 已启动，输入 'quit' 退出\n")
    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ("quit", "exit", "退出"):
            break
        if not user_input:
            continue
        answer = run_agent(user_input)
        print(f"\n最终答案: {answer}\n{'─' * 50}")

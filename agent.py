import os
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
)

# ── Tools ──────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算出错: {e}"


def get_current_time(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS: dict[str, dict] = {
    "calculator": {
        "func": calculator,
        "description": "计算数学表达式，返回结果",
        "params": "数学表达式，例如 '(2 + 3) * 4'",
    },
    "get_current_time": {
        "func": get_current_time,
        "description": "获取当前日期和时间",
        "params": "无需参数，传空字符串即可",
    },
}

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
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
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
            return content  # 模型没有按格式输出，直接返回

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

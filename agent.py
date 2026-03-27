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

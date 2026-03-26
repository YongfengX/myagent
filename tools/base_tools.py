from datetime import datetime


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

from api.qwen import chat as qwen_chat

# 注册表：api名称 → chat函数
# 新增 API 时在这里添加一行即可
_REGISTRY: dict[str, callable] = {
    "qwen": qwen_chat,
    # "openai": openai_chat,   # 示例：后续在 api/openai.py 实现后取消注释
}


def get_chat_fn(api: str):
    if api not in _REGISTRY:
        raise ValueError(f"未知 API: '{api}'，可选：{list(_REGISTRY.keys())}")
    return _REGISTRY[api]

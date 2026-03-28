"""
MCP 工具加载器。

从 mcp_servers.json 读取服务器配置，连接每个 MCP Server，
把它们暴露的工具转换成 TOOLS 字典格式，供 agent 使用。

注意：MCP SDK 是异步的，这里用 asyncio.run() 做同步桥接。
"""

import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "mcp_servers.json")


def load_mcp_tools(config_path: str = CONFIG_FILE) -> dict:
    """加载所有 MCP Server 的工具，返回 TOOLS 格式的字典。"""
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        server_configs = json.load(f)

    tools = {}
    for config in server_configs:
        try:
            server_tools = asyncio.run(_load_server_tools(config))
            tools.update(server_tools)
            print(f"[MCP] 已加载 {config['name']} 的 {len(server_tools)} 个工具")
        except Exception as e:
            print(f"[MCP] 加载 {config['name']} 失败: {e}")

    return tools


async def _load_server_tools(config: dict) -> dict:
    """连接单个 MCP Server，列出其工具。"""
    params = _build_params(config)
    tools = {}

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            for tool in result.tools:
                schema_str = json.dumps(
                    tool.inputSchema, ensure_ascii=False
                ) if tool.inputSchema else "见工具描述"
                tools[tool.name] = {
                    "func": _make_call_fn(config, tool.name),
                    "description": f"[MCP/{config['name']}] {tool.description or tool.name}",
                    "params": schema_str,
                }

    return tools


def _make_call_fn(config: dict, tool_name: str):
    """为每个 MCP 工具创建同步调用函数。"""
    def call(input_str: str) -> str:
        return asyncio.run(_call_tool(config, tool_name, input_str))
    return call


async def _call_tool(config: dict, tool_name: str, input_str: str) -> str:
    """调用 MCP 工具，返回文本结果。"""
    params = _build_params(config)

    # 尝试将输入解析为 JSON，否则包装为 {"input": ...}
    try:
        tool_input = json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        tool_input = {"input": input_str}

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_input)

    # 提取文本内容
    texts = [c.text for c in result.content if hasattr(c, "text")]
    return "\n".join(texts) if texts else "工具执行完成，无文本返回"


def _build_params(config: dict) -> StdioServerParameters:
    return StdioServerParameters(
        command=config["command"],
        args=config.get("args", []),
        env=config.get("env") or None,
    )

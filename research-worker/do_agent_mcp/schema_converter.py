"""Convert MCP tool schemas to OpenAI function-calling format."""

from __future__ import annotations

from typing import Any


def mcp_to_openai(mcp_tools: list[Any], server_name: str) -> list[dict]:
    """Convert a list of MCP Tool objects to OpenAI function-calling format.

    Each MCP tool has:
        name: str
        description: str | None
        inputSchema: dict (JSON Schema)

    Returns list of dicts matching OpenAI tool format:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Tool names are prefixed with server_name to avoid collisions across servers.
    """
    openai_tools = []
    for tool in mcp_tools:
        # Prefix tool name with server name: "playwright__navigate"
        prefixed_name = f"{server_name}__{tool.name}"

        parameters = {}
        if hasattr(tool, "inputSchema") and tool.inputSchema:
            parameters = tool.inputSchema
        else:
            # Empty parameters if no schema
            parameters = {"type": "object", "properties": {}}

        openai_tools.append({
            "type": "function",
            "function": {
                "name": prefixed_name,
                "description": tool.description or f"Tool from {server_name}",
                "parameters": parameters,
            },
            # Internal metadata for routing calls back to the right server
            "_mcp_server": server_name,
            "_mcp_tool": tool.name,
        })
    return openai_tools

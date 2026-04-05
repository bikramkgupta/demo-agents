"""Convenience functions for agent code — one-liner tool discovery."""

from __future__ import annotations

import logging

from .client import McpToolSet

logger = logging.getLogger(__name__)

# Module-level singleton so agents can call discover_tools() at startup
# and call_tool() later without managing the McpToolSet lifecycle.
_toolset: McpToolSet | None = None


async def discover_tools() -> tuple[list[dict], McpToolSet]:
    """Discover all MCP tools from configured servers.

    Reads MCP_SERVERS env var, connects to each server, discovers tools.
    Returns (openai_tools, toolset) where:
        - openai_tools: list of dicts in OpenAI function-calling format
        - toolset: McpToolSet instance for calling tools later

    Usage in agent startup:
        tools, toolset = await discover_tools()
        # tools can be passed to LLM's tools= parameter
        # toolset.call_tool("server__tool", {...}) to execute
    """
    global _toolset

    toolset = McpToolSet.from_env()
    await toolset.connect()
    _toolset = toolset

    tools = toolset.get_openai_tools()
    logger.info("Discovered %d MCP tools from %d servers", len(tools), len(toolset._servers))

    return tools, toolset


async def cleanup_sessions(conversation_id: str | None = None) -> None:
    """Clean up MCP sessions for a conversation."""
    if _toolset:
        await _toolset.cleanup(conversation_id)


async def shutdown() -> None:
    """Close all MCP connections. Call on agent shutdown."""
    global _toolset
    if _toolset:
        await _toolset.close()
        _toolset = None

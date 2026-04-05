"""do_agent_mcp — MCP tool discovery and execution for agents on App Platform.

Reads MCP_SERVERS env var, connects to MCP servers, discovers tools,
and returns OpenAI function-calling compatible specs + callable handlers.
"""

from .discover import cleanup_sessions, discover_tools

__all__ = ["discover_tools", "cleanup_sessions"]

"""MCP client — connects to MCP servers, discovers tools, executes calls.

Handles:
- Multiple MCP servers (catalog containers, external, functions)
- Session management for stateful tools (Playwright, Puppeteer)
- Bearer token auth for external servers
- Reconnection with backoff on failure
- Graceful degradation (skip unavailable servers)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .schema_converter import mcp_to_openai

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Parsed MCP server configuration from MCP_SERVERS env var."""

    name: str
    url: str
    execution: str = "container"  # "function" | "container" | "external"
    stateful: bool = False
    session_config: dict = field(default_factory=dict)
    auth: dict | None = None  # {"type": "bearer", "token": "..."}

    @property
    def auth_headers(self) -> dict[str, str]:
        if self.auth and self.auth.get("type") == "bearer":
            token = self.auth["token"]
            # Resolve env var references like ${DIGITALOCEAN_TOKEN}
            if token.startswith("${") and token.endswith("}"):
                token = os.environ.get(token[2:-1], token)
            return {"Authorization": f"Bearer {token}"}
        return {}


class McpToolSet:
    """Manages connections to multiple MCP servers and provides unified tool access.

    Usage:
        toolset = McpToolSet.from_env()
        await toolset.connect()
        tools = toolset.get_openai_tools()
        result = await toolset.call_tool("playwright__navigate", {"url": "..."}, conversation_id="abc")
        await toolset.cleanup("abc")
    """

    def __init__(self, servers: list[ServerConfig]):
        self._servers = {s.name: s for s in servers}
        self._sessions: dict[str, ClientSession] = {}  # server_name -> MCP session
        self._exit_stack: AsyncExitStack | None = None
        self._openai_tools: list[dict] = []
        # Session tracking: (server_name, conversation_id) -> session_id
        self._conversation_sessions: dict[tuple[str, str], str] = {}

    @classmethod
    def from_env(cls) -> McpToolSet:
        """Create McpToolSet from MCP_SERVERS environment variable."""
        raw = os.environ.get("MCP_SERVERS", "[]")
        try:
            configs = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid MCP_SERVERS JSON, no tools will be available")
            return cls([])

        servers = []
        for c in configs:
            servers.append(ServerConfig(
                name=c["name"],
                url=c["url"],
                execution=c.get("execution", "container"),
                stateful=c.get("stateful", False),
                session_config=c.get("session_config", {}),
                auth=c.get("auth"),
            ))
        return cls(servers)

    async def connect(self) -> None:
        """Connect to all configured MCP servers. Skips unavailable ones."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for name, server in self._servers.items():
            try:
                await self._connect_server(name, server)
                logger.info("Connected to MCP server '%s' at %s", name, server.url)
            except Exception:
                logger.warning(
                    "Failed to connect to MCP server '%s' at %s — skipping",
                    name, server.url, exc_info=True,
                )

    async def _wait_for_container_server(self, server: ServerConfig, timeout: float) -> None:
        """Wait for an internal container MCP server to accept TCP connections."""
        parsed = urlparse(server.url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            raise ValueError(f"Invalid MCP server URL: {server.url}")

        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()

    async def _connect_server(self, name: str, server: ServerConfig, retries: int = 5) -> None:
        """Connect to a single MCP server with exponential backoff.

        Default 5 attempts to handle slow-starting container MCP servers on App Platform.
        """
        last_error = None
        attempt_timeout = float(os.environ.get("MCP_CONNECT_TIMEOUT", "5"))
        for attempt in range(retries):
            try:
                headers = server.auth_headers

                if server.execution == "container":
                    await self._wait_for_container_server(server, attempt_timeout)

                # MCP transport is streamable HTTP on the /mcp endpoint.
                cm = streamablehttp_client(
                    url=server.url,
                    headers=headers,
                    timeout=attempt_timeout,
                )

                # Enter context via exit stack — keeps task scope consistent
                read_stream, write_stream, _ = await self._exit_stack.enter_async_context(cm)

                session = ClientSession(read_stream, write_stream)
                await self._exit_stack.enter_async_context(session)
                await session.initialize()

                self._sessions[name] = session

                # Discover tools
                tools_result = await session.list_tools()
                openai_tools = mcp_to_openai(tools_result.tools, name)
                self._openai_tools.extend(openai_tools)

                logger.info("Discovered %d tools from '%s'", len(openai_tools), name)
                return

            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.debug("Retry %d for '%s' in %ds", attempt + 1, name, wait)
                    await asyncio.sleep(wait)

        raise last_error  # type: ignore[misc]

    def get_openai_tools(self) -> list[dict]:
        """Return all discovered tools in OpenAI function-calling format."""
        return self._openai_tools

    def get_tool_map(self) -> dict[str, dict]:
        """Return mapping of prefixed_name -> tool metadata for call routing."""
        return {t["function"]["name"]: t for t in self._openai_tools}

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
        conversation_id: str | None = None,
    ) -> Any:
        """Call an MCP tool by its prefixed name (e.g. 'playwright__navigate').

        For stateful tools, tracks session_id per conversation to maintain state
        (e.g. same browser instance across multiple calls).
        """
        # Find the right server and original tool name
        tool_meta = self.get_tool_map().get(tool_name)
        if not tool_meta:
            raise ValueError(f"Unknown tool: {tool_name}")

        server_name = tool_meta["_mcp_server"]
        original_name = tool_meta["_mcp_tool"]

        session = self._sessions.get(server_name)
        if not session:
            raise ConnectionError(f"Not connected to MCP server '{server_name}'")

        server = self._servers[server_name]

        # Session affinity for stateful tools
        meta = {}
        if server.stateful and conversation_id:
            key = (server_name, conversation_id)
            if key in self._conversation_sessions:
                meta["session_id"] = self._conversation_sessions[key]

        # Call the tool
        result = await session.call_tool(original_name, arguments=arguments)

        # Track session ID from response (if stateful)
        if server.stateful and conversation_id and hasattr(result, "meta"):
            if result.meta and "session_id" in result.meta:
                self._conversation_sessions[(server_name, conversation_id)] = result.meta["session_id"]

        # Extract text content from MCP result
        if hasattr(result, "content") and result.content:
            texts = []
            for item in result.content:
                if hasattr(item, "text"):
                    texts.append(item.text)
                elif hasattr(item, "data"):
                    texts.append(f"[binary data: {getattr(item, 'mimeType', 'unknown')}]")
            return "\n".join(texts)

        return str(result)

    async def cleanup(self, conversation_id: str | None = None) -> None:
        """Clean up sessions for a specific conversation or all sessions."""
        keys_to_remove = []
        for key in self._conversation_sessions:
            if conversation_id is None or key[1] == conversation_id:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._conversation_sessions[key]

    async def close(self) -> None:
        """Close all MCP connections."""
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error closing MCP exit stack", exc_info=True)
            self._exit_stack = None
        self._sessions.clear()
        self._openai_tools.clear()

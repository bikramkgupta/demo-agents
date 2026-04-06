"""Platform Tools Agent — discovers and uses MCP tools at startup.

Tools run externally (fetch as function, playwright as container).
The agent waits for required MCP dependencies before serving traffic.
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("platform-tools-agent")

GRADIENT_API_KEY = os.environ.get("GRADIENT_API_KEY", "")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "https://inference.do-ai.run/v1")
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-4o")

_tools = []
_toolset = None


async def _discover_tools():
    """Discover required MCP tools before serving traffic."""
    global _tools, _toolset
    logger.info("Discovering required MCP tools before serving traffic...")
    from do_agent_mcp import discover_tools

    _tools, _toolset = await discover_tools()
    tool_names = [t["function"]["name"] for t in _tools]
    logger.info("Discovered %d tools: %s", len(_tools), tool_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _discover_tools()
    try:
        yield
    finally:
        if _toolset:
            await _toolset.close()


app = FastAPI(title="Platform Tools Agent", lifespan=lifespan)


async def run_agent(query: str, conversation_id: str) -> str:
    import httpx

    messages = [
        {"role": "system", "content": "You are a research assistant. Use the available tools to gather real data and answer questions."},
        {"role": "user", "content": query},
    ]

    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(8):  # Max rounds
            resp = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                headers={"Authorization": f"Bearer {GRADIENT_API_KEY}"},
                json={"model": MODEL, "messages": messages, "tools": _tools or None},
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            messages.append(msg)

            if choice.get("finish_reason") == "tool_calls" or msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    logger.info("Tool call: %s(%s)", fn_name, json.dumps(fn_args)[:200])

                    try:
                        result = await _toolset.call_tool(
                            fn_name, fn_args, conversation_id=conversation_id
                        )
                        result_str = str(result) if result else ""
                        logger.info("Tool result: %s", result_str[:200])
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                        logger.error("Tool error: %s", e)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })
            else:
                return msg.get("content", "")

    return "Max tool-calling rounds reached."


class ChatRequest(BaseModel):
    model: str = MODEL
    messages: list[dict] = []
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    import uuid

    query = ""
    for m in reversed(request.messages):
        if m.get("role") == "user":
            query = m.get("content", "")
            break

    conversation_id = str(uuid.uuid4())
    logger.info("Query: %s", query[:100])
    response = await run_agent(query, conversation_id)

    # Cleanup stateful sessions
    if _toolset:
        await _toolset.cleanup(conversation_id)

    return {
        "id": f"chatcmpl-{conversation_id[:8]}",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
    }


@app.get("/health")
async def health():
    tool_names = [t["function"]["name"] for t in _tools] if _tools else []
    return {"status": "ok", "tools_discovered": len(tool_names), "tools": tool_names}

"""Minimal test agent for Playwright MCP server on App Platform.

Discovers tools at startup, exposes /health with tool list.
Accepts queries and executes tool calls via MCP.
"""
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("test-playwright")

GRADIENT_API_KEY = os.environ.get("GRADIENT_API_KEY", "")
LLM_ENDPOINT = os.environ.get("LLM_GATEWAY_ENDPOINT", "https://inference.do-ai.run/v1")
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-4o")

_tools = []
_toolset = None


async def _discover_background():
    global _tools, _toolset
    delay = int(os.environ.get("MCP_STARTUP_DELAY", "15"))
    logger.info("Waiting %ds for MCP servers...", delay)
    await asyncio.sleep(delay)
    try:
        from do_agent_mcp import discover_tools
        _tools, _toolset = await discover_tools()
        names = [t["function"]["name"] for t in _tools]
        logger.info("Discovered %d tools: %s", len(names), names)
    except Exception as e:
        logger.error("Discovery failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_discover_background())
    yield
    task.cancel()
    if _toolset:
        await _toolset.close()


app = FastAPI(title="Test Playwright Agent", lifespan=lifespan)


async def run_agent(query: str, conversation_id: str) -> str:
    import httpx
    messages = [
        {"role": "system", "content": "You are a browser automation assistant. Use the available tools to navigate websites, take screenshots, and extract information."},
        {"role": "user", "content": query},
    ]
    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(8):
            resp = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                headers={"Authorization": f"Bearer {GRADIENT_API_KEY}"},
                json={"model": MODEL, "messages": messages, "tools": _tools or None},
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            messages.append(msg)
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    logger.info("Tool: %s(%s)", fn, json.dumps(args)[:200])
                    try:
                        result = await _toolset.call_tool(fn, args, conversation_id=conversation_id)
                        result_str = str(result)[:5000] if result else ""
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                        logger.error("Tool error: %s", e)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})
            else:
                return msg.get("content", "")
    return "Max rounds reached."


class ChatRequest(BaseModel):
    model: str = MODEL
    messages: list[dict] = []
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    query = ""
    for m in reversed(request.messages):
        if m.get("role") == "user":
            query = m.get("content", "")
            break
    cid = str(uuid.uuid4())
    response = await run_agent(query, cid)
    if _toolset:
        await _toolset.cleanup(cid)
    return {"id": f"chatcmpl-{cid[:8]}", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "tools_discovered": len(_tools),
            "tools": [t["function"]["name"] for t in _tools]}

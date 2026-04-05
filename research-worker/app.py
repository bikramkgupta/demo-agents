"""Research Agent — browses the web and fetches APIs for travel research.

Uses MCP tools (Playwright for browsing, fetch for APIs).
Receives LLM access via Plano gateway at LLM_GATEWAY_ENDPOINT.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("research-agent")

LLM_ENDPOINT = os.environ.get("LLM_GATEWAY_ENDPOINT", "https://inference.do-ai.run/v1")
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-4o")

_tools = []
_toolset = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tools, _toolset
    try:
        from do_agent_mcp import discover_tools
        _tools, _toolset = await discover_tools()
        logger.info("Discovered %d tools", len(_tools))
    except Exception as e:
        logger.warning("MCP discovery failed: %s", e)
    yield
    if _toolset:
        await _toolset.close()


app = FastAPI(title="Research Agent", lifespan=lifespan)

SYSTEM_PROMPT = """You are a travel research assistant. Your job is to:
- Look up weather forecasts for destinations
- Check currency exchange rates
- Browse travel sites for practical tips and information
- Summarize your findings clearly

Use the available tools to gather real data. Be thorough but concise."""


async def run_agent(messages: list[dict], conversation_id: str) -> str:
    import httpx

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(10):
            resp = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                json={"model": MODEL, "messages": full_messages, "tools": _tools or None},
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            full_messages.append(msg)

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    logger.info("Tool: %s(%s)", fn_name, json.dumps(fn_args)[:100])
                    try:
                        result = await _toolset.call_tool(fn_name, fn_args, conversation_id=conversation_id)
                        result_str = str(result) if result else ""
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                    full_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})
            else:
                return msg.get("content", "")
    return "Max rounds reached."


class ChatRequest(BaseModel):
    model: str = MODEL
    messages: list[dict] = []
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    conversation_id = str(uuid.uuid4())
    response = await run_agent(request.messages, conversation_id)
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
    return {"status": "ok", "agent": "research", "tools": len(_tools)}

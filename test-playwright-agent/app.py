"""Minimal test agent for Playwright MCP server on App Platform.

Discovers required tools at startup, exposes /health with tool list,
and only serves traffic after MCP dependencies are ready.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("test-playwright")

GRADIENT_API_KEY = os.environ.get("GRADIENT_API_KEY", "")
LLM_ENDPOINT = os.environ.get("LLM_GATEWAY_ENDPOINT", "https://inference.do-ai.run/v1")
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-4o")

_tools = []
_toolset = None
SYSTEM_PROMPT = (
    "You are a browser automation assistant. Use the available tools to navigate "
    "websites, take screenshots, and extract information."
)


async def _discover_tools():
    global _tools, _toolset
    logger.info("Discovering required MCP tools before serving traffic...")
    from do_agent_mcp import discover_tools

    _tools, _toolset = await discover_tools()
    names = [t["function"]["name"] for t in _tools]
    logger.info("Discovered %d tools: %s", len(names), names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _discover_tools()
    try:
        yield
    finally:
        if _toolset:
            await _toolset.close()


app = FastAPI(title="Test Playwright Agent", lifespan=lifespan)


def _prepare_messages(messages: list[dict]) -> list[dict]:
    prepared = list(messages or [])
    if not any(message.get("role") == "system" for message in prepared):
        prepared.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return prepared


def _iter_text_chunks(text: str, size: int = 120):
    for idx in range(0, len(text), size):
        yield text[idx:idx + size]


def _chunk_payload(response_id: str, model: str, created: int, delta: dict, finish_reason):
    return json.dumps({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })


async def _stream_response(task: asyncio.Task[str], response_id: str, model: str):
    created = int(time.time())
    yield f"data: {_chunk_payload(response_id, model, created, {'role': 'assistant'}, None)}\n\n"

    while not task.done():
        yield ": keep-alive\n\n"
        await asyncio.sleep(5)

    try:
        response = await task
    except Exception as exc:
        logger.exception("Streaming request failed")
        response = f"Internal error: {exc}"

    for chunk in _iter_text_chunks(response):
        yield f"data: {_chunk_payload(response_id, model, created, {'content': chunk}, None)}\n\n"
        await asyncio.sleep(0)

    yield f"data: {_chunk_payload(response_id, model, created, {}, 'stop')}\n\n"
    yield "data: [DONE]\n\n"


async def run_agent(messages: list[dict], conversation_id: str, model: str) -> str:
    import httpx
    messages = _prepare_messages(messages)
    async with httpx.AsyncClient(timeout=120) as client:
        for _ in range(8):
            resp = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                headers={"Authorization": f"Bearer {GRADIENT_API_KEY}"},
                json={"model": model, "messages": messages, "tools": _tools or None},
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
    model: str | None = None
    messages: list[dict] = []
    stream: bool = False
    conversation_id: str | None = None


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    effective_model = request.model or MODEL
    conversation_id = request.conversation_id or str(uuid.uuid4())
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    should_cleanup = request.conversation_id is None

    async def _run_request() -> str:
        try:
            return await run_agent(request.messages, conversation_id, effective_model)
        finally:
            if _toolset and should_cleanup:
                await _toolset.cleanup(conversation_id)

    if request.stream:
        task = asyncio.create_task(_run_request())
        return StreamingResponse(
            _stream_response(task, response_id, effective_model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = await _run_request()
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": effective_model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "tools_discovered": len(_tools),
            "tools": [t["function"]["name"] for t in _tools]}

"""Hello Agent — simplest possible agent on the platform.

Tools run in-process. No MCP, no Plano, no external services.
Just a FastAPI agent with inline tools that the LLM can call.
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("hello-agent")

app = FastAPI(title="Hello Agent")

GRADIENT_API_KEY = os.environ.get("GRADIENT_API_KEY", "")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "https://inference.do-ai.run/v1")
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-4o")

# --- Tools (in-process) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current UTC time and optionally convert to a timezone offset",
            "parameters": {
                "type": "object",
                "properties": {
                    "utc_offset": {
                        "type": "integer",
                        "description": "UTC offset in hours (e.g. 9 for Tokyo, -5 for NYC)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g. '15 * 0.15 + 85.50')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get simulated weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
]

WEATHER_DATA = {
    "tokyo": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
    "new york": {"temp": 18, "condition": "Sunny", "humidity": 45},
    "london": {"temp": 14, "condition": "Overcast", "humidity": 80},
    "san francisco": {"temp": 16, "condition": "Foggy", "humidity": 75},
}


def call_tool(name: str, args: dict) -> str:
    logger.info("Tool call: %s(%s)", name, json.dumps(args))

    if name == "get_time":
        offset = args.get("utc_offset", 0)
        from datetime import timedelta
        now = datetime.now(timezone.utc) + timedelta(hours=offset)
        return json.dumps({"time": now.strftime("%Y-%m-%d %H:%M:%S"), "utc_offset": offset})

    elif name == "calculate":
        expr = args.get("expression", "")
        # Safe eval: only allow math operations
        allowed = set("0123456789.+-*/() %")
        if not all(c in allowed for c in expr.replace(" ", "")):
            return json.dumps({"error": "Invalid expression"})
        try:
            result = eval(expr)  # noqa: S307 — safe: only math chars allowed
            return json.dumps({"expression": expr, "result": round(result, 4)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "get_weather":
        city = args.get("city", "").lower()
        weather = WEATHER_DATA.get(city, {"temp": 20, "condition": "Unknown", "humidity": 50})
        return json.dumps({"city": args.get("city"), **weather})

    return json.dumps({"error": f"Unknown tool: {name}"})


# --- LLM tool-calling loop ---

async def run_agent(query: str) -> str:
    import httpx

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the available tools to answer questions accurately."},
        {"role": "user", "content": query},
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(5):  # Max 5 tool-calling rounds
            resp = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                headers={"Authorization": f"Bearer {GRADIENT_API_KEY}"},
                json={"model": MODEL, "messages": messages, "tools": TOOLS},
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
                    result = call_tool(fn_name, fn_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
            else:
                return msg.get("content", "")

    return "Max tool-calling rounds reached."


# --- HTTP endpoints ---

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

    logger.info("Query: %s", query[:100])
    response = await run_agent(query)
    logger.info("Response: %s", response[:100])

    return {
        "id": "chatcmpl-hello",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "tools": [t["function"]["name"] for t in TOOLS]}

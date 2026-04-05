"""Booking Agent — handles flight booking, pricing, and reservations.

Uses in-process tools (simulated booking database).
Receives LLM access via Plano gateway at LLM_GATEWAY_ENDPOINT.
"""

import json
import logging
import os
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("booking-agent")

LLM_ENDPOINT = os.environ.get("LLM_GATEWAY_ENDPOINT", "https://inference.do-ai.run/v1")
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-4o")

app = FastAPI(title="Booking Agent")

# --- Simulated booking database ---

FLIGHTS = {
    "NYC-TYO-001": {"airline": "Japan Airlines", "route": "NYC → Tokyo", "price": 1250, "duration": "14h 30m", "stops": 0},
    "NYC-TYO-002": {"airline": "ANA", "route": "NYC → Tokyo", "price": 1180, "duration": "14h 45m", "stops": 0},
    "NYC-TYO-003": {"airline": "United", "route": "NYC → Tokyo", "price": 980, "duration": "16h 20m", "stops": 1},
    "NYC-LON-001": {"airline": "British Airways", "route": "NYC → London", "price": 650, "duration": "7h 15m", "stops": 0},
    "NYC-LON-002": {"airline": "Delta", "route": "NYC → London", "price": 580, "duration": "7h 30m", "stops": 0},
}

BOOKINGS = {}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin city code (e.g. NYC, LON, TYO)"},
                    "destination": {"type": "string", "description": "Destination city code"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight by its flight ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string", "description": "Flight ID (e.g. NYC-TYO-001)"},
                    "passenger_name": {"type": "string", "description": "Passenger name"},
                },
                "required": ["flight_id", "passenger_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": "Look up a booking by confirmation code",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string", "description": "Booking confirmation code"},
                },
                "required": ["booking_id"],
            },
        },
    },
]


def call_tool(name: str, args: dict) -> str:
    logger.info("Tool: %s(%s)", name, json.dumps(args))

    if name == "search_flights":
        origin = args.get("origin", "").upper()
        dest = args.get("destination", "").upper()
        key = f"{origin}-{dest}"
        results = [
            {"flight_id": fid, **info}
            for fid, info in FLIGHTS.items()
            if fid.startswith(key)
        ]
        if not results:
            return json.dumps({"flights": [], "message": f"No flights found for {origin} → {dest}"})
        return json.dumps({"flights": results, "count": len(results)})

    elif name == "book_flight":
        flight_id = args.get("flight_id", "")
        passenger = args.get("passenger_name", "")
        if flight_id not in FLIGHTS:
            return json.dumps({"error": f"Flight {flight_id} not found"})
        booking_id = f"BK-{uuid.uuid4().hex[:6].upper()}"
        BOOKINGS[booking_id] = {
            "booking_id": booking_id,
            "flight": FLIGHTS[flight_id],
            "passenger": passenger,
            "status": "confirmed",
        }
        return json.dumps(BOOKINGS[booking_id])

    elif name == "get_booking":
        booking_id = args.get("booking_id", "")
        if booking_id not in BOOKINGS:
            return json.dumps({"error": f"Booking {booking_id} not found"})
        return json.dumps(BOOKINGS[booking_id])

    return json.dumps({"error": f"Unknown tool: {name}"})


SYSTEM_PROMPT = """You are a flight booking assistant. You can:
- Search for available flights between cities
- Book flights for passengers
- Look up existing bookings
Use the available tools to help users find and book flights."""


async def run_agent(messages: list[dict]) -> str:
    import httpx

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(5):
            resp = await client.post(
                f"{LLM_ENDPOINT}/chat/completions",
                json={"model": MODEL, "messages": full_messages, "tools": TOOLS},
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            full_messages.append(msg)

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    result = call_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                    full_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            else:
                return msg.get("content", "")
    return "Max rounds reached."


class ChatRequest(BaseModel):
    model: str = MODEL
    messages: list[dict] = []
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    response = await run_agent(request.messages)
    return {
        "id": f"chatcmpl-booking",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "booking", "tools": [t["function"]["name"] for t in TOOLS]}

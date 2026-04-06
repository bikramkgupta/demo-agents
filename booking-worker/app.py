"""Booking Agent — handles flight booking, pricing, and reservations.

Uses in-process tools (simulated booking database).
Receives LLM access via Plano gateway at LLM_GATEWAY_ENDPOINT.
"""

import json
import logging
import os
import uuid
from datetime import date, datetime, timedelta

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("booking-agent")

LLM_ENDPOINT = os.environ.get("LLM_GATEWAY_ENDPOINT", "https://inference.do-ai.run/v1")
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-4o")

app = FastAPI(title="Booking Agent")

# --- Simulated booking database ---

AIRPORT_ALIASES = {
    "NYC": "NYC",
    "NEW YORK": "NYC",
    "NEW YORK CITY": "NYC",
    "LON": "LON",
    "LONDON": "LON",
    "TYO": "TYO",
    "TOKYO": "TYO",
    "SFO": "SFO",
    "SF": "SFO",
    "SAN FRANCISCO": "SFO",
    "SAN FRANCISCO INTERNATIONAL": "SFO",
    "AMS": "AMS",
    "AMSTERDAM": "AMS",
    "AMSTERDAM SCHIPHOL": "AMS",
}

FLIGHTS = {
    "NYC-TYO-001": {
        "origin": "NYC",
        "destination": "TYO",
        "departure_date": "2026-04-29",
        "airline": "Japan Airlines",
        "route": "NYC → Tokyo",
        "price": 1250,
        "duration": "14h 30m",
        "stops": 0,
    },
    "NYC-TYO-002": {
        "origin": "NYC",
        "destination": "TYO",
        "departure_date": "2026-04-30",
        "airline": "ANA",
        "route": "NYC → Tokyo",
        "price": 1180,
        "duration": "14h 45m",
        "stops": 0,
    },
    "NYC-TYO-003": {
        "origin": "NYC",
        "destination": "TYO",
        "departure_date": "2026-05-02",
        "airline": "United",
        "route": "NYC → Tokyo",
        "price": 980,
        "duration": "16h 20m",
        "stops": 1,
    },
    "NYC-LON-001": {
        "origin": "NYC",
        "destination": "LON",
        "departure_date": "2026-04-28",
        "airline": "British Airways",
        "route": "NYC → London",
        "price": 650,
        "duration": "7h 15m",
        "stops": 0,
    },
    "NYC-LON-002": {
        "origin": "NYC",
        "destination": "LON",
        "departure_date": "2026-05-01",
        "airline": "Delta",
        "route": "NYC → London",
        "price": 580,
        "duration": "7h 30m",
        "stops": 0,
    },
    "SFO-AMS-001": {
        "origin": "SFO",
        "destination": "AMS",
        "departure_date": "2026-04-27",
        "airline": "KLM",
        "route": "San Francisco → Amsterdam",
        "price": 1020,
        "duration": "10h 40m",
        "stops": 0,
    },
    "SFO-AMS-002": {
        "origin": "SFO",
        "destination": "AMS",
        "departure_date": "2026-04-29",
        "airline": "United",
        "route": "San Francisco → Amsterdam",
        "price": 870,
        "duration": "13h 10m",
        "stops": 1,
    },
    "SFO-AMS-003": {
        "origin": "SFO",
        "destination": "AMS",
        "departure_date": "2026-04-30",
        "airline": "Delta",
        "route": "San Francisco → Amsterdam",
        "price": 940,
        "duration": "13h 25m",
        "stops": 1,
    },
    "SFO-AMS-004": {
        "origin": "SFO",
        "destination": "AMS",
        "departure_date": "2026-05-02",
        "airline": "Lufthansa",
        "route": "San Francisco → Amsterdam",
        "price": 890,
        "duration": "14h 05m",
        "stops": 1,
    },
    "SFO-AMS-005": {
        "origin": "SFO",
        "destination": "AMS",
        "departure_date": "2026-05-03",
        "airline": "British Airways",
        "route": "San Francisco → Amsterdam",
        "price": 915,
        "duration": "14h 20m",
        "stops": 1,
    },
}

BOOKINGS = {}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights in the demo inventory between two cities or airport codes",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin airport code or city name (e.g. NYC, SFO, New York, San Francisco)",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport code or city name (e.g. TYO, AMS, Tokyo, Amsterdam)",
                    },
                    "departure_date": {
                        "type": "string",
                        "description": "Requested departure date. Prefer YYYY-MM-DD, but DD/Month is also accepted.",
                    },
                    "flexible_days": {
                        "type": "integer",
                        "description": "How many days before/after the requested date are acceptable.",
                        "minimum": 0,
                        "maximum": 7,
                    },
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


def _normalize_airport(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().replace("-", " ").split()).upper()
    return AIRPORT_ALIASES.get(cleaned, cleaned)


def _parse_departure_date(raw: str | None) -> date | None:
    if not raw:
        return None

    text = raw.strip()
    if not text:
        return None

    today = datetime.utcnow().date()
    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%B/%Y",
        "%d/%b/%Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d/%B",
        "%d/%b",
        "%B %d",
        "%b %d",
    )

    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue

        if "%Y" in fmt:
            return parsed.date()

        candidate_year = today.year
        candidate = date(candidate_year, parsed.month, parsed.day)
        if candidate < today:
            candidate = date(candidate_year + 1, parsed.month, parsed.day)
        return candidate

    return None


def _available_route_dates(origin: str, dest: str) -> list[str]:
    return sorted(
        {
            info["departure_date"]
            for info in FLIGHTS.values()
            if info["origin"] == origin and info["destination"] == dest
        }
    )


def call_tool(name: str, args: dict) -> str:
    logger.info("Tool: %s(%s)", name, json.dumps(args))

    if name == "search_flights":
        origin = _normalize_airport(args.get("origin", ""))
        dest = _normalize_airport(args.get("destination", ""))
        requested_date = _parse_departure_date(args.get("departure_date"))
        flexible_days = max(0, min(int(args.get("flexible_days", 0) or 0), 7))

        all_results = [
            {"flight_id": fid, **info}
            for fid, info in FLIGHTS.items()
            if info["origin"] == origin and info["destination"] == dest
        ]

        if requested_date:
            window_start = requested_date - timedelta(days=flexible_days)
            window_end = requested_date + timedelta(days=flexible_days)
            results = [
                flight
                for flight in all_results
                if window_start <= date.fromisoformat(flight["departure_date"]) <= window_end
            ]
        else:
            window_start = None
            window_end = None
            results = all_results

        results.sort(key=lambda flight: (flight["price"], flight["departure_date"], flight["stops"]))

        search_criteria = {
            "origin": origin,
            "destination": dest,
            "requested_departure_date": requested_date.isoformat() if requested_date else None,
            "flexible_days": flexible_days,
        }

        if not results:
            if all_results and requested_date and window_start and window_end:
                available_dates = _available_route_dates(origin, dest)
                return json.dumps(
                    {
                        "flights": [],
                        "message": (
                            f"No flights found in the demo inventory for {origin} → {dest} between "
                            f"{window_start.isoformat()} and {window_end.isoformat()}. "
                            f"Available demo departure dates for this route: {', '.join(available_dates)}."
                        ),
                        "search_criteria": search_criteria,
                        "available_dates": available_dates,
                    }
                )

            supported_routes = sorted({f"{info['origin']} → {info['destination']}" for info in FLIGHTS.values()})
            return json.dumps(
                {
                    "flights": [],
                    "message": (
                        f"No flights found in the demo inventory for {origin} → {dest}. "
                        f"Supported demo routes: {', '.join(supported_routes)}."
                    ),
                    "search_criteria": search_criteria,
                }
            )

        return json.dumps({"flights": results, "count": len(results), "search_criteria": search_criteria})

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
Use the available tools to help users find and book flights.

Important constraints:
- This is a demo backed by a small in-memory flight inventory, not a live airline system.
- When a user specifies a departure date, pass it to search_flights as departure_date.
- When a user says their date is flexible, pass flexible_days to search_flights.
- If the tool reports missing demo inventory, explain that limitation directly instead of implying a real-world no-availability result."""


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

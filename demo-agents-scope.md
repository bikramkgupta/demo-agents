# demo-agents scope

This repository contains deployable demo agents and MCP server containers for DigitalOcean App Platform.

The goal is to provide:

- Small end-to-end examples that can be deployed quickly.
- Reference implementations for in-process tools, MCP-backed tools, and Plano-routed multi-agent topologies.
- Demo-friendly agent behavior that is still close enough to real API contracts for integration testing.

It does not aim to be a production-ready agent framework.

## API contract

All deployable agent services in this repo expose:

- `POST /v1/chat/completions`
- `GET /health`

Current compatibility targets:

- Request `model` is honored when provided; otherwise the agent falls back to its environment default.
- Request `messages` are preserved instead of being collapsed to only the latest user turn.
- Request `stream=true` returns SSE from App Platform services.
- Non-streaming requests return OpenAI-style JSON chat completions.
- Control-plane-style model aliases such as `openai/openai-gpt-4o` are normalized into the Gradient-compatible bare model ids used by these demos.

Current intentional limitations:

- Streaming is compatibility-oriented, not token-perfect parity with OpenAI. Tool execution is still completed inside the agent before the final content stream is emitted.
- `usage`, `system_fingerprint`, and advanced response metadata are not populated.
- The repo supports `chat.completions` style requests only. It does not implement the Responses API.
- Demo agents remain Gradient-first for inference. Unsupported model ids fall back to the configured `LLM_MODEL` instead of attempting direct BYO-provider routing.

## Tool state model

Tool behavior depends on the backing execution mode:

- App Platform container tools can be stateful and long-lived. Examples: Playwright and Chrome DevTools MCP servers.
- Function-backed tools are stateless by design. They should be treated as per-call utilities.

For MCP-backed agents, this repo supports an additional request field:

- `conversation_id: string | null`

When `conversation_id` is provided:

- Stateful tool sessions can be reused across multiple requests.
- The agent keeps the MCP session affinity for that conversation instead of cleaning it up at the end of the request.

When `conversation_id` is omitted:

- The request is treated as request-scoped.
- The agent cleans up any locally tracked stateful MCP session after the request completes.

This is a repo-specific extension, not part of the OpenAI API.

## Agent inventory

### Reusable skeletons

These are the best starting points for new agents:

- `platform-tools/`
  - Generic MCP-backed single-agent template.
  - Good fit for external fetch tools, browser tools, and research-style workflows.
  - Best example of env-driven tool discovery.

- `research-worker/`
  - Specialized MCP-backed research agent for Plano deployments.
  - Good fit when you want a browser or fetch-powered worker inside a multi-agent topology.

- `plano-router/`
  - Plano ingress container for multi-agent App Platform deployments.
  - Used only when routing across two or more agents.

### Purpose-built demos

These are intentionally opinionated and contain demo data:

- `hello-agent/`
  - Minimal in-process tool calling example.
  - Uses hard-coded weather data and a tiny local toolset.
  - Best treated as a smoke-test or starter template, not as a generic knowledge agent.

- `booking-worker/`
  - Flight-booking demo worker.
  - Uses an in-memory flight inventory and in-memory bookings.
  - Good for topology demos and agent routing validation, not real travel search.

### Test harnesses

These are validation-focused rather than general-purpose products:

- `test-playwright-agent/`
  - Browser-focused smoke-test agent for Playwright MCP deployments.
  - Best for validating MCP connectivity and browser execution.

- `test-cdp-agent/`
  - Browser-focused smoke-test agent for Chrome DevTools MCP deployments.
  - Best for validating CDP-oriented browser setups.

## MCP server inventory

These components are tools, not chat agents:

- `mcp-playwright/`
- `mcp-chrome-devtools/`
- `mcp-fetch/`

They are intended to be composed into agents through `MCP_SERVERS` rather than called directly by end users.

## Genericity guidelines

When adding a new agent to this repo, prefer the following:

- Preserve caller-provided `messages` and `model`.
- While the repo is Gradient-first, normalize caller model aliases into supported Gradient model ids and fall back to the configured default for unsupported models.
- Implement SSE on App Platform endpoints when `stream=true`.
- Keep system prompts narrow and task-specific.
- Make demo limitations explicit in tool responses.
- If the agent uses stateful MCP tools, support stable `conversation_id` handling.
- If the agent calls function-backed tools, assume each call is stateless.

Avoid the following unless the directory is explicitly labeled as a demo or test harness:

- Hard-coded domain datasets presented as live data.
- Request handlers that only inspect the latest user turn.
- Ignoring `model` or `stream` fields.
- Implied cross-request browser state without an explicit conversation key.

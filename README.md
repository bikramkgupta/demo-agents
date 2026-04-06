# demo-agents

Deployable demo agents and MCP server containers for DigitalOcean App Platform.

This repo is meant to be used with the `do-agent-runtime` control plane. It does not include an admin console, database, or deployment control plane by itself.

## What is here

- `hello-agent/`
  - Minimal single-agent example with in-process tools.

- `platform-tools/`
  - User-facing name: `Web Research Agent`.
  - Generic single-agent example that discovers MCP tools such as Playwright and fetch.
  - The directory name stays `platform-tools/` for compatibility with existing deploy flows.

- `research-worker/`
  - MCP-backed worker for multi-agent Plano deployments.

- `booking-worker/`
  - Booking demo worker for Plano deployments.

- `plano-router/`
  - Router container used for multi-agent deployments.

- `test-playwright-agent/`
  - Playwright smoke-test harness.
  - The `test-` prefix is intentional.

- `test-cdp-agent/`
  - Chrome DevTools smoke-test harness.
  - The `test-` prefix is intentional.

- `mcp-playwright/`, `mcp-chrome-devtools/`, `mcp-fetch/`
  - MCP server containers, not end-user chat agents.

## Important assumptions

- Deployable agent services in this repo currently require `GRADIENT_API_KEY`.
- These demo agents are intentionally Gradient-first for inference.
- Caller-supplied model aliases such as `openai/openai-gpt-4o` are normalized into the Gradient-compatible bare model ids used by the demos.
- Unsupported model ids fall back to the agent's configured `LLM_MODEL`.
- This repo does not yet provide first-class bring-your-own provider secrets.

## Use With A Control Plane

Use `do-agent-runtime` if you want:

- an admin console
- a REST API
- CLI-driven agent management
- YAML generation and App Platform deploys
- an agent test panel and logs view

### Start a local admin console

From the `do-agent-runtime` repo:

```bash
docker compose -f docker-compose.dev.yaml up
```

The control plane is then available at `http://localhost:8000`.

The `do-agent-runtime` docs currently describe the local default login as:

- username: `admin`
- password: `admin`

Those credentials come from the control plane env vars, not from this repo.

### Connect this repo to an existing control plane

Add components from this repo by pointing the control plane at:

- repo: `bikramkgupta/demo-agents`
- branch: your target branch
- subdirectory: one of the directories in this repo such as `hello-agent` or `platform-tools`

## UI Flow

From the control plane UI:

1. Create an agent.
2. Add a component from `bikramkgupta/demo-agents`.
3. Set the component subdirectory, for example `hello-agent`, `platform-tools`, `research-worker`, or `booking-worker`.
4. Add MCP servers if the chosen agent expects them.
5. Deploy.
6. Use the test panel against the deployed URL.

## CLI Flow

From the `do-agent-runtime` repo:

```bash
export DO_AGENT_API_URL=http://localhost:8000

python -m cli.main agent create my-agent --region nyc3
python -m cli.main component add 1 --name web-research-agent --repo bikramkgupta/demo-agents --subdirectory platform-tools
python -m cli.main yaml 1
python -m cli.main deploy 1 --wait
python -m cli.main status 1
python -m cli.main logs 1
```

To talk to a deployed agent directly:

```bash
python -m cli.client --endpoint https://your-agent.ondigitalocean.app -i
```

Or:

```bash
curl -X POST https://your-agent.ondigitalocean.app/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai-gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

## Picking Names

Recommended user-facing names:

- `hello-agent` -> `Hello Agent`
- `platform-tools` -> `Web Research Agent`
- `research-worker` -> `Research Worker`
- `booking-worker` -> `Booking Worker`

Keep the `test-` prefix for `test-playwright-agent` and `test-cdp-agent`.
Those directories are validation harnesses, and the name should continue to signal that clearly.

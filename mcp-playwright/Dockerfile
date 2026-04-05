# MCP Playwright Server — Headless browser automation
# Stateful: maintains browser sessions across tool calls.

FROM node:20-slim

WORKDIR /app

# Install system deps for Playwright browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    fonts-liberation libappindicator3-1 xdg-utils wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright MCP server + browsers
RUN npm install -g @playwright/mcp@0.0.70 \
    && npx playwright install chromium \
    && npm cache clean --force

EXPOSE 8080

# --host 0.0.0.0: bind to all interfaces (required in Docker)
# --allowed-hosts *: accept connections from any hostname (other containers)
# --port: SSE transport on this port
CMD ["npx", "@playwright/mcp", "--port", "8080", "--host", "0.0.0.0", "--allowed-hosts", "*"]

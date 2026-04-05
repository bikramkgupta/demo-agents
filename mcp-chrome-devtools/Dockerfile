# Chrome DevTools MCP Server
# Uses mcp-proxy to bridge stdio → streamable HTTP
FROM node:20-slim

WORKDIR /app

# Install Chrome DevTools MCP + mcp-proxy (stdio→HTTP bridge)
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium fonts-liberation curl \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g chrome-devtools-mcp@latest mcp-proxy@latest \
    && npm cache clean --force

# Chrome needs --no-sandbox in containers
ENV CHROME_PATH=/usr/bin/chromium
ENV PUPPETEER_SKIP_DOWNLOAD=true

EXPOSE 8080

# mcp-proxy bridges stdio MCP → streamable HTTP on port 8080
CMD ["mcp-proxy", "--transport", "streamablehttp", "--port", "8080", "--", "npx", "-y", "chrome-devtools-mcp@latest"]

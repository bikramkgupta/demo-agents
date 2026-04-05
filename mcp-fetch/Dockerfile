# MCP Fetch Server — HTTP requests and web page text extraction
# Uses supergateway to bridge stdio → Streamable HTTP transport

FROM node:20-slim

WORKDIR /app

RUN npm install -g mcp-fetch-server@1.1.2 \
    && npm install -g supergateway@3.4.3 \
    && npm cache clean --force

EXPOSE 8080

# Use streamable HTTP (not SSE) — supports multiple concurrent clients
CMD ["supergateway", \
     "--stdio", "mcp-fetch-server", \
     "--port", "8080", \
     "--host", "0.0.0.0", \
     "--outputTransport", "streamableHttp"]

# MCP Playwright Server — uses official Microsoft image
# Exposes streamable HTTP on port 8080 (not SSE)
FROM mcr.microsoft.com/playwright/mcp:latest
EXPOSE 8080
CMD ["node", "/app/cli.js", "--port", "8080", "--host", "0.0.0.0"]

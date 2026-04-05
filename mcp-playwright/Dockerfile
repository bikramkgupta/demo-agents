# MCP Playwright Server — uses official Microsoft image
# Overrides entrypoint to bind to 0.0.0.0 and expose port 8080
FROM mcr.microsoft.com/playwright/mcp:latest
EXPOSE 8080
ENTRYPOINT ["node", "/app/cli.js", "--headless", "--browser", "chromium", "--no-sandbox", "--port", "8080", "--host", "0.0.0.0"]

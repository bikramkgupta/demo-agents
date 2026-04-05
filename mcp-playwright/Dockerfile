# MCP Playwright Server — uses official Microsoft image
FROM mcr.microsoft.com/playwright/mcp:latest
EXPOSE 8080
ENTRYPOINT ["node", "/app/cli.js", "--headless", "--browser", "chromium", "--no-sandbox", "--port", "8080", "--host", "0.0.0.0", "--allowed-hosts", "*"]

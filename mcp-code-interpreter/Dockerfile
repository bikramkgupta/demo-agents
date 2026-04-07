# MCP Code Interpreter — Python sandbox via PRIMS
# Runs arbitrary Python code in isolated virtualenvs
# Source: https://github.com/hileamlakB/Python-Runtime-Interpreter-MCP-Server

FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && pip install fastmcp aiohttp aiofiles

# Clone PRIMS server code
RUN git clone --depth 1 https://github.com/hileamlakB/Python-Runtime-Interpreter-MCP-Server.git /tmp/prims \
    && cp -r /tmp/prims/server /app/server \
    && rm -rf /tmp/prims

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["fastmcp", "run", "server/main.py", "--transport", "http", "--host", "0.0.0.0", "--port", "8080"]

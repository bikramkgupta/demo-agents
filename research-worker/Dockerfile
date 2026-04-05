FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY research-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY do_agent_mcp/ /app/do_agent_mcp/
COPY research-worker/app.py .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.13.5-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

WORKDIR /app

COPY engine /app/engine
COPY engine_server /app/engine_server

RUN uv sync --locked --no-dev --project /app/engine_server

WORKDIR /app/engine_server

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "-m", "uvicorn", "engine_server.app:app", "--host", "0.0.0.0", "--port", "8000"]

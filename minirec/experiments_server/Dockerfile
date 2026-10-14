FROM python:3.13.5-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

WORKDIR /app

COPY experiments_server /app/experiments_server

RUN uv sync --locked --no-dev --project /app/experiments_server

WORKDIR /app/experiments_server

EXPOSE 8010

CMD ["uv", "run", "--no-sync", "python", "-m", "uvicorn", "experiments_server.app:app", "--host", "0.0.0.0", "--port", "8010"]

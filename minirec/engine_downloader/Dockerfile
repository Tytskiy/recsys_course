FROM python:3.13.5-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

WORKDIR /app/engine_downloader

COPY engine_downloader /app/engine_downloader

RUN uv sync --locked --no-dev

CMD ["uv", "run", "--no-sync", "python", "engine_downloader.py"]

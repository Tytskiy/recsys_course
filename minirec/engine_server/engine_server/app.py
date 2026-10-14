from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import yaml
from engine.bloom import BloomToken
from engine.interfaces import RecContext
from fastapi import FastAPI

from engine_server.models import RecommendRequest, RecommendResponse
from engine_server.runtime import EngineRuntime

CONFIG_PATH_ENV = "REC_ENGINE_CONFIG_PATH"


def load_config() -> dict[str, Any]:
    config_path = os.getenv(CONFIG_PATH_ENV, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create `EngineRuntime(load_config())`, keep it in `app.state.runtime`, start it before serving
    and stop it on shutdown."""
    # TODO: Your solution


app = FastAPI(title="recommender engine", version="0.1.0", lifespan=lifespan)


# TODO: Your solution

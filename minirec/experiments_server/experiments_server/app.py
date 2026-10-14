from __future__ import annotations

from fastapi import FastAPI

from .models import ExperimentDefinition, ResolveResponse, UpsertExperimentResponse
from .repository import ExperimentRepository


DEFAULT_DB_PATH = "experiments.db"


app = FastAPI(title="experiments-server", version="0.1.0")

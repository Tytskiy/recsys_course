from __future__ import annotations

import logging
import os

import httpx
from pydantic import BaseModel, ConfigDict, Field

from engine.cg.mixing import CGSchema
from engine.interfaces import ID, SourceType

logger = logging.getLogger(__name__)


class ExperimentParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    filter_history: bool = Field(
        description="Filter out items, purchased by user before"
    )
    cg_schema: CGSchema | None = Field(
        default=None,
        description="Candidate generation schema for mixing; None keeps the configured one",
    )


DEFAULT_EXPERIMENT_PARAMS = ExperimentParams(filter_history=False)


class ExperimentsServerResponse(BaseModel):
    group_name: str
    params: ExperimentParams


class ExperimentManager:
    def __init__(self, cg_names: set[str], enabled: bool = True):
        self.cg_names = cg_names
        self.client = None
        if enabled:
            service_url = os.getenv("EXPERIMENT_SERVICE_URL", "http://localhost:8010")
            self.client = httpx.Client(base_url=service_url.rstrip("/"), timeout=0.2)

    def resolve(self, user_id: ID, source_type: SourceType) -> ExperimentParams:
        if self.client is None:
            return DEFAULT_EXPERIMENT_PARAMS
        try:
            response = self.client.get(
                f"/experiments/{source_type}/resolve",
                params={"user_id": user_id},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("Experiments service failed for %s, using defaults: %r", source_type, e)
            return DEFAULT_EXPERIMENT_PARAMS
        params = ExperimentsServerResponse(**response.json()).params
        if params.cg_schema is not None:
            unknown = set(params.cg_schema.cg_names) - self.cg_names
            if unknown:
                logger.warning("Experiment %s names unknown CGs %s, using defaults", source_type, unknown)
                return DEFAULT_EXPERIMENT_PARAMS
        return params

    def shutdown(self) -> None:
        if self.client is not None:
            self.client.close()

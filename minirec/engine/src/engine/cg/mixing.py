from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field, field_validator

from engine.candidate_logger import CandidateLogger
from engine.interfaces import (
    ID,
    Candidate,
    CandidateGenerator,
    CandidateMixer,
    ComponentConfig,
    ComponentDeps,
    RecContext,
)
from engine.score_model import ModelType, ModelWrapper, build_model_wrapper


class CGParams(BaseModel):
    cg_name: str
    num_candidates: int = Field(gt=0)


class CGSchema(BaseModel):
    ordered_cgs: list[CGParams]
    total_candidates: int = Field(gt=0)

    @field_validator("ordered_cgs")
    @classmethod
    def unique_cg_names(cls, ordered_cgs: list[CGParams]) -> list[CGParams]:
        names = [p.cg_name for p in ordered_cgs]
        if len(names) != len(set(names)):
            raise ValueError(f"CG listed more than once in schema: {names}")
        return ordered_cgs

    @property
    def cg_names(self) -> list[str]:
        return [p.cg_name for p in self.ordered_cgs]


class SchemaMixerConfig(ComponentConfig):
    default_schema: CGSchema


class SchemaCandidateMixer(CandidateMixer):
    config_model = SchemaMixerConfig

    def __init__(self, default_schema: CGSchema, candidate_logger: CandidateLogger | None = None):
        self.default_schema = default_schema
        self.candidate_logger = candidate_logger

    @property
    def name(self) -> str:
        return "SchemaCandidateMixer"

    @property
    def cg_names(self) -> list[str]:
        return self.default_schema.cg_names

    @classmethod
    def start_component(cls, config: SchemaMixerConfig, deps: ComponentDeps) -> SchemaCandidateMixer:
        return cls(default_schema=config.default_schema, candidate_logger=deps.candidate_logger)

    def merge(
        self,
        candgens: dict[str, CandidateGenerator],
        context: RecContext,
        schema: CGSchema | None = None,
    ) -> Iterable[Candidate]:
        schema = schema or self.default_schema

        raw: dict[str, list[Candidate]] = {}
        for cg_params in schema.ordered_cgs:
            cg = candgens[cg_params.cg_name]
            raw[cg.name] = list(cg.extract_candidates(cg_params.num_candidates, context))

        if self.candidate_logger is not None:
            self.candidate_logger.log_cg_candidates(
                context,
                {name: [c.item_id for c in cands] for name, cands in raw.items()},
            )

        seen: set[ID] = set()
        for cg_params in schema.ordered_cgs:
            cg = candgens[cg_params.cg_name]
            for candidate in raw[cg.name]:
                if candidate.item_id not in seen:
                    seen.add(candidate.item_id)
                    candidate.factors["source_cg"] = cg.name
                    yield candidate
                if len(seen) >= schema.total_candidates:
                    return


class MixigenConfig(ComponentConfig):
    model_path: str
    model_type: ModelType
    cg_names: list[str]
    context_features: list[str] = []
    total_candidates: int = Field(gt=0)
    num_candidates_per_cg: int = Field(gt=0)
    default_feature_value: float = 0.0


class MixigenCandidateMixer(CandidateMixer):
    """
    Context-aware CG mixer implementing the Mixigen algorithm.

    Prescores (cg_name, position) slots using a model, merges top-K by
    prescore, and retrieves candidates from each CG up to its highest
    selected position (quota = max_position_in_topK + 1).
    """

    config_model = MixigenConfig

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        cg_names: list[str],
        context_features: list[str],
        total_candidates: int,
        num_candidates_per_cg: int,
        default_feature_value: float = 0.0,
        candidate_logger: CandidateLogger | None = None,
    ):
        self.model_wrapper = model_wrapper
        self.cg_names = cg_names
        self.cg_name_to_index = {name: i for i, name in enumerate(cg_names)}
        self.context_features = context_features
        self.total_candidates = total_candidates
        self.num_candidates_per_cg = num_candidates_per_cg
        self.default_feature_value = default_feature_value
        self.candidate_logger = candidate_logger

    @property
    def name(self) -> str:
        return "MixigenCandidateMixer"

    @classmethod
    def start_component(cls, config: MixigenConfig, deps: ComponentDeps) -> MixigenCandidateMixer:
        return cls(
            model_wrapper=build_model_wrapper(config.model_type, config.model_path, deps.engine_data),
            cg_names=config.cg_names,
            context_features=config.context_features,
            total_candidates=config.total_candidates,
            num_candidates_per_cg=config.num_candidates_per_cg,
            default_feature_value=config.default_feature_value,
            candidate_logger=deps.candidate_logger,
        )

    def build_prescore_row(self, cg_name: str, position: int, context: RecContext) -> list:
        """Build a single feature row for one (CG, position) slot.

        Layout: `[*context_features (from context.user_profile, default-filled),
        cg_index, position]`. This row is fed to `self.model_wrapper.predict` to score
        how valuable the slot is for the current user.
        """
        # TODO: Your solution

    def compute_quotas(self, context: RecContext) -> dict[str, int]:
        """Batch-prescore all (CG, position) slots and return how many to fetch per CG.

        Enumerate every `(cg_name, pos)` for `pos in range(num_candidates_per_cg)`,
        score them all in one `predict` call, take the top `total_candidates` slot
        indices, and for each CG return `max_selected_position + 1` (the smallest
        prefix that covers every selected slot from that CG).
        """
        # TODO: Your solution

    def merge(
        self,
        candgens: dict[str, CandidateGenerator],
        context: RecContext,
        schema: CGSchema | None = None,
    ) -> Iterable[Candidate]:
        """Yield deduplicated candidates assembled from per-CG quotas.

        Compute quotas via `compute_quotas`, fetch that many candidates from each
        CG, optionally log raw lists via `candidate_logger.log_cg_candidates`, then
        yield in `self.cg_names` order, skipping duplicates and tagging each yielded
        candidate with `factors["source_cg"]`. Mixigen picks its own quotas, so an
        experiment `schema` is not used.
        """
        # TODO: Your solution


CANDGEN_MIXERS: dict[str, type[CandidateMixer]] = {
    "SchemaCandidateMixer": SchemaCandidateMixer,
    "MixigenCandidateMixer": MixigenCandidateMixer,
}

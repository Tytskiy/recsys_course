from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar, Generator, Iterable, Literal, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from engine.bloom import BloomToken
from engine.utils import EngineData

if TYPE_CHECKING:
    from engine.candidate_logger import CandidateLogger
    from engine.cg.mixing import CGSchema


type ID = int
type Embedding = np.ndarray
type FactorValue = str | int | float | Embedding
type Factor = FactorValue | list[FactorValue]
type EmbeddingMetric = Literal["Dot", "Cosine", "Euclid"]

type SourceType = Literal[
    "ST_CheckoutUpsale",
    "ST_Upsale",
    "ST_Feed",
    "ST_SearchComplementRec",
    "ST_Search",
    "ST_PreviousBuyCarousel",
    "ST_OnClickDelivery",
    "ST_SearchStartRec",
    "ST_Uplift",
    "ST_Catalog",
    "ST_PreviousBuyHub",
    "ST_ItemPageCarousel",
]

type ActionType = Literal[
    "AT_View",
    "AT_Click",
    "AT_CartUpdate",
    "AT_Purchase",
]


@dataclass
class UserAction:
    item_id: ID
    action_type: ActionType


@dataclass
class RecContext:
    user_id: ID
    user_history: dict[ActionType, list[ID]]
    user_profile: dict[str, Factor]
    current_time: datetime
    context_item_ids: list[ID]
    item_id_filter: BloomToken
    source_type: SourceType
    request_id: str

    @classmethod
    def from_request(
        cls,
        user_id: ID,
        current_time: datetime | None,
        context_item_ids: list[ID] | None,
        item_id_filter: BloomToken,
        source_type: SourceType,
    ) -> RecContext:
        return RecContext(
            user_id=user_id,
            user_history={},
            user_profile={},
            current_time=current_time or datetime.now(timezone.utc),
            context_item_ids=context_item_ids or [],
            item_id_filter=item_id_filter,
            source_type=source_type,
            request_id=str(uuid.uuid4()),
        )


@dataclass
class Candidate:
    item_id: ID
    factors: dict[str, Factor] = field(default_factory=dict)


class Recommender(Protocol):
    @classmethod
    def startup(cls, config: Any) -> Recommender:
        """Load and start the recommender"""

    def shutdown(self) -> None:
        """Shutdown the recommender"""

    def recommend(
        self,
        num_candidates: int,
        context: RecContext,
    ) -> list[Candidate]:
        """Recommend a set of items."""


class EmbeddingStorage(Protocol):
    @classmethod
    def start(cls, config: dict[str, Any]) -> EmbeddingStorage:
        """Build and initialize embedding storage from config."""

    def search(self, embedding_type: str, query: Embedding, k: int) -> list[tuple[ID, float]]:
        """Search for top-k embeddings by distance to query"""

    def get(self, embedding_type: str, ids: list[ID]) -> list[tuple[ID, Embedding]]:
        """Fetch embeddings by IDs"""

    def score(
        self, embedding_type: str, query: Embedding, ids: list[ID]
    ) -> list[tuple[ID, float]]:
        """Score query x embeddings by IDs"""

    def upsert_embeddings(
        self,
        embedding_type: str,
        embeddings: list[tuple[ID, Embedding]],
    ) -> None:
        """Upload embeddings into collection (insert or update)."""

    def build_embedding_index(
        self,
        embedding_type: str,
        embeddings: list[tuple[ID, Embedding]],
        metric: EmbeddingMetric = "Dot",
    ) -> None:
        """Build or rebuild embedding index for collection from scratch."""

    def delete_embedding_index(self, embedding_type: str) -> None:
        """Delete embedding collection index."""

    def close(self) -> None:
        """Close the client connection."""


class FeatureStorage(Protocol):
    @classmethod
    def start(cls, config: dict[str, Any]) -> FeatureStorage:
        """Build and initialize feature storage from config."""

    def extract(
        self, features: list[str], user_id: ID | None = None, item_id: ID | None = None
    ) -> dict[str, Factor]:
        """Fetch a list of features for a single (user_id, item_id)."""

    def bulk_extract(
        self,
        features: list[str],
        item_ids: list[ID],
        user_id: ID | None = None,
    ) -> dict[ID, dict[str, Factor]]:
        """Fetch a list of features for multiple item_ids under the same user_id."""

    def update(
        self,
        features: dict[str, Factor],
        user_id: ID | None = None,
        item_id: ID | None = None,
    ) -> None:
        """Upload or overwrite features for a key."""

    def delete(
        self,
        features: list[str],
        user_id: ID | None = None,
        item_id: ID | None = None,
    ) -> int:
        """Delete selected features for a key and return deleted count."""

    def close(self) -> None:
        """Close the client connection."""


class ComponentConfig(BaseModel):
    """Base of every component's `parameters` model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class ComponentDeps:
    """Shared resources a component may use; built once by `RecommenderEngine.startup`."""

    feature_storage: FeatureStorage
    embedding_storage: EmbeddingStorage
    engine_data: EngineData
    candidate_logger: CandidateLogger


class RecommenderComponent(Protocol):
    config_model: ClassVar[type[ComponentConfig]]

    @classmethod
    def start_component(cls, config: Any, deps: ComponentDeps) -> Self:
        """Load and start the RecommenderComponent from its validated config."""

    @property
    def name(self) -> str:
        """Name of component for configuration and logging"""


class RuntimeProfile(RecommenderComponent, Protocol):
    def hydrate_context(
        self,
        context: RecContext
    ) -> RecContext:
        """Hydrate context with runtime-computed information"""


class CandidateGenerator(RecommenderComponent, Protocol):
    def extract_candidates(
        self,
        num_candidates: int,
        context: RecContext,
    ) -> Generator[Candidate, None, None]:
        """Retrieve candidates from CG"""


class CandidateMixer(RecommenderComponent, Protocol):
    cg_names: list[str]

    def merge(
        self,
        candgens: dict[str, CandidateGenerator],
        context: RecContext,
        schema: CGSchema | None = None,
    ) -> Iterable[Candidate]:
        """Merge lists of candidates and deduplicate.

        `schema` is an experiment's CG schema that overrides the configured one.
        """


class Ranker(RecommenderComponent, Protocol):
    def rerank(
        self,
        candidates: list[Candidate],
        context: RecContext,
    ) -> list[Candidate]:
        """Rank list of candidates"""


class Filterer(RecommenderComponent, Protocol):
    def filter(
        self,
        candidates: Iterable[Candidate],
        context: RecContext,
    ) -> Iterable[Candidate]:
        """Filters list of candidates"""


class FeatureManager(RecommenderComponent, Protocol):
    def hydrate_context(
        self,
        context: RecContext
    ) -> RecContext:
        """Hydrate context with factors and transforms, defined in schema."""

    def hydrate_candidates(
        self,
        candidates: list[Candidate],
        context: RecContext,
    ) -> list[Candidate]:
        """Hydrate candidates with factors and transforms, defined in schema."""

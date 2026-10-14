from __future__ import annotations

from typing import Dict, Generator

import numpy as np

from engine.interfaces import (
    ID,
    Candidate,
    CandidateGenerator,
    ComponentConfig,
    ComponentDeps,
    RecContext,
)


class TopPopConfig(ComponentConfig):
    path: str
    seed: int = 42


class TopPop(CandidateGenerator):
    config_model = TopPopConfig

    def __init__(
        self,
        counts: Dict[ID, int],
        seed: int | None = 42,
    ):
        """Store the sampling distribution `counts[id] / sum(counts)` in
        `self.probabilities` (in the key order of `counts`) and a `numpy`
        generator seeded with `seed`.
        """
        # TODO: Your solution

    @property
    def name(self) -> str:
        return "TopPop"

    @classmethod
    def start_component(cls, config: TopPopConfig, deps: ComponentDeps) -> TopPop:
        data = deps.engine_data.read_json(config.path)
        counts = {int(k): int(v) for k, v in data.items()}
        return cls(counts=counts, seed=config.seed)

    def extract_candidates(
        self,
        num_candidates: int,
        context: RecContext,
    ) -> Generator[Candidate, None, None]:
        """Yield `min(num_candidates, len(counts))` distinct items drawn without
        replacement with probability proportional to their counts.
        """
        # TODO: Your solution

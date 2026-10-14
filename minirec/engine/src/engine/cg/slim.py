from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from engine.interfaces import (
    ActionType,
    Candidate,
    CandidateGenerator,
    ComponentConfig,
    ComponentDeps,
    RecContext,
)


class SLIMToHistoryConfig(ComponentConfig):
    action_type: ActionType
    mappings_path: str
    weights_path: str


class SLIMToHistory(CandidateGenerator):
    config_model = SLIMToHistoryConfig

    def __init__(
        self,
        mappings: dict[str, dict[int, int]],
        item_weights: sps.sparray,
        action_type: ActionType,
    ):
        """Keep the mappings with int keys and values; also store what
        `extract_candidates` needs to size the history vector.
        """
        self.mappings = {
            name: {int(k): int(v) for k, v in m.items()}
            for name, m in mappings.items()
        }
        self.item_weights = item_weights
        self.action_type = action_type
        # TODO: Your solution

    @classmethod
    def start_component(cls, config: SLIMToHistoryConfig, deps: ComponentDeps) -> SLIMToHistory:
        """Build the CG from `config`: `action_type`, the JSON at `mappings_path`
        and the sparse weight matrix at `weights_path` (read via `deps.engine_data`).
        """
        # TODO: Your solution

    @property
    def name(self) -> str:
        return f"SLIM_{self.action_type}_history"

    def extract_candidates(self, num_candidates: int, context: RecContext):
        """Sum item-to-item SLIM weights over the user's history and yield top items.

        Steps: build a binary history vector in mapping space → multiply by
        `item_weights` → take the top-`num_candidates` indices by score → translate
        indices back to item_ids via `index_to_item`.
        """
        # TODO: Your solution

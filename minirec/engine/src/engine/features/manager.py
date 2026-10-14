from __future__ import annotations

import re
from collections import defaultdict

from engine.features.config import (
    AVAILABLE_FEATURES,
    FEATURE_TO_STORAGE_KEYS,
    FEATURE_TO_USER_EMBEDDINGS,
    ITEM_LEVEL_FEATURES,
    ITEM_LEVEL_TRANSFORMS,
    ITEM_STATISTICS,
    REQUEST_LEVEL_FEATURES,
    REQUEST_LEVEL_TRANSFORMS,
    USER_STATISTICS,
)
from engine.features.feature import (
    item_item_dot_product,
    item_to_history_mean_dot_product,
    user_item_dot_product,
)
from engine.features.transform import (
    item_level_cart_update_turn_rate,
    request_level_cart_update_turn_rate,
)
from engine.interfaces import (
    ID,
    Candidate,
    ComponentConfig,
    ComponentDeps,
    EmbeddingStorage,
    Factor,
    FeatureManager,
    FeatureStorage,
    RecContext,
)

TURN_RATE_PATTERN = re.compile(r"^(?P<level>item|user):cart_update_turn_rate(?P<period>_30d)?$")


class SimpleManagerConfig(ComponentConfig):
    feature_schema: list[str]


class SimpleManager(FeatureManager):
    config_model = SimpleManagerConfig

    def __init__(
        self,
        schema: list[str],
        feature_storage: FeatureStorage,
        embedding_storage: EmbeddingStorage,
    ):
        unknown = [fn for fn in schema if fn not in AVAILABLE_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features in schema: {unknown}")
        for transform in schema:
            if m := TURN_RATE_PATTERN.fullmatch(transform):
                level, period = m.group("level"), m.group("period") or ""
                missing = [
                    d for d in (f"{level}:num_views{period}", f"{level}:num_cart_updates{period}")
                    if d not in schema
                ]
                if missing:
                    raise ValueError(f"{transform} needs {missing} in the schema")

        self.feature_storage = feature_storage
        self.embedding_storage = embedding_storage

        self.item_level_features = [fn for fn in schema if fn in ITEM_LEVEL_FEATURES]
        self.request_level_features = [fn for fn in schema if fn in REQUEST_LEVEL_FEATURES]
        self.item_level_transforms = [fn for fn in schema if fn in ITEM_LEVEL_TRANSFORMS]
        self.request_level_transforms = [fn for fn in schema if fn in REQUEST_LEVEL_TRANSFORMS]
        self.item_level_storage_keys = [
            FEATURE_TO_STORAGE_KEYS[fn]
            for fn in self.item_level_features
            if fn in FEATURE_TO_STORAGE_KEYS
        ]
        self.request_level_storage_keys = [
            FEATURE_TO_STORAGE_KEYS[fn]
            for fn in self.request_level_features
            if fn in FEATURE_TO_STORAGE_KEYS
        ]
        self.user_embedding_keys = [
            FEATURE_TO_USER_EMBEDDINGS[fn]
            for fn in self.request_level_features + self.item_level_features
            if fn in FEATURE_TO_USER_EMBEDDINGS
        ]

    @property
    def name(self) -> str:
        return "SimpleManager"

    @classmethod
    def start_component(cls, config: SimpleManagerConfig, deps: ComponentDeps) -> SimpleManager:
        return cls(
            schema=config.feature_schema,
            feature_storage=deps.feature_storage,
            embedding_storage=deps.embedding_storage,
        )

    def compute_request_level_feature(
        self,
        feature_name: str,
        preloaded_factors: dict[str, Factor],
        context: RecContext,
    ) -> Factor:
        if feature_name in USER_STATISTICS:
            return preloaded_factors.get(FEATURE_TO_STORAGE_KEYS[feature_name], 0)
        if feature_name == "source_type":
            return context.source_type
        if m := TURN_RATE_PATTERN.fullmatch(feature_name):
            return request_level_cart_update_turn_rate(context, m.group("period"))
        raise ValueError(f"Feature {feature_name} isn't defined")

    def compute_item_level_feature(
        self,
        feature_name: str,
        candidate_map: dict[ID, Candidate],
        preloaded_factors: dict[str, dict[ID, Factor]],
        context: RecContext,
    ) -> dict[ID, Factor]:
        if feature_name in ITEM_STATISTICS:
            return preloaded_factors.get(FEATURE_TO_STORAGE_KEYS[feature_name], {})
        if m := TURN_RATE_PATTERN.fullmatch(feature_name):
            return item_level_cart_update_turn_rate(candidate_map, m.group("period"))
        if feature_name == "iALS_score":
            return user_item_dot_product(
                candidate_map=candidate_map,
                embedding_type="iALS",
                item_storage=self.embedding_storage,
                context=context,
            )
        if feature_name == "iALS_to_history":
            return item_to_history_mean_dot_product(
                candidate_map=candidate_map,
                embedding_type="iALS",
                item_storage=self.embedding_storage,
                context=context,
            )
        if feature_name == "iALS_context_score":
            return item_item_dot_product(
                candidate_map=candidate_map,
                embedding_type="iALS",
                item_storage=self.embedding_storage,
                context=context,
            )
        raise ValueError(f"Feature {feature_name} isn't defined")

    def hydrate_context(self, context: RecContext) -> RecContext:
        # A runtime profile's fresh embedding wins over the offline one in Qdrant.
        for embed_key in self.user_embedding_keys:
            if embed_key in context.user_profile:
                continue
            response = self.embedding_storage.get(embed_key, [context.user_id])
            if response:
                context.user_profile[embed_key] = response[0][1]

        preloaded_factors: dict[str, Factor] = {}
        if self.request_level_storage_keys:
            preloaded_factors = self.feature_storage.extract(
                features=self.request_level_storage_keys,
                user_id=context.user_id,
            )

        # Transforms come last: they read the statistics computed before them.
        for feature_name in self.request_level_features + self.request_level_transforms:
            context.user_profile[feature_name] = self.compute_request_level_feature(
                feature_name, preloaded_factors, context
            )
        return context

    def hydrate_candidates(
        self,
        candidates: list[Candidate],
        context: RecContext,
    ) -> list[Candidate]:
        storage_response = self.feature_storage.bulk_extract(
            features=self.item_level_storage_keys,
            item_ids=[c.item_id for c in candidates],
        )

        preloaded_factors: defaultdict[str, dict[ID, Factor]] = defaultdict(dict)
        for iid, factors in storage_response.items():
            for feature, value in factors.items():
                preloaded_factors[feature][iid] = value

        cmap: dict[ID, Candidate] = {c.item_id: c for c in candidates}
        for feature_name in self.item_level_features + self.item_level_transforms:
            computed = self.compute_item_level_feature(
                feature_name, cmap, preloaded_factors, context
            )
            for iid, val in computed.items():
                cmap[iid].factors[feature_name] = val
        return list(cmap.values())

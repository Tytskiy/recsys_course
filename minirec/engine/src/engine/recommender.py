from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, get_args

import engine.candidate_logger as candidate_logging
from engine.cg import CANDGEN_MIXERS, CANDGENS
from engine.features import FEATURE_MANAGERS
from engine.filters import HistoryFilterer, SeenItemsFilterer
from engine.interfaces import (
    ID,
    ActionType,
    Candidate,
    CandidateGenerator,
    CandidateMixer,
    ComponentDeps,
    EmbeddingStorage,
    FeatureManager,
    FeatureStorage,
    Ranker,
    RecContext,
    Recommender,
    RecommenderComponent,
    RuntimeProfile,
)
from engine.parameters import ExperimentManager
from engine.ranking import RANKERS
from engine.runtime_profile import RUNTIME_PROFILES
from engine.storages import (
    EMBEDDING_STORAGES,
    FEATURE_STORAGES,
)
from engine.utils import EngineData

MAX_HISTORY_ACTIONS = 30


def _build_component[T: RecommenderComponent](
    registry: dict[str, type[T]], subconfig: dict[str, Any], deps: ComponentDeps
) -> T:
    cls = registry[subconfig["type"]]
    return cls.start_component(cls.config_model.model_validate(subconfig["parameters"]), deps)


class RecommenderEngine(Recommender):
    def __init__(
        self,
        feature_storage: FeatureStorage,
        embedding_storage: EmbeddingStorage,
        runtime_profiles: list[RuntimeProfile],
        candgens: dict[str, CandidateGenerator],
        candgen_mixer: CandidateMixer,
        feature_manager: FeatureManager,
        rankers: list[Ranker],
        engine_data: EngineData,
        exp_manager: ExperimentManager,
        candidate_logger: candidate_logging.CandidateLogger,
    ):
        self.runtime_profiles = runtime_profiles
        self.candgens = candgens
        self.feature_storage = feature_storage
        self.embedding_storage = embedding_storage
        self.feature_manager = feature_manager
        self.candgen_mixer = candgen_mixer
        self.rankers = rankers
        self.exp_manager = exp_manager
        self.engine_data = engine_data
        self.candidate_logger = candidate_logger
        self.seen_items_filter = SeenItemsFilterer()
        self.history_filter = HistoryFilterer()

    @staticmethod
    def setup_storages(
        config: Any,
    ) -> tuple[FeatureStorage, EmbeddingStorage]:
        subconfig = config["embedding_storage"]
        embedding_storage = EMBEDDING_STORAGES[subconfig["name"]].start(subconfig)
        try:
            subconfig = config["feature_storage"]
            feature_storage = FEATURE_STORAGES[subconfig["name"]].start(subconfig)
        except BaseException:
            embedding_storage.close()
            raise
        return feature_storage, embedding_storage

    @classmethod
    def startup(cls, config: Any) -> RecommenderEngine:
        with ExitStack() as cleanup:
            feature_storage, embedding_storage = cls.setup_storages(config)
            cleanup.callback(embedding_storage.close)
            cleanup.callback(feature_storage.close)
            engine_data = EngineData(config["engine_data_path"])
            cleanup.callback(engine_data.clear)
            # read at call time so tests can point the logs at a temp dir
            candidate_logger = candidate_logging.CandidateLogger(
                Path(candidate_logging.CANDIDATES_LOG_PATH),
                Path(candidate_logging.CG_CANDIDATES_LOG_PATH),
            )
            cleanup.callback(candidate_logger.close)
            deps = ComponentDeps(
                feature_storage=feature_storage,
                embedding_storage=embedding_storage,
                engine_data=engine_data,
                candidate_logger=candidate_logger,
            )

            runtime_profiles = [
                _build_component(RUNTIME_PROFILES, sub, deps) for sub in config["runtime_profile"]
            ]
            candgens = [
                _build_component(CANDGENS, sub, deps) for sub in config["candidate_generators"]
            ]
            named_candgens = {cg.name: cg for cg in candgens}
            candgen_mixer = _build_component(CANDGEN_MIXERS, config["candidate_mixer"], deps)
            unknown = set(candgen_mixer.cg_names) - named_candgens.keys()
            if unknown:
                raise ValueError(f"Mixer references unknown CGs {unknown}; known: {list(named_candgens)}")
            rankers = [_build_component(RANKERS, sub, deps) for sub in config["ranking"]]
            feature_manager = _build_component(FEATURE_MANAGERS, config["feature_manager"], deps)

            exp_manager = ExperimentManager(
                cg_names=set(named_candgens),
                enabled=bool(config["experiment_manager_enabled"]),
            )
            cleanup.pop_all()

        return cls(
            feature_storage=feature_storage,
            embedding_storage=embedding_storage,
            runtime_profiles=runtime_profiles,
            candgens=named_candgens,
            candgen_mixer=candgen_mixer,
            feature_manager=feature_manager,
            rankers=rankers,
            engine_data=engine_data,
            exp_manager=exp_manager,
            candidate_logger=candidate_logger,
        )

    def get_user_history(self, user_id: ID) -> dict[ActionType, list[ID]]:
        action_types: tuple[ActionType, ...] = get_args(ActionType.__value__)
        actions: dict[str, list[ID]] = self.feature_storage.extract(
            user_id=user_id,
            features=[f"user_history_{at}" for at in action_types],
        ) # ty: ignore[invalid-assignment]
        return {
            at: actions.get(f"user_history_{at}", [])[:MAX_HISTORY_ACTIONS]
            for at in action_types
        }

    def hydrate_context(self, context: RecContext) -> RecContext:
        context.user_history = self.get_user_history(context.user_id)
        for prof in self.runtime_profiles:
            context = prof.hydrate_context(context)
        context = self.feature_manager.hydrate_context(context)
        return context

    def recommend(
        self,
        num_candidates: int,
        context: RecContext,
    ) -> list[Candidate]:
        context = self.hydrate_context(context)
        params = self.exp_manager.resolve(context.user_id, context.source_type)

        candidates = self.candgen_mixer.merge(
            self.candgens, context, schema=params.cg_schema
        )

        candidates = self.seen_items_filter.filter(candidates, context)
        if params.filter_history:
            candidates = self.history_filter.filter(candidates, context)

        candidates = list(candidates)
        for position, candidate in enumerate(candidates, start=1):
            candidate.factors["mixer_position"] = position
        candidates = self.feature_manager.hydrate_candidates(candidates, context)

        for ranker in self.rankers:
            candidates = ranker.rerank(candidates, context)

        self.candidate_logger.log_candidates(
            request_id=context.request_id,
            context=context,
            candidates=candidates,
            num_served=num_candidates,
        )

        candidates = candidates[:num_candidates]

        return candidates

    def shutdown(self) -> None:
        self.engine_data.clear()
        self.candidate_logger.close()
        self.exp_manager.shutdown()
        self.feature_storage.close()
        self.embedding_storage.close()

from __future__ import annotations

from typing import Literal

from engine.interfaces import (
    Candidate,
    ComponentConfig,
    ComponentDeps,
    RecContext,
)
from engine.ranking.feature_ranker import FeatureReranker
from engine.score_model import ModelType, ModelWrapper, build_model_wrapper


class ModelRankerConfig(ComponentConfig):
    model_name: str
    model_type: ModelType
    model_path: str
    prediction_type: Literal["RawFormulaVal", "Probability"]
    factor_schema_path: str
    default_feature_value: float


class ModelRanker(FeatureReranker):
    config_model = ModelRankerConfig

    def __init__(
        self,
        model_name: str,
        model_wrapper: ModelWrapper,
        prediction_type: Literal["RawFormulaVal", "Probability"],
        factor_schema: list[tuple[str, bool]],
        default_feature_value: float,
    ):
        super().__init__(
            feature_key=f"{model_name}_score",
            default=float("-inf"),
            descending=True,
        )
        self.model_name = model_name
        self.prediction_type = prediction_type
        self.factor_schema = factor_schema
        self.default_feature_value = default_feature_value
        self.model_wrapper = model_wrapper

    @property
    def name(self):
        return f"{self.model_name}_Ranker"

    @classmethod
    def start_component(cls, config: ModelRankerConfig, deps: ComponentDeps) -> ModelRanker:  # ty: ignore[invalid-method-override]
        """Construct ModelRanker from its validated config and `deps.engine_data`.

        Read the factor schema JSON at `config.factor_schema_path` (list of
        `[factor_name, is_context_factor]`, cast to `(str, bool)`), build the wrapper
        for `config.model_type` with `build_model_wrapper`, and return an initialized
        instance.
        """
        # TODO: Your solution

    def get_factor_matrix(self, candidates: list[Candidate], context: RecContext):
        """Assemble the feature matrix that `model_wrapper.predict` expects.

        One row per candidate, one column per `(factor_name, is_context_factor)` in
        `self.factor_schema`. For context features pull from `context.user_profile`,
        for candidate features pull from `candidate.factors`; missing values become
        `self.default_feature_value`.
        """
        # TODO: Your solution

    def rerank(
        self,
        candidates: list[Candidate],
        context: RecContext,
    ) -> list[Candidate]:
        """Score every candidate with the model and return them sorted desc by score.

        Build the feature matrix, run `model_wrapper.predict`, write each score as a
        Python `float` to `candidate.factors[f"{model_name}_score"]`
        (= `self.feature_key`), then delegate the final descending sort to
        `FeatureReranker.rerank` via super().
        """
        # TODO: Your solution

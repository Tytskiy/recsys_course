"""Spec for `ModelRanker` (task `model_ranker`). Behaviours:
- `start_component` loads the wrapper for `model_type` (Linear or CatBoost) and parses the
  JSON factor schema; an unknown `model_type` is rejected by `ModelRankerConfig`;
- `get_factor_matrix`: one row per candidate, columns in `factor_schema` order, context
  columns from `context.user_profile`, candidate columns from `candidate.factors`, missing
  values -> `default_feature_value`;
- `rerank` writes each score to `factors[f"{model_name}_score"]` and sorts by it, descending;
- end to end with the real stats CatBoost ranker (`tests/test_data/ranker.cbm`).
"""

from __future__ import annotations

import numpy as np
import orjson
import pytest
from engine.interfaces import Candidate
from engine.ranking.model_ranker import ModelRanker, ModelRankerConfig
from engine.score_model import CatBoostWrapper, LinearModelWrapper
from scipy.special import expit


class TestStartComponent:
    """`ModelRanker.start_component` constructs a ranker from its typed config
    and the `deps.engine_data` file store."""

    @pytest.fixture
    def start_component(self, deps):
        """Factory: call with overrides; returns a constructed ModelRanker.

        Defaults form a valid Linear setup; each test overrides only the
        fields that matter to its assertion."""
        def factory(**overrides):
            config = {
                "model_name": "test",
                "model_path": "weights.npy",
                "model_type": "Linear",
                "prediction_type": "RawFormulaVal",
                "factor_schema_path": "schema.json",
                "default_feature_value": 0.0,
            }
            config.update(overrides)
            return ModelRanker.start_component(ModelRankerConfig.model_validate(config), deps)
        return factory

    def test_loads_linear_wrapper_for_linear_model_type(
        self, rec_engine_files_dir, start_component,
    ):
        np.save(rec_engine_files_dir / "weights.npy", np.array([1.0, 2.0]))
        (rec_engine_files_dir / "schema.json").write_bytes(
            orjson.dumps([["user_bias", True], ["item_score", False]]),
        )

        ranker = start_component(model_name="SimpleModel", model_type="Linear")

        assert ranker.name == "SimpleModel_Ranker"
        assert isinstance(ranker.model_wrapper, LinearModelWrapper)
        assert ranker.factor_schema == [("user_bias", True), ("item_score", False)]
        assert np.allclose(ranker.model_wrapper.model, [1.0, 2.0])

    def test_loads_catboost_wrapper_for_catboost_model_type(
        self, rec_engine_files_dir, engine_data, stats_ranker_model,
        canonical_factor_schema, start_component,
    ):
        (rec_engine_files_dir / "schema.json").write_bytes(
            orjson.dumps([list(pair) for pair in canonical_factor_schema]),
        )

        ranker = start_component(
            model_name="CBModel",
            model_type="CatBoost",
            model_path=stats_ranker_model,
            default_feature_value=-99999.0,
        )

        assert ranker.name == "CBModel_Ranker"
        assert isinstance(ranker.model_wrapper, CatBoostWrapper)
        assert ranker.factor_schema == canonical_factor_schema


def test_get_factor_matrix_pulls_context_candidate_and_default_values(
    rec_engine_files_dir, engine_data, feed_context,
):
    np.save(rec_engine_files_dir / "w.npy", np.array([1.0, 1.0, 1.0]))
    ranker = ModelRanker(
        model_name="m",
        model_wrapper=LinearModelWrapper("w.npy", engine_data=engine_data),
        prediction_type="RawFormulaVal",
        factor_schema=[
            ("user_bias", True),
            ("item_score", False),
            ("missing_feat", False),
        ],
        default_feature_value=-1.0,
    )
    candidates = [
        Candidate(item_id=1, factors={"item_score": 2.0}),
        Candidate(item_id=2, factors={}),
    ]

    matrix = ranker.get_factor_matrix(candidates, feed_context)

    assert matrix == [
        [1.0, 2.0, -1.0],
        [1.0, -1.0, -1.0],
    ]


@pytest.mark.parametrize(
    ("schema_entry", "expected"),
    [
        pytest.param(("user_bias", True), 1.0, id="context_factor_not_read_from_candidate"),
        pytest.param(("item_score", False), -1.0, id="candidate_factor_not_read_from_context"),
    ],
)
def test_get_factor_matrix_reads_only_the_declared_source(
    rec_engine_files_dir, engine_data, feed_context, schema_entry, expected,
):
    """The `is_context_factor` flag picks the source; falling back to the other source
    would let a stale per-candidate copy of a user feature reach the model."""
    np.save(rec_engine_files_dir / "w.npy", np.array([1.0]))
    feed_context.user_profile["item_score"] = 5.0
    ranker = ModelRanker(
        model_name="m",
        model_wrapper=LinearModelWrapper("w.npy", engine_data=engine_data),
        prediction_type="RawFormulaVal",
        factor_schema=[schema_entry],
        default_feature_value=-1.0,
    )

    matrix = ranker.get_factor_matrix([Candidate(item_id=1, factors={"user_bias": 99.0})], feed_context)

    assert matrix == [[expected]]


class TestRerank:
    """`rerank(candidates, context)` scores every candidate, stores the score
    under `f"{model_name}_score"`, and returns candidates sorted descending."""

    @pytest.fixture
    def linear_ranker(self, rec_engine_files_dir, engine_data) -> ModelRanker:
        np.save(rec_engine_files_dir / "w.npy", np.array([2.0, 1.0]))
        return ModelRanker(
            model_name="linear",
            model_wrapper=LinearModelWrapper("w.npy", engine_data=engine_data),
            prediction_type="RawFormulaVal",
            factor_schema=[("user_bias", True), ("item_score", False)],
            default_feature_value=0.5,
        )

    def test_writes_score_to_candidate_factors_under_model_name_key(
        self, linear_ranker, feed_context,
    ):
        candidates = [
            Candidate(item_id=1, factors={"item_score": 1.0}),
            Candidate(item_id=2, factors={"item_score": -1.0}),
        ]

        reranked = linear_ranker.rerank(candidates, feed_context)

        scores = {c.item_id: c.factors["linear_score"] for c in reranked}
        assert scores == pytest.approx({1: 3.0, 2: 1.0})

    def test_orders_candidates_descending_by_score(
        self, linear_ranker, feed_context,
    ):
        candidates = [
            Candidate(item_id=1, factors={"item_score": 1.0}),
            Candidate(item_id=2, factors={"item_score": -1.0}),
            Candidate(item_id=3, factors={}),
        ]

        reranked = linear_ranker.rerank(candidates, feed_context)

        assert [c.item_id for c in reranked] == [1, 3, 2]

    def test_writes_scores_as_python_float(self, linear_ranker, feed_context):
        """`predict` returns NumPy scalars; the candidate log serializes factors
        with orjson, so scores must be stored as plain `float`."""
        candidates = [Candidate(item_id=1, factors={"item_score": 1.0})]

        reranked = linear_ranker.rerank(candidates, feed_context)

        assert type(reranked[0].factors["linear_score"]) is float

    @pytest.mark.parametrize("prediction_type", ["RawFormulaVal", "Probability"])
    def test_passes_prediction_type_through_to_wrapper(
        self, rec_engine_files_dir, engine_data, feed_context, prediction_type,
    ):
        np.save(rec_engine_files_dir / "w.npy", np.array([2.0, 1.0]))
        ranker = ModelRanker(
            model_name="m",
            model_wrapper=LinearModelWrapper("w.npy", engine_data=engine_data),
            prediction_type=prediction_type,
            factor_schema=[("user_bias", True), ("item_score", False)],
            default_feature_value=0.5,
        )
        candidates = [Candidate(item_id=1, factors={"item_score": 1.0})]

        reranked = ranker.rerank(candidates, feed_context)

        expected = expit(3.0) if prediction_type == "Probability" else 3.0
        assert reranked[0].factors["m_score"] == pytest.approx(expected)


def test_end_to_end_with_real_catboost_ranker(
    rec_engine_files_dir, deps, feed_context,
    stats_ranker_model, canonical_factor_schema,
    sample_factors, expected_raw_score,
):
    """Pinned regression: real stats ranker driven via `start_component` +
    `rerank`. A high-signal candidate scores `expected_raw_score` and ranks
    above an all-zero baseline.

    Uses RawFormulaVal — Probability on a binary CatBoostClassifier returns
    shape (n, 2) which the scalar-keyed FeatureReranker can't sort.
    """
    (rec_engine_files_dir / "schema.json").write_bytes(
        orjson.dumps([list(pair) for pair in canonical_factor_schema]),
    )

    config = ModelRankerConfig(
        model_name="stats",
        model_path=stats_ranker_model,
        model_type="CatBoost",
        prediction_type="RawFormulaVal",
        factor_schema_path="schema.json",
        default_feature_value=-99999.0,
    )
    ranker = ModelRanker.start_component(config, deps)
    candidates = [
        Candidate(item_id=1, factors=dict(sample_factors)),
        Candidate(item_id=2, factors={name: 0 for name in sample_factors}),
    ]

    reranked = ranker.rerank(candidates, feed_context)

    assert [c.item_id for c in reranked] == [1, 2]
    assert reranked[0].factors["stats_score"] == pytest.approx(expected_raw_score, abs=1e-5)

"""Spec for `ModelRanker` (task model_ranker). Behaviours:
- `start_component` loads the wrapper for `model_type` (Linear or CatBoost) and reads the JSON
  factor schema as `(str, bool)` pairs; an unknown `model_type` is rejected by `ModelRankerConfig`;
- `get_factor_matrix`: one row per candidate, columns in `factor_schema` order, context columns
  from `context.user_profile`, candidate columns from `candidate.factors`, a value missing from
  its declared source -> `default_feature_value` (a present `0.0` stays `0.0`);
- `rerank` writes each score as a Python `float` to `factors[f"{model_name}_score"]`, passes
  `prediction_type` to the wrapper and returns the candidates sorted by score, descending;
- end to end with the real stats CatBoost ranker (`tests/test_data/ranker.cbm`).
"""

from __future__ import annotations

import numpy as np
import orjson
import pytest
from engine.interfaces import Candidate
from engine.ranking.model_ranker import ModelRanker, ModelRankerConfig
from engine.score_model import CatBoostWrapper, LinearModelWrapper
from engine.utils import EngineData
from scipy.special import expit

SCHEMA = [("user_bias", True), ("item_score", False), ("missing_feat", False)]


def linear_ranker(engine_data: EngineData, weights: list[float], schema=SCHEMA, prediction_type="RawFormulaVal", default=-1.0):
    np.save(engine_data.dir_path / "w.npy", np.array(weights))
    return ModelRanker("m", LinearModelWrapper("w.npy", engine_data=engine_data), prediction_type, list(schema), default)


def write_schema(engine_data: EngineData, schema) -> None:
    (engine_data.dir_path / "schema.json").write_bytes(orjson.dumps([list(pair) for pair in schema]))


def test_start_component_linear(engine_data, deps):
    np.save(engine_data.dir_path / "weights.npy", np.array([1.0, 2.0]))
    write_schema(engine_data, SCHEMA[:2])
    config = ModelRankerConfig(
        model_name="SimpleModel", model_type="Linear", model_path="weights.npy", prediction_type="RawFormulaVal",
        factor_schema_path="schema.json", default_feature_value=0.0,
    )

    ranker = ModelRanker.start_component(config, deps)

    assert ranker.name == "SimpleModel_Ranker"
    assert isinstance(ranker.model_wrapper, LinearModelWrapper)
    assert np.allclose(ranker.model_wrapper.model, [1.0, 2.0])
    assert ranker.factor_schema == [("user_bias", True), ("item_score", False)]
    assert (ranker.prediction_type, ranker.default_feature_value) == ("RawFormulaVal", 0.0)


def test_start_component_catboost(engine_data, deps, stats_ranker_model, canonical_factor_schema):
    write_schema(engine_data, canonical_factor_schema)
    config = ModelRankerConfig(
        model_name="CBModel", model_type="CatBoost", model_path=stats_ranker_model, prediction_type="RawFormulaVal",
        factor_schema_path="schema.json", default_feature_value=-99999.0,
    )

    ranker = ModelRanker.start_component(config, deps)

    assert ranker.name == "CBModel_Ranker"
    assert isinstance(ranker.model_wrapper, CatBoostWrapper)
    assert ranker.factor_schema == canonical_factor_schema


def test_get_factor_matrix_pulls_context_candidate_and_default_values(engine_data, feed_context):
    """Schema order: user_bias (context, 1.0), item_score (candidate), missing_feat (nowhere -> -1.0)."""
    ranker = linear_ranker(engine_data, [1.0, 1.0, 1.0])
    candidates = [
        Candidate(item_id=1, factors={"item_score": 2.0}),
        Candidate(item_id=2, factors={"item_score": 0.0}),
        Candidate(item_id=3, factors={}),
    ]

    assert ranker.get_factor_matrix(candidates, feed_context) == [
        [1.0, 2.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, -1.0, -1.0],
    ]


@pytest.mark.parametrize(
    ("schema_entry", "expected"),
    [
        pytest.param(("user_bias", True), 1.0, id="context_factor_not_read_from_candidate"),
        pytest.param(("item_score", False), -1.0, id="candidate_factor_not_read_from_context"),
    ],
)
def test_get_factor_matrix_reads_only_the_declared_source(engine_data, feed_context, schema_entry, expected):
    """Falling back to the other source would let a stale per-candidate copy of a user
    feature reach the model: the candidate carries user_bias=99, the profile item_score=5."""
    feed_context.user_profile["item_score"] = 5.0
    ranker = linear_ranker(engine_data, [1.0], schema=[schema_entry])

    assert ranker.get_factor_matrix([Candidate(item_id=1, factors={"user_bias": 99.0})], feed_context) == [[expected]]


# weights (2, 1, 0) over (user_bias=1, item_score, missing_feat): score = 2 + item_score, default 0.5
def test_rerank_writes_score_under_model_name_key_and_sorts_descending(engine_data, feed_context):
    ranker = linear_ranker(engine_data, [2.0, 1.0, 0.0], default=0.5)
    candidates = [
        Candidate(item_id=1, factors={"item_score": 1.0}),
        Candidate(item_id=2, factors={"item_score": -1.0}),
        Candidate(item_id=3, factors={}),
    ]

    reranked = ranker.rerank(candidates, feed_context)

    assert [(c.item_id, c.factors["m_score"]) for c in reranked] == [(1, 3.0), (3, 2.5), (2, 1.0)]


def test_rerank_writes_scores_as_python_float(engine_data, feed_context):
    """`predict` returns NumPy scalars; the candidate log serializes factors with orjson."""
    reranked = linear_ranker(engine_data, [2.0, 1.0, 0.0]).rerank([Candidate(item_id=1, factors={"item_score": 1.0})], feed_context)

    assert type(reranked[0].factors["m_score"]) is float


@pytest.mark.parametrize(
    ("prediction_type", "expected"),
    [
        pytest.param("RawFormulaVal", 3.0, id="raw"),
        pytest.param("Probability", expit(3.0), id="probability"),
    ],
)
def test_rerank_passes_prediction_type_to_the_wrapper(engine_data, feed_context, prediction_type, expected):
    ranker = linear_ranker(engine_data, [2.0, 1.0, 0.0], prediction_type=prediction_type)

    reranked = ranker.rerank([Candidate(item_id=1, factors={"item_score": 1.0})], feed_context)

    assert reranked[0].factors["m_score"] == pytest.approx(expected)


def test_end_to_end_with_real_catboost_ranker(
    engine_data, deps, feed_context, stats_ranker_model, canonical_factor_schema, sample_factors, expected_raw_score,
):
    """A logged high-signal row scores `expected_raw_score` and ranks above an all-zero row.
    RawFormulaVal: Probability on a binary CatBoostClassifier returns shape (n, 2)."""
    write_schema(engine_data, canonical_factor_schema)
    config = ModelRankerConfig(
        model_name="stats", model_type="CatBoost", model_path=stats_ranker_model, prediction_type="RawFormulaVal",
        factor_schema_path="schema.json", default_feature_value=-99999.0,
    )
    ranker = ModelRanker.start_component(config, deps)
    candidates = [
        Candidate(item_id=1, factors=dict(sample_factors)),
        Candidate(item_id=2, factors={name: 0 for name in sample_factors}),
    ]

    reranked = ranker.rerank(candidates, feed_context)

    assert [c.item_id for c in reranked] == [1, 2]
    assert reranked[0].factors["stats_score"] == pytest.approx(expected_raw_score, abs=1e-5)

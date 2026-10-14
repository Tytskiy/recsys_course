"""Unit tests for the `ModelWrapper` implementations consumed by `ModelRanker`.

These wrappers are provided — students do not implement them — but `ModelRanker`
depends on the contract pinned here:

  - `predict(rows, "RawFormulaVal")` returns one raw score per row.
  - `predict(rows, "Probability")` returns sigmoid-bounded scores in [0, 1]
    (LinearModelWrapper) or per-class probabilities of shape (n, 2)
    (CatBoostWrapper, binary classifier).
  - Empty input returns an empty array without raising.
  - Unknown `prediction_type` raises `AssertionError`.
"""

from __future__ import annotations

import numpy as np
import pytest
from engine.score_model import CatBoostWrapper, LinearModelWrapper
from scipy.special import expit


class TestLinearModelWrapper:
    @pytest.fixture
    def wrapper(self, rec_engine_files_dir, engine_data) -> LinearModelWrapper:
        np.save(rec_engine_files_dir / "weights.npy", np.array([1.0, 2.0]))
        return LinearModelWrapper("weights.npy", engine_data=engine_data)

    def test_raw_prediction_is_dot_product(self, wrapper):
        scores = wrapper.predict([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], "RawFormulaVal")
        assert list(scores) == pytest.approx([1.0, 2.0, 1.5])

    def test_probability_applies_sigmoid_to_dot_product(self, wrapper):
        scores = wrapper.predict([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], "Probability")
        assert list(scores) == pytest.approx(list(expit([1.0, 2.0, 1.5])))
        assert np.all((scores >= 0) & (scores <= 1))

    def test_empty_input_returns_empty_array(self, wrapper):
        assert len(wrapper.predict([], "RawFormulaVal")) == 0

    def test_rejects_unknown_prediction_type(self, wrapper):
        with pytest.raises(AssertionError):
            wrapper.predict([[1.0, 0.0]], "InvalidType")


class TestCatBoostWrapper:
    @pytest.fixture
    def wrapper(self, engine_data, stats_ranker_model) -> CatBoostWrapper:
        return CatBoostWrapper(stats_ranker_model, engine_data=engine_data)

    def test_raw_prediction_matches_pinned_value(
        self, wrapper, sample_factor_row, expected_raw_score,
    ):
        scores = wrapper.predict([sample_factor_row], "RawFormulaVal")
        assert scores[0] == pytest.approx(expected_raw_score, abs=1e-5)

    def test_probability_returns_two_class_distribution(
        self, wrapper, sample_factor_row, expected_prob_positive,
    ):
        scores = wrapper.predict([sample_factor_row], "Probability")
        assert scores.shape == (1, 2)
        assert np.all((scores >= 0) & (scores <= 1))
        assert scores[0, 1] == pytest.approx(expected_prob_positive, abs=1e-5)

    def test_empty_input_returns_empty_array(self, wrapper):
        assert len(wrapper.predict([], "RawFormulaVal")) == 0

    def test_rejects_unknown_prediction_type(self, wrapper, sample_factor_row):
        with pytest.raises(AssertionError):
            wrapper.predict([sample_factor_row], "InvalidType")

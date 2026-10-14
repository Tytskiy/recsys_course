from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from scipy.special import expit

from engine.interfaces import Factor
from engine.utils import EngineData


class ModelWrapper(Protocol):
    def predict(
        self,
        factors: list[list[Factor]],
        prediction_type: Literal["RawFormulaVal", "Probability"],
    ) -> np.ndarray:
        """Compute scores for items, presented as rows of factors"""


class LinearModelWrapper(ModelWrapper):
    def __init__(
        self,
        model_path: str,
        engine_data: EngineData,
    ):
        self.model = engine_data.read_np(model_path)

    def predict(
        self,
        factors: list[list[Factor]],
        prediction_type: Literal["RawFormulaVal", "Probability"],
    ) -> np.ndarray:
        assert prediction_type in ["RawFormulaVal", "Probability"]
        if not factors:
            return np.array([], dtype=float)
        scores = np.array(factors, dtype=float) @ self.model
        if prediction_type == "Probability":
            scores = expit(scores)
        return scores


class CatBoostWrapper(ModelWrapper):
    def __init__(
        self,
        model_path: str,
        engine_data: EngineData,
    ):
        self.model = engine_data.read_cbm(model_path)

    def predict(
        self,
        factors: list[list[Factor]],
        prediction_type: Literal["RawFormulaVal", "Probability"],
    ) -> np.ndarray:
        assert prediction_type in ["RawFormulaVal", "Probability"]
        if not factors:
            return np.array([], dtype=float)
        return self.model.predict(factors, prediction_type=prediction_type)


type ModelType = Literal["CatBoost", "Linear"]


def build_model_wrapper(
    model_type: ModelType, model_path: str, engine_data: EngineData
) -> ModelWrapper:
    wrapper = CatBoostWrapper if model_type == "CatBoost" else LinearModelWrapper
    return wrapper(model_path, engine_data=engine_data)

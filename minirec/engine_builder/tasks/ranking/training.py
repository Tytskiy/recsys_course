from __future__ import annotations

import dataclasses

from catboost import CatBoostClassifier, CatBoostRanker, Pool
from loguru import logger

_RANKER_LOSSES = {"YetiRank", "YetiRankPairwise", "QueryRMSE", "QuerySoftMax", "PairLogit", "PairLogitPairwise"}

_EVAL_METRICS = ["AUC:type=Ranking", "NDCG", "NDCG:top=10"]


@dataclasses.dataclass
class CatBoostTrainParams:
    loss_function: str = "YetiRank"
    iterations: int = 100
    eval_metric: str = "NDCG:top=10"
    early_stopping_rounds: int = 20
    random_seed: int = 42


def _get_model(params: CatBoostTrainParams) -> CatBoostClassifier | CatBoostRanker:
    kwargs = dict(
        loss_function=params.loss_function,
        iterations=params.iterations,
        eval_metric=params.eval_metric,
        random_seed=params.random_seed,
        verbose=False,
    )
    if params.loss_function in _RANKER_LOSSES:
        return CatBoostRanker(**kwargs)
    return CatBoostClassifier(**kwargs)


def train_catboost(
    train_pool: Pool,
    test_pool: Pool,
    params: CatBoostTrainParams,
) -> CatBoostClassifier | CatBoostRanker:
    """Train a CatBoost model and log evaluation metrics with loguru."""
    model = _get_model(params)
    model_type = type(model).__name__

    logger.info(
        "Starting {} training: loss={}, iterations={}, eval_metric={}",
        model_type, params.loss_function, params.iterations, params.eval_metric,
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=params.early_stopping_rounds,
    )

    best_iter = model.get_best_iteration()
    logger.info("Training complete, best iteration: {}", best_iter)

    metrics = model.eval_metrics(test_pool, _EVAL_METRICS)
    for metric_name, values in metrics.items():
        logger.info("  {}: {:.6f}", metric_name, values[best_iter])

    return model


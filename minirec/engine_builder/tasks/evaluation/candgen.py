import polars as pl
from engine.cg import UserToItemKNN
from engine.interfaces import ActionType, Candidate, EmbeddingStorage, RecContext
from loguru import logger

from tasks.evaluation.metric import Metric, Precision, Recall
from tasks.evaluation.pipeline import EvaluationPipeline


def evaluate_knn_cg(
    dm_action: pl.DataFrame,
    embedding_storage: EmbeddingStorage,
    embedding_type: str,
) -> dict[str, float]:
    logger.info("Creating CG instance")
    cg = UserToItemKNN(embedding_type, embedding_storage)

    user_key = f"user_{embedding_type}"

    def recommend(num_candidates: int, ctx: RecContext) -> list[Candidate]:
        # the engine hydrates the user embedding before CGs run; do the same here
        ctx.user_profile.update({user_key: embed for _, embed in embedding_storage.get(user_key, [ctx.user_id])})
        return list(cg.extract_candidates(num_candidates, ctx))

    logger.info("Running evaluation pipeline")
    pipe = EvaluationPipeline(
        dm_action=dm_action,
        recommend_handle=recommend,
        reload_handle=lambda dt: None,
    )

    logger.info("Running evaluation")
    responses = pipe.run()

    logger.info("Computing metrics on responses")
    metric_spec: list[tuple[Metric, ActionType]] = [
        (Recall(10), "AT_CartUpdate"),
        (Recall(50), "AT_CartUpdate"),
        (Precision(10), "AT_CartUpdate"),
        (Precision(50), "AT_CartUpdate"),
    ]
    metrics = pipe.compute_metrics(responses, metric_spec)
    logger.info(f"{embedding_type} kNN metrics: {metrics}")
    return metrics

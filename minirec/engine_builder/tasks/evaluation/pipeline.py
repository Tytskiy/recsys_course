from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import chain, groupby
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable

import polars as pl
from engine.bloom import BloomToken
from engine.interfaces import ID, ActionType, Candidate, RecContext, SourceType
from loguru import logger

from tasks.evaluation.metric import Metric

type RecommendHandle = Callable[[int, RecContext], list[Candidate]]
type ReloadHandle = Callable[[str], None]


@dataclass
class RequestItem:
    item_id: int
    timestamp: int
    action_type: ActionType


@dataclass
class Request:
    dt: str
    user_id: int
    request_id: int
    request_time: datetime
    source_type: SourceType
    actions: list[RequestItem]

    @property
    def item_ids(self) -> list[int]:
        return list(set(a.item_id for a in self.actions))


@dataclass
class EvaluationResponse:
    request: Request
    candidates: list[int]

    def item_id_subset(self, action_type: ActionType) -> list[ID]:
        subset = filter(lambda r: r.action_type == action_type, self.request.actions)
        return [r.item_id for r in subset]


class EvaluationPipeline:
    def __init__(
        self,
        dm_action: pl.DataFrame,
        recommend_handle: RecommendHandle,
        reload_handle: ReloadHandle,
        num_candidates: int = 100,
        use_candidates_from_actions: bool = False,
    ):
        self.recommend_handle = recommend_handle
        self.reload_handle = reload_handle
        self.num_candidates = num_candidates
        self.use_candidates_from_actions = use_candidates_from_actions
        self.requests = self.build_requests(dm_action)

    def build_recom_args(self, request: Request) -> tuple[int, RecContext]:
        if self.use_candidates_from_actions:
            context_item_ids = request.item_ids
            num_candidates = len(context_item_ids)
        else:
            context_item_ids = None
            num_candidates = self.num_candidates
        ctx = RecContext.from_request(
            user_id=request.user_id,
            current_time=request.request_time,
            context_item_ids=context_item_ids,
            item_id_filter=BloomToken(),
            source_type=request.source_type,
        )
        return num_candidates, ctx

    def run_batch(self, requests: Iterable[Request]) -> list[EvaluationResponse]:
        results = []
        for request in requests:
            num_candidates, ctx = self.build_recom_args(request)
            candidates = self.recommend_handle(num_candidates, ctx)
            results.append(
                EvaluationResponse(
                    request=request,
                    candidates=[c.item_id for c in candidates],
                )
            )
        return results

    def run(self) -> list[list[EvaluationResponse]]:
        requests_by_dt = sorted(self.requests, key=lambda x: x.dt)
        num_dates = len({r.dt for r in requests_by_dt})
        results = []
        for i, (dt, requests) in enumerate(groupby(requests_by_dt, key=lambda x: x.dt)):
            t0 = perf_counter()
            self.reload_handle(dt)
            t_reload = perf_counter() - t0

            t1 = perf_counter()
            batch = self.run_batch(requests)
            t_batch = perf_counter() - t1

            results.append(batch)
            logger.info(
                "[{}/{}] dt={} | reload={:.1f}s | batch={:.1f}s | requests={}",
                i + 1, num_dates, dt, t_reload, t_batch, len(batch),
            )
        return results

    @staticmethod
    def build_requests(dm_action: pl.DataFrame) -> list[Request]:
        requests_dataframe = (
            dm_action
            .filter(
                pl.col("request_id").is_not_null()
                & pl.col("user_id").is_not_null()
                & pl.col("item_id").is_not_null()
            )
            .with_columns(
                pl.struct(["item_id", "timestamp", "action_type"]).alias("action")
            )
            .group_by(
                "dt",
                "request_id",
                "user_id",
                "source_type",
            )
            .agg(
                pl.col("action").struct.field("timestamp").min().alias("min_ts"),
                pl.col("action"),
            )
        )
        requests: list[Request] = []
        for request in requests_dataframe.iter_rows(named=True):
            item_actions = []
            for action in request["action"]:
                item_actions.append(
                    RequestItem(
                        item_id=action["item_id"],
                        timestamp=action["timestamp"],
                        action_type=action["action_type"],
                    )
                )
            request_time = datetime.fromtimestamp(request["min_ts"], tz=UTC) - timedelta(seconds=5)
            requests.append(
                Request(
                    dt=request["dt"],
                    user_id=request["user_id"],
                    request_time=request_time,
                    request_id=request["request_id"],
                    source_type=request["source_type"],
                    actions=item_actions,
                )
            )
        return requests

    @staticmethod
    def compute_batch_metric(
        batch: list[EvaluationResponse],
        metric: Metric,
        action_type: ActionType = "AT_CartUpdate",
    ) -> float:
        batch_values = []
        for sample in batch:
            positives = sample.item_id_subset(action_type)
            if len(positives) > 0:
                value = metric.compute(
                    sample.candidates,
                    positives,
                    sample.request.user_id,
                )
                batch_values.append(value)
        return mean(batch_values) if batch_values else float("nan")

    @staticmethod
    def compute_metrics(
        batches: list[list[EvaluationResponse]],
        metric_spec: list[tuple[Metric, ActionType]],
    ) -> dict[str, float]:
        batch = list(chain.from_iterable(batches))
        metrics = {
            "num_requests": len(batch),
        }
        for metric, action_type in metric_spec:
            metrics[metric.name] = EvaluationPipeline.compute_batch_metric(
                batch, metric, action_type
            )
        return metrics

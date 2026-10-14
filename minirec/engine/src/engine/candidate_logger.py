from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

import orjson

from engine.interfaces import ID, Candidate, RecContext

CANDIDATES_LOG_PATH = os.getenv("CANDIDATES_LOG_PATH", "/logs/candidates.jsonl")
CG_CANDIDATES_LOG_PATH = os.getenv("CG_CANDIDATES_LOG_PATH", "/logs/cg_candidates.jsonl")
ROTATING_LOGFILE_MAX_BYTES = 100 * 1024 * 1024
ROTATING_LOGFILE_BACKUP_COUNT = 5


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Serialize `record.msg` (a dict) to a single-line JSON string via orjson.

        Factors may hold NumPy scalars (`np.float64`), so enable NumPy serialization.
        """
        # TODO: Your solution


class CandidateLogger:
    def __init__(self, candidates_path: Path, cg_candidates_path: Path) -> None:
        """Open two JSONL logs: `self.logger` (named `engine.candidates`, final ranked
        candidates) writes `candidates_path`, `self.cg_logger` (named
        `engine.cg_candidates`, per-CG raw lists) writes `cg_candidates_path`.

        Each gets its own `RotatingFileHandler` (`ROTATING_LOGFILE_MAX_BYTES`,
        `ROTATING_LOGFILE_BACKUP_COUNT`, UTF-8) with `JSONFormatter`, level INFO;
        create missing parent directories. The loggers belong to this instance: build
        them with `logging.Logger(name)`, not `logging.getLogger(name)`, so two
        instances never share handlers or write to each other's files.
        """
        # TODO: Your solution

    def close(self) -> None:
        for handler in [*self.logger.handlers, *self.cg_logger.handlers]:
            handler.close()

    @staticmethod
    def filter_factors(factors: dict) -> dict:
        """Return only entries whose value is a scalar (`str`, `int`, `float`).

        Vectors/embeddings/nested dicts blow up log size and are not useful for
        training, so they're dropped before logging.
        """
        # TODO: Your solution

    def log_candidates(
        self,
        request_id: str,
        context: RecContext,
        candidates: list[Candidate],
        num_served: int,
    ) -> None:
        """Write one JSONL record per request to the post-ranking log.

        Fields: `request_id`, `user_id`, `request_time` (ISO), `num_candidates`,
        `num_served` (how many of the logged candidates, from the top, the client
        actually got: `num_served` capped at `len(candidates)`),
        `user_profile` (scalar entries of `context.user_profile` — user-level
        features computed once per request, e.g. `user:num_views`,
        `source_type`), and `candidates` (each as `{item_id, factors}` with
        factors filtered to scalars only). The user profile is emitted at the
        top level rather than duplicated per candidate to keep the log compact;
        the offline pool builder spreads it into every candidate row.
        Use `self.logger.info(record)` — `JSONFormatter` will serialize the
        dict.
        """
        # TODO: Your solution

    def log_cg_candidates(
        self,
        context: RecContext,
        candidate_lists: dict[str, list[ID]],
    ) -> None:
        """Write one JSONL record per request to the pre-feature, per-CG log.

        Fields: `request_id`, `user_id`, `source_type`, `request_time` (ISO),
        `features` (scalar entries of `context.user_profile` only), and
        `candidate_lists` (`{cg_name: [item_ids]}`). Use `self.cg_logger.info`.
        """
        # TODO: Your solution

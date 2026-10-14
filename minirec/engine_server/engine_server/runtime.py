from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from engine.interfaces import Candidate, RecContext, Recommender

from engine import RecommenderEngine

LOCAL_ARCHIVE_PATH_ENV = "REC_ENGINE_LOCAL_ARCHIVE_PATH"
UPDATE_MARKER_PATH_ENV = "REC_ENGINE_UPDATE_MARKER_PATH"

DEFAULT_POLL_INTERVAL_SEC = 30.0
DEFAULT_TIMEOUT_INTERVAL_SEC = 180.0
DRAIN_TIMEOUT_SEC = 10.0
DEFAULT_LOCAL_ARCHIVE_PATH = "/shared/archive.tar.gz"
DEFAULT_UPDATE_MARKER_PATH = "/shared/archive.version.json"

logger = logging.getLogger(__name__)


class EngineRuntime:
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.engine: Recommender | None = None

        self.config = {
            **config,
            "engine_data_path": os.getenv(
                LOCAL_ARCHIVE_PATH_ENV, DEFAULT_LOCAL_ARCHIVE_PATH
            ),
        }
        self.update_marker_path = Path(
            os.getenv(UPDATE_MARKER_PATH_ENV, DEFAULT_UPDATE_MARKER_PATH)
        )

        # guards reading and swapping self.engine and the in-flight counts
        self.engine_lock = threading.Lock()
        self.engine_idle = threading.Condition(self.engine_lock)
        self.in_flight: Counter[Recommender] = Counter()

        self.update_stop_event: asyncio.Event | None = None
        self.update_task: asyncio.Task | None = None
        self.drain_tasks: set[asyncio.Task] = set()
        self.loaded_marker: str | None = None

    def read_marker(self) -> str | None:
        try:
            value = self.update_marker_path.read_text().strip()
            return value or None
        except FileNotFoundError:
            return None

    def reload_engine(self) -> Recommender:
        """Wait for the archive (FileNotFoundError on timeout), then start an engine."""
        # TODO: Your solution

    async def try_reload_engine(self) -> Recommender | None:
        try:
            return await asyncio.to_thread(self.reload_engine)
        except Exception:
            logger.exception("engine reload failed, keeping the current engine")
            return None

    async def wait_for_stop(self) -> bool:
        """Wait up to DEFAULT_POLL_INTERVAL_SEC for the stop event; True means stop."""
        # TODO: Your solution

    def drain_and_shutdown(self, engine: Recommender, timeout: float) -> bool:
        """Wait until `engine` has no `recommend` in flight or `timeout` seconds pass, then call
        `engine.shutdown()` once. True if it drained, False if the shutdown was forced."""
        # TODO: Your solution

    def retire(self, engine: Recommender) -> None:
        """Start `drain_and_shutdown(engine, DRAIN_TIMEOUT_SEC)` in a thread as a background task
        kept in `self.drain_tasks`; never wait for it here."""
        # TODO: Your solution

    async def poll_for_updates(self) -> None:
        """Until `wait_for_stop` says stop: on marker content other than `loaded_marker`, reload,
        swap under the lock, retire the old engine and set `loaded_marker`. A failed load retries
        next tick."""
        # TODO: Your solution

    async def start(self) -> None:
        """Read the marker, load the first engine (errors propagate), keep that marker as
        `loaded_marker`, then start the poll task."""
        # TODO: Your solution

    async def stop(self) -> None:
        """Stop the poll task, retire the current engine and await every drain."""
        # TODO: Your solution

    def recommend(self, num_candidates: int, context: RecContext) -> list[Candidate]:
        with self.engine_lock:
            engine = self.engine
            if engine is None:
                raise RuntimeError("runtime is not started")
            self.in_flight[engine] += 1
        try:
            return engine.recommend(num_candidates=num_candidates, context=context)
        finally:
            with self.engine_idle:
                self.in_flight[engine] -= 1
                if self.in_flight[engine] == 0:
                    del self.in_flight[engine]
                self.engine_idle.notify_all()

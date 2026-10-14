"""Helpers for the EngineRuntime tests: fake engines, snapshot files and a tick driver.

No test sleeps to wait for the runtime. The poll loop is driven one tick at a time by
`Ticks`, and request threads block on `threading.Event`s the test releases. Every wait
has a timeout only to bound a hung implementation."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import engine_server.runtime as runtime_module
from engine.bloom import BloomToken
from engine.interfaces import Candidate, RecContext, Recommender

WAIT_SEC = 5.0
CONTEXT = RecContext.from_request(
    user_id=1,
    current_time=None,
    context_item_ids=None,
    item_id_filter=BloomToken(),
    source_type="ST_Feed",
)


class FakeEngine(Recommender):
    """Answers `recommend` with [item_id], optionally blocking until `release` is set.
    `shutdown` counts its calls and records whether a request was still running."""

    def __init__(self, item_id: int, blocking: bool = False):
        self.item_id = item_id
        self.arrived = threading.Semaphore(0)
        self.release = threading.Event()
        if not blocking:
            self.release.set()
        self.running = 0
        self.shutdown_calls = 0
        self.shut_down_while_running = False
        self.shut_down = threading.Event()

    def recommend(self, num_candidates, context) -> list[Candidate]:
        self.running += 1
        self.arrived.release()
        self.release.wait(WAIT_SEC)
        self.running -= 1
        return [Candidate(item_id=self.item_id)]

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.shut_down_while_running |= self.running > 0
        self.shut_down.set()

    def __repr__(self) -> str:
        return f"FakeEngine({self.item_id})"


class Snapshot:
    """The two files the downloader maintains: the archive and its version marker."""

    def __init__(self, tmp_path: Path):
        self.archive = tmp_path / "archive.tar.gz"
        self.marker = tmp_path / "archive.version.json"

    def publish(self, content: bytes, version: str) -> None:
        """A complete download: the archive, then the marker."""
        self.archive.write_bytes(content)
        self.marker.write_text(json.dumps({"version": version}))


def make_runtime(
    monkeypatch, tmp_path: Path, engines: Sequence[FakeEngine | Exception]
) -> runtime_module.EngineRuntime:
    """An EngineRuntime whose `RecommenderEngine.startup` returns `engines` in order,
    with the paths in tmp_path and the published snapshot `v1`."""
    snapshot = Snapshot(tmp_path)
    snapshot.publish(b"v1", "v1")
    monkeypatch.setenv(
        runtime_module.LOCAL_ARCHIVE_PATH_ENV, snapshot.archive.as_posix()
    )
    monkeypatch.setenv(
        runtime_module.UPDATE_MARKER_PATH_ENV, snapshot.marker.as_posix()
    )
    monkeypatch.setattr(runtime_module, "DEFAULT_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(runtime_module, "DEFAULT_TIMEOUT_INTERVAL_SEC", 1.0)
    monkeypatch.setattr(runtime_module, "RecommenderEngine", StartupSequence(engines))
    return runtime_module.EngineRuntime({})


class StartupSequence:
    """Stands in for RecommenderEngine: `startup` returns the next engine, or raises it if it is an exception."""

    def __init__(self, engines: Sequence[FakeEngine | Exception]):
        self.engines: Iterator[FakeEngine | Exception] = iter(engines)
        self.calls = 0

    def startup(self, config):
        self.calls += 1
        engine = next(self.engines)
        if isinstance(engine, Exception):
            raise engine
        return engine


def startup_calls() -> int:
    """How many engines the runtime started (through the installed StartupSequence)."""
    startup = runtime_module.RecommenderEngine
    assert isinstance(startup, StartupSequence)
    return startup.calls


class Ticks:
    """Replaces `runtime.wait_for_stop`, so the poll loop runs one iteration per `step()`.

    The loop announces each call of `wait_for_stop`; `step()` lets it continue and waits
    for the next call, so everything the iteration does (reload, swap, retire) is done
    when `step()` returns. `stop()` on the runtime still ends the loop."""

    def __init__(self, runtime: runtime_module.EngineRuntime):
        self.runtime = runtime
        self.reached: asyncio.Queue[None] = asyncio.Queue()
        self.go: asyncio.Queue[None] = asyncio.Queue()
        setattr(runtime, "wait_for_stop", self.wait_for_stop)

    async def wait_for_stop(self) -> bool:
        self.reached.put_nowait(None)
        waiters = [asyncio.ensure_future(self.go.get())]
        if self.runtime.update_stop_event is not None:
            waiters.append(asyncio.ensure_future(self.runtime.update_stop_event.wait()))
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        for waiter in waiters:
            waiter.cancel()
        event = self.runtime.update_stop_event
        return event is not None and event.is_set()

    async def ready(self) -> None:
        """Wait until the loop reaches its first `wait_for_stop`."""
        await self.next_call()

    async def step(self, times: int = 1) -> None:
        for _ in range(times):
            self.go.put_nowait(None)
            await self.next_call()

    async def next_call(self) -> None:
        poll = self.runtime.update_task
        assert poll is not None, "start() did not create update_task"
        reached = asyncio.ensure_future(self.reached.get())
        await asyncio.wait(
            {reached, poll}, timeout=WAIT_SEC, return_when=asyncio.FIRST_COMPLETED
        )
        if not reached.done():
            reached.cancel()
            state = "ended" if poll.done() else "hangs"
            raise AssertionError(f"poll loop {state} instead of calling wait_for_stop")


async def start_request(
    runtime: runtime_module.EngineRuntime, engine: FakeEngine
) -> asyncio.Task:
    """A `runtime.recommend` call in a thread; returns once it is running on `engine`."""
    request = asyncio.create_task(asyncio.to_thread(runtime.recommend, 1, CONTEXT))
    assert await asyncio.to_thread(engine.arrived.acquire, timeout=WAIT_SEC), (
        "request did not reach the engine"
    )
    return request


async def wait_shut_down(engine: FakeEngine) -> bool:
    return await asyncio.to_thread(engine.shut_down.wait, WAIT_SEC)

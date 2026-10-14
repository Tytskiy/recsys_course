"""EngineRuntime (task engine_server_reload). Behaviours, one test or param id each:
reload_engine waits for the archive or times out; start loads the first engine (errors propagate)
and starts polling; wait_for_stop; drain_and_shutdown waits for in-flight requests up to a
timeout, then shuts down once; the poll loop reloads only on marker content other than the marker
read before the last load (a first marker after a start without one is new), retires the old
engine in the background and keeps polling on failures; a snapshot published during the first
load is picked up; stop ends polling (a failed poll task is logged, not raised), drains the last
engine and awaits every pending drain. Helpers: tests/runtime_harness.py."""

from __future__ import annotations

import asyncio
import logging
import threading

import engine_server.runtime as runtime_module
import pytest

from tests.runtime_harness import (
    CONTEXT,
    WAIT_SEC,
    FakeEngine,
    Snapshot,
    StartupSequence,
    Ticks,
    make_runtime,
    start_request,
    startup_calls,
    wait_shut_down,
)

EARLY_SHUTDOWN_GRACE_SEC = 0.2


@pytest.mark.parametrize(
    "load",
    [
        pytest.param(lambda runtime: runtime.reload_engine(), id="reload_engine"),
        pytest.param(lambda runtime: asyncio.run(runtime.start()), id="start_propagates_failed_first_load"),
    ],
)
def test_times_out_without_archive(monkeypatch, tmp_path, load) -> None:
    runtime = make_runtime(monkeypatch, tmp_path, [FakeEngine(1)])
    Snapshot(tmp_path).archive.unlink()
    monkeypatch.setattr(runtime_module, "DEFAULT_TIMEOUT_INTERVAL_SEC", 0.1)

    with pytest.raises(FileNotFoundError, match="Timed out waiting for archive"):
        load(runtime)


def test_reload_engine_waits_for_archive(monkeypatch, tmp_path) -> None:
    """The archive appears after a few polls, well before the 1 s timeout."""
    engine = FakeEngine(1)
    runtime = make_runtime(monkeypatch, tmp_path, [engine])
    snapshot = Snapshot(tmp_path)
    snapshot.archive.unlink()
    threading.Timer(0.05, snapshot.archive.write_bytes, args=[b"v1"]).start()

    assert runtime.reload_engine() is engine


def test_start_loads_engine_and_polls(monkeypatch, tmp_path) -> None:
    engine = FakeEngine(1)
    runtime = make_runtime(monkeypatch, tmp_path, [engine])

    async def scenario() -> None:
        await runtime.start()
        assert runtime.engine is engine
        assert runtime.recommend(1, CONTEXT)[0].item_id == 1
        assert runtime.update_task is not None and not runtime.update_task.done()
        await runtime.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("create_event", "set_event", "expected"),
    [
        pytest.param(False, False, True, id="no_event_means_stop"),
        pytest.param(True, True, True, id="event_set_means_stop"),
        pytest.param(True, False, False, id="poll_interval_passed_means_check_marker"),
    ],
)
def test_wait_for_stop(
    monkeypatch, tmp_path, create_event, set_event, expected
) -> None:
    runtime = make_runtime(monkeypatch, tmp_path, [])

    async def scenario() -> bool:
        if create_event:
            runtime.update_stop_event = asyncio.Event()
            if set_event:
                runtime.update_stop_event.set()
        return await asyncio.wait_for(runtime.wait_for_stop(), WAIT_SEC)

    assert asyncio.run(scenario()) is expected


def test_drain_idle_engine(monkeypatch, tmp_path) -> None:
    engine = FakeEngine(1)
    runtime = make_runtime(monkeypatch, tmp_path, [])

    assert runtime.drain_and_shutdown(engine, timeout=WAIT_SEC) is True
    assert engine.shutdown_calls == 1


def test_drain_waits_for_in_flight_request(monkeypatch, tmp_path) -> None:
    """The request picked the engine before it was swapped out, so it finishes on it first."""
    engine = FakeEngine(1, blocking=True)
    runtime = make_runtime(monkeypatch, tmp_path, [])
    runtime.engine = engine

    async def scenario() -> bool:
        request = await start_request(runtime, engine)
        runtime.engine = None
        drain = asyncio.create_task(
            asyncio.to_thread(runtime.drain_and_shutdown, engine, WAIT_SEC)
        )
        try:
            assert not await asyncio.to_thread(
                engine.shut_down.wait, EARLY_SHUTDOWN_GRACE_SEC
            )
        finally:
            engine.release.set()
        await request
        return await drain

    assert asyncio.run(scenario()) is True
    assert engine.shutdown_calls == 1
    assert not engine.shut_down_while_running


def test_drain_forces_shutdown_after_timeout(monkeypatch, tmp_path) -> None:
    """A stuck request must not keep the engine alive forever: after `timeout` it is shut down anyway."""
    engine = FakeEngine(1, blocking=True)
    runtime = make_runtime(monkeypatch, tmp_path, [])
    runtime.engine = engine

    async def scenario() -> bool:
        request = await start_request(runtime, engine)
        runtime.engine = None
        try:
            return await asyncio.to_thread(runtime.drain_and_shutdown, engine, 0.1)
        finally:
            engine.release.set()
            await request

    assert asyncio.run(scenario()) is False
    assert engine.shutdown_calls == 1
    assert engine.shut_down_while_running


@pytest.mark.parametrize(
    "versions",
    [
        pytest.param([b"v2"], id="new_content"),
        pytest.param([b"v2", b"v1"], id="back_to_earlier_content"),
    ],
)
def test_marker_change_swaps_engine(monkeypatch, tmp_path, versions) -> None:
    engines = [FakeEngine(i) for i in range(len(versions) + 1)]
    runtime = make_runtime(monkeypatch, tmp_path, engines)
    snapshot = Snapshot(tmp_path)

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        for content in versions:
            snapshot.publish(content, content.decode())
            await ticks.step()
        assert runtime.engine is engines[-1]
        assert runtime.recommend(1, CONTEXT)[0].item_id == engines[-1].item_id
        for old in engines[:-1]:
            assert await wait_shut_down(old)
        await runtime.stop()

    asyncio.run(scenario())
    assert [engine.shutdown_calls for engine in engines] == [1] * len(engines)


def test_first_marker_after_start_without_marker_swaps_engine(monkeypatch, tmp_path) -> None:
    v1, v2 = FakeEngine(1), FakeEngine(2)
    runtime = make_runtime(monkeypatch, tmp_path, [v1, v2])
    snapshot = Snapshot(tmp_path)
    snapshot.marker.unlink()

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        await ticks.step()
        assert runtime.engine is v1
        snapshot.publish(b"v2", "v2")
        await ticks.step()
        assert runtime.engine is v2
        await runtime.stop()

    asyncio.run(scenario())


def rewrite_same_marker(snapshot: Snapshot) -> None:
    snapshot.marker.write_text(snapshot.marker.read_text())


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(rewrite_same_marker, id="same_content_rewritten"),
        pytest.param(
            lambda snapshot: snapshot.marker.unlink(missing_ok=True),
            id="marker_deleted",
        ),
    ],
)
def test_no_reload_without_new_marker_content(monkeypatch, tmp_path, change) -> None:
    engine = FakeEngine(1)
    runtime = make_runtime(monkeypatch, tmp_path, [engine, FakeEngine(2)])

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        for _ in range(5):
            change(Snapshot(tmp_path))
            await ticks.step()
        assert runtime.engine is engine
        await runtime.stop()

    asyncio.run(scenario())
    assert startup_calls() == 1


def test_snapshot_published_during_first_load_is_picked_up(
    monkeypatch, tmp_path
) -> None:
    """v2 lands while `start` is still loading v1: the marker read before that load is v1, so
    the first tick sees v2 as new and reloads."""
    v1, v2 = FakeEngine(1), FakeEngine(2)
    snapshot = Snapshot(tmp_path)

    class PublishDuringFirstLoad(StartupSequence):
        def startup(self, config):
            if self.calls == 0:
                snapshot.publish(b"v2", "v2")
            return super().startup(config)

    runtime = make_runtime(monkeypatch, tmp_path, [])
    monkeypatch.setattr(
        runtime_module, "RecommenderEngine", PublishDuringFirstLoad([v1, v2])
    )

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        assert runtime.engine is v1
        await ticks.step()
        assert runtime.engine is v2
        await runtime.stop()

    asyncio.run(scenario())
    assert startup_calls() == 2


def test_failed_reload_keeps_engine_and_retries(monkeypatch, tmp_path) -> None:
    """A snapshot that fails to start must not kill hot reload: the current engine keeps
    serving and the same marker is retried on the next tick."""
    engines = [FakeEngine(1), KeyError("broken snapshot"), FakeEngine(2)]
    runtime = make_runtime(monkeypatch, tmp_path, engines)

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        Snapshot(tmp_path).publish(b"v2", "v2")
        await ticks.step()
        assert runtime.engine is engines[0]
        await ticks.step()
        assert runtime.engine is engines[2]
        await runtime.stop()

    asyncio.run(scenario())
    assert startup_calls() == 3


async def trigger(
    runtime: runtime_module.EngineRuntime, ticks: Ticks, how: str
) -> asyncio.Task | None:
    """Retire the current engine by a reload (returns None) or by `stop` (returns the stop task)."""
    if how == "marker_change":
        Snapshot(ticks.runtime.update_marker_path.parent).publish(b"v2", "v2")
        await ticks.step()
        return None
    return asyncio.create_task(runtime.stop())


@pytest.mark.parametrize("how", ["marker_change", "stop"])
def test_old_engine_shut_down_after_in_flight_request(
    monkeypatch, tmp_path, how
) -> None:
    old, new = FakeEngine(1, blocking=True), FakeEngine(2)
    runtime = make_runtime(monkeypatch, tmp_path, [old, new])

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        request = await start_request(runtime, old)
        try:
            stopping = await trigger(runtime, ticks, how)
            if how == "marker_change":
                assert runtime.recommend(1, CONTEXT)[0].item_id == 2
            assert not await asyncio.to_thread(
                old.shut_down.wait, EARLY_SHUTDOWN_GRACE_SEC
            )
        finally:
            old.release.set()
        assert (await request)[0].item_id == 1
        assert await wait_shut_down(old)
        await (stopping or runtime.stop())

    asyncio.run(scenario())
    assert old.shutdown_calls == 1
    assert not old.shut_down_while_running


@pytest.mark.parametrize("how", ["marker_change", "stop"])
def test_stuck_request_forces_shutdown_after_drain_timeout(
    monkeypatch, tmp_path, how
) -> None:
    """With a request that never returns, the old engine is shut down DRAIN_TIMEOUT_SEC later and
    `stop` still returns."""
    old = FakeEngine(1, blocking=True)
    runtime = make_runtime(monkeypatch, tmp_path, [old, FakeEngine(2)])
    monkeypatch.setattr(runtime_module, "DRAIN_TIMEOUT_SEC", 0.2)

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        request = await start_request(runtime, old)
        try:
            stopping = await trigger(runtime, ticks, how)
            assert await wait_shut_down(old)
            assert old.shut_down_while_running
            await asyncio.wait_for(stopping or runtime.stop(), WAIT_SEC)
        finally:
            old.release.set()
            await request

    asyncio.run(scenario())
    assert old.shutdown_calls == 1


def test_reload_not_blocked_by_draining_engine(monkeypatch, tmp_path) -> None:
    """v1 has a stuck request; v2 and then v3 still swap in, and idle v2 shuts down before v1."""
    v1, v2, v3 = FakeEngine(1, blocking=True), FakeEngine(2), FakeEngine(3)
    runtime = make_runtime(monkeypatch, tmp_path, [v1, v2, v3])
    snapshot = Snapshot(tmp_path)

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        request = await start_request(runtime, v1)
        try:
            snapshot.publish(b"v2", "v2")
            await ticks.step()
            assert runtime.engine is v2
            snapshot.publish(b"v3", "v3")
            await ticks.step()
            assert runtime.engine is v3
            assert await wait_shut_down(v2)
            assert not v1.shut_down.is_set()
        finally:
            v1.release.set()
        await request
        assert await wait_shut_down(v1)
        await runtime.stop()

    asyncio.run(scenario())
    assert [v1.shutdown_calls, v2.shutdown_calls, v3.shutdown_calls] == [1, 1, 1]


def test_stop_ends_polling_and_shuts_down_engine(monkeypatch, tmp_path) -> None:
    engine = FakeEngine(1)
    runtime = make_runtime(monkeypatch, tmp_path, [engine])

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        await asyncio.wait_for(runtime.stop(), WAIT_SEC)
        assert engine.shutdown_calls == 1
        assert runtime.update_task is not None and runtime.update_task.done()
        assert runtime.engine is None

    asyncio.run(scenario())
    with pytest.raises(RuntimeError, match="runtime is not started"):
        runtime.recommend(1, CONTEXT)


def test_stop_awaits_pending_drains(monkeypatch, tmp_path) -> None:
    """v1 is still draining (stuck request) when stop retires v2: stop returns only after both
    are shut down, v1 by DRAIN_TIMEOUT_SEC."""
    v1, v2 = FakeEngine(1, blocking=True), FakeEngine(2)
    runtime = make_runtime(monkeypatch, tmp_path, [v1, v2])
    monkeypatch.setattr(runtime_module, "DRAIN_TIMEOUT_SEC", 0.2)

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        request = await start_request(runtime, v1)
        try:
            Snapshot(tmp_path).publish(b"v2", "v2")
            await ticks.step()
            assert not v1.shut_down.is_set()
            await asyncio.wait_for(runtime.stop(), WAIT_SEC)
            assert (v1.shutdown_calls, v2.shutdown_calls) == (1, 1)
        finally:
            v1.release.set()
            await request

    asyncio.run(scenario())


def test_stop_logs_failed_poll_task(monkeypatch, tmp_path, caplog) -> None:
    """The poll task died on an error: stop logs it instead of raising and still shuts down."""
    engine = FakeEngine(1)
    runtime = make_runtime(monkeypatch, tmp_path, [engine])

    def broken_marker():
        raise OSError("marker unreadable")

    async def scenario() -> None:
        ticks = Ticks(runtime)
        await runtime.start()
        await ticks.ready()
        monkeypatch.setattr(runtime, "read_marker", broken_marker)
        ticks.go.put_nowait(None)
        poll = runtime.update_task
        assert poll is not None
        await asyncio.wait({poll}, timeout=WAIT_SEC)
        assert poll.done()
        await asyncio.wait_for(runtime.stop(), WAIT_SEC)

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert engine.shutdown_calls == 1
    assert caplog.records

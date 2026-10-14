"""engine_server app (task engine_server). Behaviours, one test or param id each:
the lifespan builds one EngineRuntime from the YAML config, starts it, keeps it in
app.state.runtime and stops it on shutdown; POST /recommend builds the RecContext from the
request, returns the runtime's ids in order and a page token; the token excludes earlier
pages; an unknown source_page is a 422. The runtime is a fake (tests/server_harness.py)."""

from __future__ import annotations

from datetime import datetime, timezone

import engine_server.app as app_module
import pytest
from engine.bloom import BloomToken
from fastapi.testclient import TestClient

from tests.server_harness import CONFIG, FakeRuntime, patch_app


@pytest.fixture
def client(monkeypatch, tmp_path):
    patch_app(monkeypatch, tmp_path)
    with TestClient(app_module.app) as client:
        yield client


def post(client: TestClient, **body):
    return client.post(
        "/recommend", json={"user_id": 42, "source_page": "ST_Feed", **body}
    )


def test_lifespan_starts_and_stops_one_runtime(monkeypatch, tmp_path) -> None:
    patch_app(monkeypatch, tmp_path)

    with TestClient(app_module.app) as client:
        post(client, num_candidates=1)
        post(client, num_candidates=2)
        [runtime] = FakeRuntime.instances
        assert app_module.app.state.runtime is runtime
        assert runtime.events == ["start"]

    assert runtime.events == ["start", "stop"]
    assert runtime.config == CONFIG


def test_recommend_returns_runtime_candidates_in_order(client) -> None:
    response = post(client, num_candidates=3)

    assert response.status_code == 200
    assert response.json()["candidates"] == [0, 1, 2]
    assert response.json()["next_page_token"]


def test_recommend_builds_context_from_request(client) -> None:
    post(
        client,
        current_time="2026-01-01T10:20:30+00:00",
        source_page="ST_Search",
        num_candidates=3,
        context_item_ids=[99],
    )

    [runtime] = FakeRuntime.instances
    num_candidates, context = runtime.calls[-1]
    assert num_candidates == 3
    assert context.user_id == 42
    assert context.source_type == "ST_Search"
    assert context.context_item_ids == [99]
    assert context.current_time == datetime(2026, 1, 1, 10, 20, 30, tzinfo=timezone.utc)


def test_recommend_context_defaults(client) -> None:
    """No current_time / context_item_ids / page_token: now, no context items, an empty filter."""
    before = datetime.now(timezone.utc)
    post(client, num_candidates=1)

    [runtime] = FakeRuntime.instances
    _, context = runtime.calls[-1]
    assert context.context_item_ids == []
    assert context.current_time >= before
    assert runtime.filtered == [set()]


def test_page_token_excludes_earlier_pages(client) -> None:
    first = post(client, num_candidates=2).json()
    second = post(client, num_candidates=3, page_token=first["next_page_token"]).json()
    third = post(client, num_candidates=4, page_token=second["next_page_token"]).json()

    assert [first["candidates"], second["candidates"], third["candidates"]] == [
        [0, 1],
        [2, 3, 4],
        [5, 6, 7, 8],
    ]
    [runtime] = FakeRuntime.instances
    assert runtime.filtered == [set(), {0, 1}, {0, 1, 2, 3, 4}]


def test_page_token_holds_returned_ids(client) -> None:
    """The token is BloomToken(page_token) plus every returned id, encoded."""
    token = post(client, num_candidates=2).json()["next_page_token"]

    bloom = BloomToken(token)
    assert 0 in bloom and 1 in bloom
    assert 2 not in bloom


def test_last_page_is_empty(client) -> None:
    first = post(client, num_candidates=10).json()
    last = post(client, num_candidates=5, page_token=first["next_page_token"])

    assert last.status_code == 200
    assert last.json()["candidates"] == []


def test_unknown_source_page_is_rejected(client) -> None:
    """source_page is validated against the engine's SourceType before the runtime is called."""
    response = post(client, source_page="home", num_candidates=1)

    assert response.status_code == 422
    [runtime] = FakeRuntime.instances
    assert runtime.calls == []

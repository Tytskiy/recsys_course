"""experiments_server: the spec of the two endpoints.

- PUT /experiments/{name} stores an experiment and returns its name; a second PUT replaces it.
- Groups must tile buckets 1..mod exactly; the schema errors and the three split errors are
  422, the split errors with the exact `detail` strings below. Group order in the body is free.
- GET /experiments/{name}/resolve: bucket = `salted_hash(user_id, salt, mod)` with the stored
  salt and mod; the group whose [range_start, range_stop] holds it wins (both ends inclusive);
  params = `default | overrides` (top-level keys only). Unknown experiment: 404.
"""

from __future__ import annotations

import experiments_server.app as app_module
import pytest

salted_hash = app_module.ExperimentRepository.salted_hash


def group(name: str, start: int, stop: int, overrides: dict | None = None) -> dict:
    return {
        "name": name,
        "range_start": start,
        "range_stop": stop,
        "overrides": overrides or {},
    }


def experiment(
    groups: list[dict], mod: int = 10, salt: str = "abc", default: dict | None = None
) -> dict:
    return {
        "salt": salt,
        "mod": mod,
        "default": default or {"filter_history": True},
        "groups": groups,
    }


def resolve(client, name: str, user_id: int) -> dict:
    response = client.get(f"/experiments/{name}/resolve", params={"user_id": user_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_upsert_returns_experiment_name(client) -> None:
    response = client.put(
        "/experiments/ST_Feed", json=experiment([group("all", 1, 10)])
    )
    assert response.status_code == 200
    assert response.json() == {"experiment_name": "ST_Feed"}


def test_resolve_merges_overrides_over_default(client) -> None:
    """Shallow merge: an override replaces the whole top-level value, other defaults stay."""
    body = experiment(
        [group("all", 1, 10, {"cg_schema": {"TopPop": 5}, "ranker": "v2"})],
        default={
            "filter_history": False,
            "cg_schema": {"TopPop": 10, "History": 10},
            "ranker": "v1",
        },
    )
    client.put("/experiments/ST_Upsale", json=body)

    assert resolve(client, "ST_Upsale", 42) == {
        "group_name": "all",
        "params": {"filter_history": False, "cg_schema": {"TopPop": 5}, "ranker": "v2"},
    }


def test_second_upsert_replaces_experiment(client) -> None:
    """PUT is an upsert: default, groups, salt and mod of the second body replace the first."""
    first = experiment([group("old", 1, 10, {"ranker": "v1"})], salt="old-salt")
    client.put("/experiments/ST_Feed", json=first)
    second_groups = [group(f"new{b}", b, b, {"ranker": "v2"}) for b in range(1, 5)]
    second = experiment(
        second_groups, mod=4, salt="new-salt", default={"filter_history": False}
    )
    assert client.put("/experiments/ST_Feed", json=second).status_code == 200

    for user_id in range(10):
        bucket = salted_hash(user_id=user_id, salt="new-salt", mod=4)
        assert resolve(client, "ST_Feed", user_id) == {
            "group_name": f"new{bucket}",
            "params": {"filter_history": False, "ranker": "v2"},
        }


@pytest.mark.parametrize(
    "bad_group",
    [
        pytest.param(
            group("control", 1, 10) | {"extra": "not-allowed"}, id="extra_field"
        ),
        pytest.param(
            group("control", 1, 10) | {"range_start": "one", "range_stop": [10]},
            id="wrong_type",
        ),
        pytest.param(group("control", 10, 1), id="range_stop_before_start"),
    ],
)
def test_upsert_rejects_invalid_schema(client, bad_group) -> None:
    """Given by models.py: the schema itself rejects these with 422."""
    response = client.put("/experiments/ST_Catalog", json=experiment([bad_group]))
    assert response.status_code == 422


def test_upsert_rejects_zero_mod(client) -> None:
    """mod is the bucket count; mod=0 would make resolve divide by zero."""
    response = client.put(
        "/experiments/ST_Feed", json=experiment([group("all", 1, 0)], mod=0)
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("groups", "error_message"),
    [
        pytest.param([], "groups must be non-empty", id="empty_groups"),
        pytest.param(
            [group("control", 2, 5), group("treatment", 6, 10)],
            "groups must fully cover the modulo range",
            id="first_start_after_1",
        ),
        pytest.param(
            [group("control", 1, 5), group("treatment", 6, 9)],
            "groups must fully cover the modulo range",
            id="last_stop_before_mod",
        ),
        pytest.param(
            [group("control", 1, 5), group("treatment", 6, 11)],
            "groups must fully cover the modulo range",
            id="last_stop_past_mod",
        ),
        pytest.param(
            [group("control", 1, 3), group("treatment", 6, 10)],
            "groups must be contiguous",
            id="gap_between_groups",
        ),
        pytest.param(
            [group("control", 1, 6), group("treatment", 5, 10)],
            "groups must be contiguous",
            id="overlapping_groups",
        ),
    ],
)
def test_upsert_rejects_invalid_split(client, groups, error_message) -> None:
    """Checks run in this order: non-empty, then the ends (1 and mod), then contiguity."""
    response = client.put("/experiments/ST_BrokenSplit", json=experiment(groups))
    assert response.status_code == 422
    assert response.json() == {"detail": error_message}


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param([group("all", 1, 1)], id="single_bucket_group"),
        pytest.param(
            [group("treatment", 6, 10), group("control", 1, 5)],
            id="groups_in_any_order",
        ),
    ],
)
def test_upsert_accepts_valid_split(client, groups) -> None:
    mod = max(g["range_stop"] for g in groups)
    response = client.put("/experiments/ST_Feed", json=experiment(groups, mod=mod))
    assert response.status_code == 200


def test_resolve_unknown_experiment_returns_404(client) -> None:
    response = client.get("/experiments/ST_Missing/resolve", params={"user_id": 1})
    assert response.status_code == 404
    assert response.json() == {"detail": "Experiment 'ST_Missing' not found"}


@pytest.mark.parametrize(
    ("bucket", "expected_group", "expected_ranker"),
    [
        pytest.param(1, "control", "v1", id="bucket_1_first_of_control"),
        pytest.param(5, "control", "v1", id="bucket_5_last_of_control"),
        pytest.param(6, "treatment", "v2", id="bucket_6_first_of_treatment"),
        pytest.param(10, "treatment", "v2", id="bucket_10_mod_last_of_treatment"),
    ],
)
def test_resolve_routes_bucket_to_group(
    client, monkeypatch, bucket, expected_group, expected_ranker
) -> None:
    """Range ends are inclusive: control is [1, 5], treatment [6, 10]. resolve must call
    `ExperimentRepository.salted_hash`, which this test replaces with a constant bucket."""
    monkeypatch.setattr(
        app_module.ExperimentRepository,
        "salted_hash",
        staticmethod(lambda user_id, salt, mod: bucket),
    )
    body = experiment(
        [
            group("control", 1, 5, {"ranker": "v1"}),
            group("treatment", 6, 10, {"ranker": "v2"}),
        ],
        default={"filter_history": True, "ranker": "v0"},
    )
    client.put("/experiments/ST_Split", json=body)

    assert resolve(client, "ST_Split", 13) == {
        "group_name": expected_group,
        "params": {"filter_history": True, "ranker": expected_ranker},
    }


def test_resolve_hashes_with_stored_salt_and_mod(client) -> None:
    """Without patching: each experiment uses its own salt and mod, and every bucket maps to
    exactly the group holding it. Group i owns bucket i, so the group name is the bucket."""
    setups = {"ST_Feed": ("feed-2026", 7), "ST_Catalog": ("catalog", 3)}
    for name, (salt, mod) in setups.items():
        groups = [group(str(bucket), bucket, bucket) for bucket in range(1, mod + 1)]
        client.put(f"/experiments/{name}", json=experiment(groups, mod=mod, salt=salt))

    for user_id in range(20):
        for name, (salt, mod) in setups.items():
            bucket = salted_hash(user_id=user_id, salt=salt, mod=mod)
            assert resolve(client, name, user_id)["group_name"] == str(bucket), (
                name,
                user_id,
            )

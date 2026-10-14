"""ValkeyFeatureStorage: `start` builds the client with `get_client()`; values round-trip
through JSON; every (user_id, item_id) shape is its own key (`build_hash_key`); missing
features are omitted; `bulk_extract` returns one dict per requested item under one user;
`update` merges into a key; `delete` returns the number of removed features; readonly
storages refuse writes; `close` closes the client."""

from __future__ import annotations

import pytest
from engine.storages import ValkeyFeatureStorage

KEY_SHAPES = {
    "user_only": {"user_id": 1},
    "item_only_same_number_as_user": {"item_id": 1},
    "user_and_item": {"user_id": 1, "item_id": 1},
    "other_user_same_item": {"user_id": 2, "item_id": 1},
    "same_user_other_item": {"user_id": 1, "item_id": 2},
}


def test_start_uses_get_client(valkey_env):
    storage = ValkeyFeatureStorage.start({"readonly": False})
    storage.update({"f": 1}, item_id=5)
    assert valkey_env.dbsize() == 1


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(42, id="int"),
        pytest.param(3.14, id="float"),
        pytest.param("hello", id="string"),
        pytest.param(True, id="bool"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param({"nested": [1, "a"]}, id="dict"),
    ],
)
def test_values_round_trip(feature_storage, value):
    feature_storage.update({"f": value}, user_id=123, item_id=456)
    assert feature_storage.extract(["f"], user_id=123, item_id=456) == {"f": value}


@pytest.mark.parametrize("shape", KEY_SHAPES)
def test_each_key_shape_is_isolated(feature_storage, shape):
    for value, key in enumerate(KEY_SHAPES.values()):
        feature_storage.update({"f": value}, **key)
    assert feature_storage.extract(["f"], **KEY_SHAPES[shape]) == {"f": list(KEY_SHAPES).index(shape)}


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        pytest.param(["a", "c"], {"a": 1, "c": 3}, id="subset"),
        pytest.param(["a", "missing"], {"a": 1}, id="missing_feature_omitted"),
        pytest.param(["missing"], {}, id="only_missing"),
        pytest.param([], {}, id="no_features"),
    ],
)
def test_extract(feature_storage, features, expected):
    feature_storage.update({"a": 1, "b": 2, "c": 3}, user_id=7, item_id=8)
    assert feature_storage.extract(features, user_id=7, item_id=8) == expected


def test_extract_unknown_key_is_empty(feature_storage):
    assert feature_storage.extract(["a"], user_id=999) == {}


@pytest.mark.parametrize(
    ("features", "item_ids", "user_id", "expected"),
    [
        pytest.param(["f1", "f2"], [1, 2], 123, {1: {"f1": "a"}, 2: {"f1": "b", "f2": 2}}, id="per_item"),
        pytest.param(["f1"], [2, 1], 123, {2: {"f1": "b"}, 1: {"f1": "a"}}, id="items_requested_in_reverse"),
        pytest.param(["f1"], [1, 3], 123, {1: {"f1": "a"}, 3: {}}, id="unknown_item_empty_dict"),
        pytest.param(["f1"], [1], None, {1: {"f1": "x"}}, id="item_only_keys"),
        pytest.param(["f1"], [1], 456, {1: {}}, id="other_user_sees_nothing"),
        pytest.param(["f1"], [], 123, {}, id="no_items"),
        pytest.param([], [1, 2], 123, {1: {}, 2: {}}, id="no_features"),
    ],
)
def test_bulk_extract(feature_storage, features, item_ids, user_id, expected):
    feature_storage.update({"f1": "a"}, user_id=123, item_id=1)
    feature_storage.update({"f1": "b", "f2": 2}, user_id=123, item_id=2)
    feature_storage.update({"f1": "x"}, item_id=1)
    assert feature_storage.bulk_extract(features, item_ids=item_ids, user_id=user_id) == expected


@pytest.mark.parametrize(
    ("second_update", "expected"),
    [
        pytest.param({"a": "new"}, {"a": "new", "b": 2}, id="overwrites_given_keeps_others"),
        pytest.param({"c": 3}, {"a": 1, "b": 2, "c": 3}, id="adds_new_feature"),
        pytest.param({}, {"a": 1, "b": 2}, id="empty_update_is_noop"),
    ],
)
def test_update_merges(feature_storage, second_update, expected):
    feature_storage.update({"a": 1, "b": 2}, user_id=1)
    feature_storage.update(second_update, user_id=1)
    assert feature_storage.extract(["a", "b", "c"], user_id=1) == expected


@pytest.mark.parametrize(
    ("features", "deleted", "left"),
    [
        pytest.param(["a"], 1, {"b": 2, "c": 3}, id="one"),
        pytest.param(["a", "b"], 2, {"c": 3}, id="two"),
        pytest.param(["a", "missing"], 1, {"b": 2, "c": 3}, id="missing_not_counted"),
        pytest.param([], 0, {"a": 1, "b": 2, "c": 3}, id="no_features"),
    ],
)
def test_delete(feature_storage, features, deleted, left):
    feature_storage.update({"a": 1, "b": 2, "c": 3}, user_id=1, item_id=2)
    assert feature_storage.delete(features, user_id=1, item_id=2) == deleted
    assert feature_storage.extract(["a", "b", "c"], user_id=1, item_id=2) == left


def test_delete_is_scoped_to_its_key(feature_storage):
    feature_storage.update({"a": 1}, user_id=1, item_id=2)
    feature_storage.update({"a": 1}, item_id=2)
    feature_storage.delete(["a"], user_id=1, item_id=2)
    assert feature_storage.extract(["a"], item_id=2) == {"a": 1}


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda s: s.update({"key": "value"}, user_id=123), id="update"),
        pytest.param(lambda s: s.delete(["key"], user_id=123), id="delete"),
    ],
)
def test_readonly_refuses_writes(readonly_storage, write):
    with pytest.raises(ValueError, match="readonly"):
        write(readonly_storage)


def test_readonly_can_read(valkey_env):
    ValkeyFeatureStorage.start({"readonly": False}).update({"f": 1}, item_id=3)
    assert ValkeyFeatureStorage.start({"readonly": True}).extract(["f"], item_id=3) == {"f": 1}


def test_close_closes_client(feature_storage, valkey_env, monkeypatch):
    closed = []
    monkeypatch.setattr(valkey_env, "close", lambda: closed.append(True))
    feature_storage.close()
    assert closed == [True]

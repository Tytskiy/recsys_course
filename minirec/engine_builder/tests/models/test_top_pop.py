"""train_top_pop: writes `{item_id: count}` as JSON (keys are decimal ids, u64 included)
into `workdir / filename`; counts only rows of the requested `action_type`;
items without such a row are absent; returns TrainingArtifacts(files=[filename])."""

import orjson
import polars as pl
import pytest
from tasks.models.top_pop import train_top_pop

BIG_ID = 2**64 - 5

ACTIONS = pl.DataFrame(
    {
        "user_id": [1, 2, 1, 2, 3, 1, 2, 3, 3],
        "item_id": [1, 1, 1, 2, BIG_ID, BIG_ID, BIG_ID, 3, 3],
        "action_type": [
            "AT_View",
            "AT_View",
            "AT_CartUpdate",
            "AT_CartUpdate",
            "AT_CartUpdate",
            "AT_CartUpdate",
            "AT_CartUpdate",
            "AT_Purchase",
            "AT_Purchase",
        ],
    },
    schema={"user_id": pl.UInt64, "item_id": pl.UInt64, "action_type": pl.String},
)


def read_counts(actions: pl.DataFrame, tmp_path, **kwargs) -> dict[str, int]:
    artifacts = train_top_pop(actions, tmp_path, **kwargs)
    return orjson.loads((tmp_path / artifacts.files[0]).read_bytes())


@pytest.mark.parametrize(
    ("action_type", "expected"),
    [
        pytest.param("AT_CartUpdate", {"1": 1, "2": 1, str(BIG_ID): 3}, id="cart_update_with_u64_id"),
        pytest.param("AT_View", {"1": 2}, id="view_other_types_ignored"),
        pytest.param("AT_Purchase", {"3": 2}, id="purchase"),
        pytest.param("AT_Click", {}, id="no_rows_of_type_empty"),
    ],
)
def test_counts_per_action_type(tmp_path, action_type, expected):
    assert read_counts(ACTIONS, tmp_path, action_type=action_type) == expected


def test_writes_named_file_and_returns_it(tmp_path):
    artifacts = train_top_pop(ACTIONS, tmp_path, filename="pop.json")
    assert artifacts.files == ["pop.json"]
    assert (tmp_path / "pop.json").exists()


def test_default_action_type_is_cart_update(tmp_path):
    assert read_counts(ACTIONS, tmp_path) == {"1": 1, "2": 1, str(BIG_ID): 3}


@pytest.mark.parametrize(
    ("action_type", "item_id", "expected"),
    [
        pytest.param("AT_CartUpdate", "11172727313663880600", 128, id="fixture_cart_update"),
        pytest.param("AT_CartUpdate", "11618334071743605794", 44, id="fixture_cart_update_2"),
        pytest.param("AT_Click", "11172727313663880600", 19, id="fixture_click"),
        pytest.param("AT_Click", "12327270727736893607", 8, id="fixture_click_2"),
    ],
)
def test_counts_on_fixture_log(dm_action, tmp_path, action_type, item_id, expected):
    assert read_counts(dm_action, tmp_path, action_type=action_type)[item_id] == expected


@pytest.mark.parametrize(
    ("action_type", "num_rows"),
    [
        pytest.param("AT_CartUpdate", 2612, id="fixture_cart_update_rows"),
        pytest.param("AT_Click", 312, id="fixture_click_rows"),
    ],
)
def test_counts_sum_to_rows_of_type(dm_action, tmp_path, action_type, num_rows):
    assert sum(read_counts(dm_action, tmp_path, action_type=action_type).values()) == num_rows

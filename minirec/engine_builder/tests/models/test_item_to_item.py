"""ItemToItemTrainer.train / train_item_to_item: for every item, the `num_items_per_list`
other items with the highest cosine similarity of their (normalized) name embeddings,
ordered by similarity desc, then item id asc; the anchor itself is never in its list;
fewer other items than `num_items_per_list` gives all of them. `train_item_to_item`
returns them as `item_profiles = {id: {feature_name: list}}`. The embedder is faked."""

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
import tasks.models.item_to_item as item_to_item_module
from sklearn.feature_extraction.text import TfidfVectorizer

TEST_DATA_PATH = Path(__file__).parents[1] / "test_data"


def unit(degrees: float) -> list[float]:
    return [float(np.cos(np.radians(degrees))), float(np.sin(np.radians(degrees)))]


# 2-d unit vectors, so similarity = cos(angle difference). c and d are the same vector.
VECTORS = {"a": unit(0), "b": unit(10), "c": unit(30), "d": unit(30), "e": unit(90)}
ITEMS = pl.DataFrame({"item_id": [1, 2, 3, 4, 5], "item_name": ["a", "b", "c", "d", "e"]})


class VectorEmbedder:
    def embed(self, text: str | list[str]) -> np.ndarray:
        text = [text] if isinstance(text, str) else list(text)
        return np.array([VECTORS[t] for t in text], dtype=np.float32)


class TfidfEmbedder:
    def __init__(self):
        self.text_embedder: TfidfVectorizer = joblib.load(TEST_DATA_PATH / "embedder.pkl")

    def embed(self, text: str | list[str]) -> np.ndarray:
        text = [text] if isinstance(text, str) else text
        return np.asarray(self.text_embedder.transform(text).todense())


@pytest.fixture
def vector_embedder(monkeypatch):
    monkeypatch.setattr(item_to_item_module, "TextEmbedder", VectorEmbedder)


@pytest.mark.parametrize(
    ("item_id", "expected"),
    [
        pytest.param(1, [2, 3], id="highest_similarity_first_tie_by_id"),
        pytest.param(3, [4, 2], id="anchor_excluded_identical_vector_kept"),
        pytest.param(4, [3, 2], id="anchor_excluded_other_direction"),
        pytest.param(5, [3, 4], id="tie_by_item_id_asc"),
    ],
)
def test_lists_by_similarity(vector_embedder, item_id, expected):
    lists = item_to_item_module.ItemToItemTrainer(num_items_per_list=2).train(ITEMS)
    assert set(lists) == {1, 2, 3, 4, 5}
    assert lists[item_id] == expected


@pytest.mark.parametrize(
    ("num_items_per_list", "expected"),
    [
        pytest.param(1, [2], id="one"),
        pytest.param(4, [2, 3, 4, 5], id="exactly_all_other_items"),
        pytest.param(5, [2, 3, 4, 5], id="one_past_all_other_items"),
    ],
)
def test_list_length(vector_embedder, num_items_per_list, expected):
    lists = item_to_item_module.ItemToItemTrainer(num_items_per_list=num_items_per_list).train(ITEMS)
    assert lists[1] == expected


def test_train_item_to_item_returns_item_profiles(vector_embedder):
    artifacts = item_to_item_module.train_item_to_item(ITEMS, feature_name="similar", num_items_per_list=2)
    assert artifacts.item_profiles[1] == {"similar": [2, 3]}
    assert set(artifacts.item_profiles) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize(
    ("item_id", "expected"),
    [
        # shares no TF-IDF word with any product: all similarities 0, so the 10 lowest other ids
        pytest.param(
            483595287616760637,
            [170252366898875931, 920359474935463889, 1625303204762302691, 2016012677920539891, 2476482649068877701,
             2993989429751501082, 3251178016424488087, 3885658100512376707, 4060838321469170071, 4179305647534841727],
            id="no_similar_items_ids_asc",
        ),
        # "Cornichon cucumbers": three similar items (.31, .24, .07), then similarity-0 items by id
        pytest.param(
            7428439742475507501,
            [6352056596100857023, 2476482649068877701, 4060838321469170071, 170252366898875931, 483595287616760637,
             920359474935463889, 1625303204762302691, 2016012677920539891, 2993989429751501082, 3251178016424488087],
            id="similar_items_then_zero_ties_by_id",
        ),
    ],
)
def test_train_item_to_item_on_fixture(dm_product, monkeypatch, item_id, expected):
    monkeypatch.setattr(item_to_item_module, "TextEmbedder", TfidfEmbedder)
    profiles = item_to_item_module.train_item_to_item(dm_product).item_profiles
    assert profiles[item_id]["item_to_item"] == expected

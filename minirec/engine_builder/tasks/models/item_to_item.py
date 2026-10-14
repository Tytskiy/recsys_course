from itertools import batched
from typing import Generator, Iterable, Sequence

import numpy as np
import polars as pl
import torch
from engine.interfaces import ID
from transformers import AutoModel, AutoTokenizer

from tasks.models.artifacts import TrainingArtifacts


class TextEmbedder:
    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        compile_model: bool = True,
    ):
        """Load the `transformers` model and tokenizer for `model_id`;
        wrap the model in `torch.compile` when `compile_model` is set."""
        self.model = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if compile_model:
            self.model = torch.compile(self.model)
    
    def embed(self, text: str | list[str]) -> np.ndarray:
        """Processes batch of texts to normalized embeddings."""
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            model_output = self.model(**t)
        mask = t["attention_mask"].unsqueeze(-1).to(model_output.last_hidden_state.dtype)
        embeddings = (model_output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.cpu().numpy()


class ItemToItemTrainer:
    def __init__(
        self,
        num_items_per_list: int = 10,
    ):
        self.num_items_per_list = num_items_per_list
        self.model = TextEmbedder()

    # TODO: Your solution

    def train(self, items: pl.DataFrame) -> dict[ID, list[ID]]:
        """Choose top-K items, closest by emedding distance, to each item."""

        assert len(items.columns) == 2, "DataFrame schema should be [ID, text]"
        # TODO: Your solution


def train_item_to_item(
    dim_product: pl.DataFrame,
    feature_name: str = "item_to_item",
    num_items_per_list: int = 10,
) -> TrainingArtifacts:
    """Calculates top-K item lists for each item via text content embeddings,
    returned as item profiles `{ID: {feature_name: list}}`.
    """
    trainer = ItemToItemTrainer(num_items_per_list=num_items_per_list)
    items = (
        dim_product
        .select("item_id", "item_name")
        .unique()
        .sort("item_id")
    )
    lists = trainer.train(items)
    return TrainingArtifacts(item_profiles={iid: {feature_name: lst} for iid, lst in lists.items()})

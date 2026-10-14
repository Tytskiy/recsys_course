from __future__ import annotations

import numpy as np
import pytest
from engine.bloom import BloomToken


@pytest.mark.parametrize(
    ("added", "queried"),
    [
        pytest.param(np.int64(5), 5, id="numpy_int64_matches_int"),
        pytest.param(np.uint64(2**64 - 1), 2**64 - 1, id="max_uint64_id"),
    ],
)
def test_seen_item_matches_after_round_trip(added, queried):
    """CGs yield IDs as NumPy or Python ints; a seen item must match either way
    after the token goes to the client and back."""
    token = BloomToken()
    token.add(added)

    assert queried in BloomToken(token.encode())

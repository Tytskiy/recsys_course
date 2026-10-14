# SLIM to History Candidate Generator

## About the task
SLIM to History is a candidate generator that uses the user's history and, for each candidate item, computes the similarity of the candidate to the user's history with the item-to-item weight matrix obtained when training the model. The candidate generator returns a more similar item (with a larger total weight) earlier.

Training SLIM correctly and fast is also not the easiest task for Python code; for training, use the `fastslim` library. In this task you only need to implement inference that supports using recent history at runtime: this way we can recommend something to relatively new users without retraining.

## Statement
Implement the `SLIMToHistory` candidate generator in [`engine/cg/slim.py`](../../../engine/src/engine/cg/slim.py).
- In `start_component(config, deps)`, build the component from the already validated `SLIMToHistoryConfig` config (fields `action_type`, `mappings_path`, `weights_path`): read the mappings and the weight matrix from the snapshot through `deps.engine_data` (`read_json`, `read_sparse_npz`).
- The action type from the history must match the one the model was trained on; it is set by the `action_type` parameter.
- Iterate over the history and, with the `ID -> idx` mappings, build the user history vector in the original space.
- Alternatively, you can skip building a sparse vector and sum the weights directly during the iteration.
- The history is binary: a repeated item in the history does not increase its contribution. Items that are not in the mapping are skipped.
- Return the top-`num_candidates` items by the sum of weights (no more than the catalog size). Items from the history are not excluded. The order of items with equal scores is not checked by the tests.


## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/cgs/test_slim.py)
```

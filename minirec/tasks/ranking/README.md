# Ranking & logging

In the third week you will close the loop of learned ranking: the engine logs candidates → the offline pipeline replays historical requests through the engine and labels the logged candidates with the users' future actions → a CatBoost model is trained and appended to the snapshot on S3 → the runtime downloads it and ranks candidates with `ModelRanker`.

## Common theme: point-in-time
Everything the model sees must be known at request time, and everything it learns from must happen after it.
- Features strictly before the request. The replay in `build_ranker.py` builds the snapshot for the requests of day `D` from days `< D`. If it included day `D`, the `item:*`/`user:*` counters would already contain the very add-to-carts and purchases that later become labels.
- Labels strictly after the request, within the `window_size` window (forward `join_asof`, task `build_pool`).
- Train/test split by time, not at random (`build_pool`).
- Positions are also "from the past": clicks at the top of the list are partly explained by the position, not by relevance. The click model (`build_pool`) accounts for this explicitly. `mixer_position` and `position` are logged, but must not be model features.

## Tasks
1. [`candidate_logger`](./candidate_logger/README.md) - two-level candidate logging, including `num_served`. This is the input data for all the following tasks.
2. [`build_pool`](./build_pool/README.md) - JSONL logs → point-in-time labeling → dataset → split by time, plus the positional click model (EM) and IPW weights for positives.
3. [`model_ranker`](./model_ranker/README.md) - the trained model in the runtime via `ModelWrapper` and `factor_schema`.
4. [`mixigen`](./mixigen/README.md) (optional) - a contextual candidate mixer: a model prescores `(CG, position)` slots and sets quotas for each CG.

The recommended order is 1 → 2; tasks 3 and 4 are independent. The tasks' tests do not depend on each other: where someone else's functionality is needed, the tests use their own stubs and the seeded simulator [`engine_builder/tests/ranking/sim.py`](../../engine_builder/tests/ranking/sim.py).

## Ready-made pipeline
[`engine_builder/build_ranker.py`](../../engine_builder/build_ranker.py) is provided in full and uses your functions: `load_actions → collect_logged_features (replay) → PoolBuilder → theta per surface → ipw_weights → train_catboost → add_files_to_archive`. It needs the `docker compose` stack running (S3, Valkey, Qdrant).

Run it after `build_recommender.py`, which rebuilds the snapshot and drops the ranker from it, with `--last-dt` equal to its `--latest-dt`. The first replay day is built from the previous day's data, so take `--first-dt` one day earlier than the window you need. Copy the `default_feature_value` value from `ranker_config.yaml` into the `ModelRanker` config, otherwise the same candidate gets different features offline and online.

## Checking
The commands for checking each task locally are in its README.

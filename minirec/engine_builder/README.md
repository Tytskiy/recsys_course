# engine builder

A toy offline data processing pipeline for recommenders built with the [`engine`](../engine/README.md) framework. It works with MinIO but, for simplicity and clarity, contains no orchestration or data partitioning.

## Description
This module contains the code for building the snapshot of the `RecommenderEngine` recommendation engine.
- Training and uploading the files of different components
- Pool building, training and metrics for ranking and candidate generation models
- Computing features and updating profiles in key-value storages
- CLI scripts for updating the recommender using a slice of logs up to a given date

## How to build your snapshot
To run the service locally in a container, you first need to build its snapshot and upload it to object storage.
1. Start MinIO, Qdrant and Valkey in local containers.
```bash
docker compose up -d storages minio-setup
```
2. Upload all the required logs and files to object storage.
Two files are mandatory: the actions `s3://rec-engine/data/lavka/dm_action.parquet` and the product information reference `s3://rec-engine/data/lavka/dm_product.parquet`. You can download them [here](https://disk.yandex.ru/d/KvyEFHD-sjHSAg); figuring out the data schema and how to put the files into object storage is up to you.

3. Run the snapshot build graph:
```bash
cd engine_builder
uv run build_recommender.py --latest-dt <date>
```

## Offline ranker training

The [`build_ranker.py`](build_ranker.py) script replays the `dm_action` logs through a live `RecommenderEngine`, writes the results to `CANDIDATES_LOG_PATH` (`candidates.jsonl`) and then builds a CatBoost pool from them with [`PoolBuilder`](tasks/ranking/pool_builder.py).

Run it after `build_recommender.py`: the trained `ranker.cbm` and `factor_schema.json` are added to the snapshot `s3://rec-engine/service/v1/archive.tar.gz`, from which `ModelRanker` reads them. The next `build_recommender.py` run rebuilds the snapshot without them. During the replay, the daily snapshots are written to a separate `s3://rec-engine/service/replay/archive.tar.gz`, while the profiles in Valkey stay in their state as of `--last-dt`: pass the same `--last-dt` as the `--latest-dt` of `build_recommender.py`. The candidate log is cleared before every replay.

The `candidates.jsonl` format: for each request, one JSONL record is written with the fields `request_id`, `user_id`, `request_time`, `num_candidates`, `num_served` (how many candidates were given to the user after the cut), **`user_profile`** (user-level scalar features: `user:*`, `source_type`, computed exactly once in `SimpleManager.hydrate_context`) and `candidates` (a list of `{item_id, factors}` with item-level factors: `item:*`). Embeddings (iALS vectors and the like) are filtered out by `CandidateLogger.filter_factors` so as not to bloat the log. `parse_log_json` does not read a log in the old format (without `num_served` and `user_profile`, for example `datasets/candidates.jsonl`): rebuild it by running `build_ranker.py`.

`PoolBuilder.parse_log_json` expands a record into one row per candidate and mixes `user_profile` into every row with a `**` spread next to `candidate["factors"]`. Therefore, the `num_features` / `cat_features` fields in `ranker_config.yaml` can contain `item:*`, `user:*` and `source_type` keys at the same time.

`context_features` in `ranker_config.yaml` is the list of features that `ModelRanker` must read at runtime from `context.user_profile` rather than from `candidate.factors`. The training script saves this mapping to `factor_schema.json` as `[feature_name, is_context_factor]` pairs.

A minimal `ranker_config.yaml` example with one item feature and one user feature:

```yaml
num_features:
  - "item:num_views"
  - "user:num_views"
positive_actions:
  - "AT_CartUpdate"
  - "AT_Purchase"
negative_actions:
  - "AT_View"
window_size: 1800
context_features:
  - "user:num_views"
```

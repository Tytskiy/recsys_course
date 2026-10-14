# Build Pool

## About the task
To train a ranking model, you need to turn the `CandidateLogger` logs into a CatBoost `Pool` so that no label and no feature "peeks" into the future, and clicks from the top positions do not outweigh everything else. The task has two parts.

**`PoolBuilder`** in [`engine_builder/tasks/ranking/pool_builder.py`](../../../engine_builder/tasks/ranking/pool_builder.py): reading the logs, point-in-time labeling, the dataset and the time-based split. `parse_log_json` (expanding a log record into per-candidate rows, adding `position` and `served`) and `build_cbm_pool` (building the `Pool`) are already written.

**Position-based click model** in [`engine_builder/tasks/ranking/click_model.py`](../../../engine_builder/tasks/ranking/click_model.py). A user clicks an item only if they scrolled down to it (examination) and the item suits them (relevance): `P(click | item i at position k) = theta[k] * gamma[i]` (position-based model). Naive CTR by position mixes both effects: the old ranker puts relevant items at the top, and the top positions look "more clickable" than they are. The EM algorithm separates `theta` and `gamma` if the same items were shown at different positions. The estimated `theta` gives weights `1 / theta[p]` for positives (inverse propensity weighting): a click at the bottom of the feed means more than a click at the top. Aggregating the `dm_action` logs into counters `(item_id, position) -> impressions, clicks` is done by the ready `pbm_counts`.

### Point-in-time join
A candidate's target is the user's best action on this item that happened after the impression within a `window_size`-second window (30 minutes by default). Looking only forward is critical; otherwise information from the future relative to the request time leaks into training. If there are several actions in the window, take the one with the highest priority: `AT_View < AT_Click < AT_CartUpdate < AT_Purchase`. Together with the action type, we save where it was observed: `label_position` (`position_in_request` in the historical request) and `label_source_type` (the surface). Purchases do not have them (`null`).

### Dataset and split
Keep only requests with at least one positive (otherwise there is nothing to learn) and up to 9 negatives per request. `target` is `1.0/0.0`, `group_id` is the string `request_id`. The test set is always the later requests: a random split would put the user's future into training and their past into the test set.

## Statement
`PoolBuilder`:
- `read_logs`: read the file line by line with `orjson.loads`, skipping empty lines, and expand the records with `parse_log_json`. Types: `user_id`, `item_id` → `UInt64`, `request_time` → `Datetime("us")` without a time zone, but in UTC: the logger writes the time with an offset (`...+00:00`), so first convert it to UTC and then drop the time zone (`str.to_datetime(..., time_zone="UTC")` + `dt.replace_time_zone(None)`).
- `join_asof_actions_to_features`: `pl.DataFrame.join_asof(strategy="forward", by=["user_id", "item_id"])` for each `action_type` from `ACTION_PRIORITY`, filter `timestamp <= request_timestamp + window_size`, then the best action per `(request_id, item_id)` pair (sort by priority and `unique(keep="first")`). Add `label_position` and `label_source_type` from the chosen action; do not touch the log's own `source_type` (from `user_profile`). Drop the helper columns.
- `build_dataset`: filter by `action_type ∈ positive_actions ∪ negative_actions`, drop requests without positives, up to 9 negatives per request via `sample_per_partition`, `target` and `group_id = str(request_id)`. The output is a deduplicated dataframe sorted by `group_id`.
- `train_test_split_by_time`: order the requests by their earliest `request_time`; the first `int(n * (1 - test_size))` go to train, the rest to test; rows of one request are not split.

Click model:
- `estimate_pbm(counts, max_pos, n_iter)`: exactly `n_iter` EM iterations starting from `theta = gamma = 0.5`; on each, first record the log-likelihood of the current parameters, then the E-step and the M-step (both updates from one E-step). Return `theta / theta[0]`, normalizing once, after the last iteration. A row without non-clicks (`clicks == impressions`) contributes nothing to the non-click terms, even if `theta * gamma == 1`. The formulas are in the docstring.
- `ipw_weights(df, theta, clip)`: for a positive at position `p` the weight is `min(1 / theta[min(p, len(theta)) - 1], clip)`; for negatives and for positives without a position, 1.0.

## Tips
- The ready `Pool` is the input of the `train_catboost` function; the full pipeline is in [`engine_builder/build_ranker.py`](../../../engine_builder/build_ranker.py): it estimates `theta` separately for each surface (`source_type`) on the training window, weights the positives and prints the `theta` curves next to the naive CTR. On the full data the search curve is steep (`theta` at the 10th position is about 0.15) and the catalog curve is flat (about 0.67), so surfaces must not be mixed.
- `theta` here is the probability of examination given that the impression was logged as `AT_View`, not a "real" look by the user.
- IPW improves calibration (the estimate of the click probability), not necessarily the order: if the old ranker already sorted by relevance, the position effect does not spoil the order, and the weights only add variance. The tests check calibration.
- If the model does not beat the original candidate order, most likely `build_dataset` loses positives, or join-asof finds no actions inside the window.
- The click model tests use the seeded simulator [`engine_builder/tests/ranking/sim.py`](../../../engine_builder/tests/ranking/sim.py) with a known `theta`.

## How to check yourself
```bash
(cd engine_builder && uv run pytest tests/ranking/test_pool_builder.py tests/ranking/test_click_model.py)
```

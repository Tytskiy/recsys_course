# Mixigen

## About the task
`SchemaCandidateMixer` takes a fixed schema `(cg_name, num_candidates)` and pulls the given number of candidates from each CG. Simple and predictable, but it adapts poorly to the user: for some users it is better to show personal recommendations (`UserToItemKNN`), for others popular items (`TopPop`), and for a new user, the history.

Mixigen solves this problem: instead of fixed quotas, it prescores slots, i.e. `(CG, position)` pairs, with a trained model and forms the quotas itself. The algorithm is:
1. For each slot `(cg_name, position)`, build a feature row from the user's context features plus the CG index and the position number.
2. Prescore all slots in one batch with `model_wrapper.predict(rows, "RawFormulaVal")`.
3. Take the top-K slots by descending score, where `K = total_candidates`. For each CG, the quota is the maximum selected position + 1.
4. Request the quotas from each CG, deduplicate by `item_id` and set `factors["source_cg"]` to track the source of a candidate in the logs and at runtime.

You can see how to train such a model in the homework [yandexdataschool/recsys_course/week03_candgen](https://github.com/yandexdataschool/recsys_course/blob/main/week03_candgen/homework/homework.ipynb). Training the Mixigen model itself is not part of the assignment: it is a separate pipeline.

## Statement
Implement three methods of `MixigenCandidateMixer` in [`engine/src/engine/cg/mixing.py`](../../../engine/src/engine/cg/mixing.py).
- `build_prescore_row` builds the row for one slot: the context features in the order of `self.context_features` (with the default `default_feature_value`), then `cg_name_to_index[cg_name]` as an int, then `float(position)`.
- `compute_quotas` forms all slots `(cg_name, position)` for `position ∈ [0, num_candidates_per_cg)`, prescores them in a batch, takes the top-`total_candidates` by descending score, computes the maximum position per CG and returns `{cg_name: max_pos + 1}`.
- `merge` gets the quotas, pulls the required number of candidates from each CG, logs them through `log_cg_candidates` if `self.candidate_logger` is set, deduplicates by `item_id` (iterate over `self.cg_names` for a stable order) and sets `factors["source_cg"]`. Mixigen does not use the `schema` argument (the experiment's CG schema): it computes the quotas itself.

## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/cgs/test_mixigen.py)
```

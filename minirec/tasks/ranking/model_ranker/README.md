# ModelRanker

## About the task
`FeatureReranker`, already available in the engine, sorts candidates by a single factor: a simple and fast heuristic, but without interactions between factors. `ModelRanker` loads a trained model (CatBoost or linear) and scores each candidate on the whole factor vector; some factors are candidate factors (`candidate.factors`), some are context factors (`context.user_profile`). Which is which is set by `factor_schema`, a list of `(factor_name, is_context_factor)` pairs that is saved next to the model at training time.

`ModelRanker` inherits from `FeatureReranker`: you write the model score into the corresponding item factor, and the sorting is called from the parent class. To support both models, there is the `ModelWrapper` protocol in [`engine/src/engine/score_model.py`](../../../engine/src/engine/score_model.py) with ready `CatBoostWrapper` and `LinearModelWrapper`; use it!

## Statement
Implement `ModelRanker` in [`engine/src/engine/ranking/model_ranker.py`](../../../engine/src/engine/ranking/model_ranker.py).
- `start_component(config, deps)` receives the already validated `ModelRankerConfig` (all fields are described in it; an unknown `model_type` is rejected by the config validation itself) and the shared `ComponentDeps` dependencies. Read `factor_schema` through `deps.engine_data.read_json(config.factor_schema_path)` (cast the types to `(str, bool)`), and build the wrapper for `config.model_type`, `"CatBoost"` or `"Linear"`, with the ready function `build_model_wrapper` from `score_model.py`.
- `get_factor_matrix` builds a factor row for each candidate in the order of `self.factor_schema`: context factors come from `context.user_profile`, candidate factors from `candidate.factors`, and missing ones are replaced with `default_feature_value`.
- `rerank` builds the matrix, runs it through `model_wrapper.predict(matrix, prediction_type)`, puts the resulting scores into `candidate.factors` as plain Python `float`s (otherwise NumPy scalars end up in the candidate log) and delegates sorting to the parent ranker class.

## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/rankers/test_model_ranker.py)
```

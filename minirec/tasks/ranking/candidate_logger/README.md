# CandidateLogger

## About the task
To train a ranking model (or a mixer), we need logs of candidate lists at two different places in the engine. The first is after generation and after feature enrichment: the full list of candidates with their factors before cutting the top `num_candidates`, so that we can build a training set by matching these lists with user actions. The second is before enrichment with item-level features, right at the output of the CG mixer: the raw `item_id` lists from each candidate generator, including duplicates between them. These logs let us evaluate the quality of each CG independently of the ranker and train an adaptive mixer ([mixigen](../mixigen/README.md)).

Both logs are written by a single `CandidateLogger` class through two different handlers to different files. The engine creates one logger for the whole process in `RecommenderEngine.startup` (the paths come from the constants `CANDIDATES_LOG_PATH` and `CG_CANDIDATES_LOG_PATH`, which are read from environment variables) and hands it to the components through `ComponentDeps.candidate_logger`; `shutdown` closes its files with the `close` method.

## Statement
Implement `CandidateLogger` in [`engine/src/engine/candidate_logger.py`](../../../engine/src/engine/candidate_logger.py).
- `__init__(candidates_path, cg_candidates_path)` receives the paths to the two files. Prepare two `RotatingFileHandler`s with `JSONFormatter` (create missing directories) and store the loggers in `self.logger` and `self.cg_logger` (named `engine.candidates` and `engine.cg_candidates`). The handlers belong to the instance: create the loggers with `logging.Logger(name)`, not with the global `logging.getLogger(name)`, so that two loggers with different paths neither share nor duplicate records.
- `JSONFormatter.format` serializes `record.msg` with `orjson.dumps`; factors can contain NumPy scalars (`np.float64`), so enable `orjson.OPT_SERIALIZE_NUMPY`.
- `filter_factors` keeps only values of primitive types: `str`, `int`, `float` (we assume we do not need embeddings in the logs, only the features of the current ranking model, which does not work with embeddings).
- `log_candidates(request_id, context, candidates, num_served)` writes the request record with the full list of candidates (in ranking order) and their `factors`. `num_served` is how many candidates from the top the client actually received: the engine passes the requested number, and the log gets the same number, but no more than `len(candidates)`. From this field and the order, the offline pipeline restores each candidate's position and whether the user saw it.
- `log_cg_candidates` writes a record with the `item_id` lists from each CG and the filtered features from `context.user_profile`.

See the tests for the exact record formats and the constants for the rotation sizes.

## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/test_logging.py)
```

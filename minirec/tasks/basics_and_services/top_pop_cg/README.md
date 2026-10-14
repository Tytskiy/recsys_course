# TopPop Candidate Generator

## About the task
TopPop is a simple candidate generator that returns popular items, and a popular baseline. By the way, it is not that easy to beat, especially in its DecayPop variant! But we are interested in the sampling version: instead of taking the top popular items, we sample items with probability proportional to the number of interactions with each item.

## Statement
Implement the `TopPop` candidate generator in [`engine/cg/top_pop.py`](../../../engine/src/engine/cg/top_pop.py).
- The `start_component(config, deps)` method is already written: the engine passes it the validated `TopPopConfig` config (fields `path` and `seed`, 42 by default) and the shared `ComponentDeps` dependencies. The method reads the JSON with item frequencies through `deps.engine_data` (the path is relative to the snapshot directory, which `EngineData` unpacks into its own subdirectory of `REC_ENGINE_FILES_DIR`) and passes `counts` and `seed` to the constructor.
- In `__init__`, build a distribution proportional to the frequencies (the `probabilities` attribute, in the key order of `counts`) and fix the seed (`numpy.random.default_rng` will help).
- In `extract_candidates`, sample candidates without replacement from the distribution derived from the frequencies. If more candidates are requested than there are items, return all items (each once).
- The tests check sampling statistically (frequencies over many calls), not a specific output: the same seed gives the same outputs, a different seed gives different ones.

Implement the statistics computation for this candidate generator in [`engine_builder/tasks/models/top_pop.py`](../../../engine_builder/tasks/models/top_pop.py): you only need to count how often a given action occurs for each item.

## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/cgs/test_top_pop.py)
(cd engine_builder && uv run pytest tests/models/test_top_pop.py)
```

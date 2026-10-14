# Engine Server

## About the task
Most often, recommendations are delivered to users through some microservice that the backend calls to get a user's feed; all rendering and enrichment with extra information happen separately. One way to build it is to separate the web service itself (REST API, runtime and file loading) from the recommendation logic. In this task you need to develop such a server on FastAPI, given a ready `RecommenderEngine` object that, surprisingly, can do `/recommend`.

## Statement

Implement a REST service on FastAPI with a single Recommend method, whose parameters are described in `engine_server/engine_server/models.py`. You need to support creating the RecommenderEngine object by loading a config, and pagination.

More on pagination:
- We do not want to hand the backend all 500 recommendation candidates at once, even if we computed them: backends most often use only one or two pages and do not render the whole feed at once.
- For this we use Bloom-filter-based pagination: with each request, the caller of the endpoint sends a filter encoded as a string that contains the identifiers of items already seen. This logic is implemented in the BloomToken class.
- The `RecommenderEngine.recommend` method guarantees that items contained in context.item_id_filter will not be returned as candidates.
- At the same time, we compute candidates for every pagination page instead of caching them after the first request, because by the time the second page is requested the context, the features or the recommender itself may have changed.
- Accordingly, the endpoint only needs to get the list of items, update the Bloom filter and send the new token in the response together with the recommendations.

## Tips
- Inside the service, the whole `engine` library and all its methods are available to you (the reference solution needs nothing besides `RecommenderEngine`, `BloomToken`, `RecContext`).
- Build a correct `RecContext` from the request parameters and pass it to the recommender: the information in it is needed (at least the user ID).
- `source_page` in the request is already validated as a `SourceType` from `engine.interfaces` (for example, `ST_Feed`): an unknown page returns 422 before the recommender is called.
- In the test configs `engine_data_path` is not set: `EngineRuntime.__init__` already fills it in from the `REC_ENGINE_LOCAL_ARCHIVE_PATH` environment variable (the `LOCAL_ARCHIVE_PATH_ENV` constant); you do not need to do this.
- `RecommenderEngine` is created and reloaded inside `EngineRuntime` (its API is described in the [engine_server_reload](../engine_server_reload/README.md) task); in `lifespan` work only with the runtime.
- In the tests `EngineRuntime` is replaced by a fake (`engine_server/tests/server_harness.py`), so you can pass this task without solving engine_server_reload. Create the runtime through the name `EngineRuntime` (`EngineRuntime(config)`), call `await runtime.start()` on startup and `await runtime.stop()` on shutdown, and store it in `app.state.runtime`; pass requests to `runtime.recommend(num_candidates=..., context=...)`.
- To run the service live, the runtime needs at least `reload_engine`, `start` and `stop`; to begin with, a version without reloading is enough (load the engine once and shut it down on stop).

## How to run the tests
```bash
cd engine_server
uv run pytest tests/test_server.py
```

In disputed cases, follow the tests in `tests/`.

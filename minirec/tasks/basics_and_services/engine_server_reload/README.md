# Engine Server Reload

## About the task
This task is an extension of Engine Server. Suppose we already have a service that creates the recommender object once, at initialization. But what if we want to periodically recreate it (or simply load new files into it: models, indexes, features) when updates arrive? With many service instances running in parallel you can simply recreate containers one / a few at a time, but in our case there is a single container, and we do not want to lose availability. In this situation, FastAPI lets you implement polling and data loading with `asyncio`: the runtime will have a background task that periodically reads a marker, compares it with the previous value and updates the recommender when it changes.


## Runtime methods
Implement the `EngineRuntime` methods (`engine_server/engine_server/runtime.py`) whose bodies are left as `# TODO` stubs. The rest are already implemented.

### Marker and archive
The downloader service (`engine_downloader`) downloads the snapshot to `archive.tar.gz` and then writes the marker `archive.version.json` next to it. New marker contents mean a new snapshot.

### `__init__`
Implemented for you; it prepares the runtime and the synchronization: `engine_lock`, the `engine_idle` condition over it, the counter of in-flight requests `in_flight` and the set of background tasks `drain_tasks`, as well as `loaded_marker`, the marker read before loading the current engine (`None` for now).

### `read_marker`
Implemented for you: reads the marker file and returns its contents with surrounding whitespace stripped. If the file does not exist or is empty, it returns `None`. Versions are compared by contents: a rewritten file with the same contents is the same version.

### `reload_engine`
Waits for the local archive to appear at `engine_data_path`. The check must run in a loop: check the file → `sleep(DEFAULT_POLL_INTERVAL_SEC)` → repeat.
If the file still has not appeared by the `DEFAULT_TIMEOUT_INTERVAL_SEC` deadline, raise `FileNotFoundError` with the text `"Timed out waiting for archive: <path>"`.
As soon as the file is found, call `RecommenderEngine.startup(config)` and return the new instance.

### `try_reload_engine`
Implemented for you: an async wrapper over `reload_engine` via `asyncio.to_thread`. If loading fails for any reason, it logs the error and returns `None`, so that polling continues and the old engine keeps serving.

### `wait_for_stop`
An async pause between marker checks in the polling loop, which also supports early stopping through an `asyncio` event.
- If `update_stop_event` has not been created, it returns `True` immediately (meaning polling is not needed).
- If `update_stop_event` exists, it waits for it with the `DEFAULT_POLL_INTERVAL_SEC` timeout:
    - If the wait succeeds, it returns `True`: the runtime has stopped.
    - When the timeout expires, it returns `False` (time for the next marker check).

### `recommend(num_candidates, context)`
Implemented for you: takes the current `engine` under `engine_lock`, increments the `in_flight[engine]` counter for it and calls `recommend` without the lock, so that requests are processed in parallel. On completion (including with an exception) it decrements the counter, removes the key at zero and wakes those waiting on `engine_idle` (`notify_all`). If `engine` has not been created yet, it raises `RuntimeError("runtime is not started")`.

### `drain_and_shutdown(engine, timeout)`
A blocking method (called in a thread). Waits on `engine_idle` until `engine` has no in-flight requests left (`in_flight[engine] == 0`), but no longer than `timeout` seconds. Then calls `engine.shutdown()` exactly once.
- Returns `True` if the engine became idle, and `False` if the timeout fired and `shutdown` was forced (log a warning).
- The timeout is needed so that one hung request does not hold the old engine (and with it the reload and the service shutdown) forever. The price is that this request will fail on closed storages.
- Do not remove keys from `in_flight`: `recommend` does that when the request finally completes.

### `retire(engine)`
Sends a retired engine to background shutdown: creates the task `asyncio.create_task(asyncio.to_thread(self.drain_and_shutdown, engine, DRAIN_TIMEOUT_SEC))`, puts it into `self.drain_tasks` and removes it from there on completion (`add_done_callback`). The method itself waits for nothing. Call it from the runtime coroutines after `engine` has been removed from `self.engine`: then new requests will no longer reach it.

### `poll_for_updates`
The background update loop. The last loaded marker is stored in `self.loaded_marker` (it is set by `start`, see below).

The loop repeats until `self.wait_for_stop()` returns `True` (exactly one `wait_for_stop` call per iteration; the tests replace it to control the iterations):
- reads the marker; if there is none (`None`) or it equals the last loaded one, it does nothing;
- otherwise it loads a new engine through `try_reload_engine`; on failure (`None`) it does nothing, and the same marker will be retried on the next iteration;
- on success it atomically swaps `engine` under `engine_lock`, sends the old engine to `retire` (without waiting for its shutdown: the next reload must not wait for a hung request) and writes the marker to `self.loaded_marker`.

### `start`
Reads the marker through `read_marker`, then loads the first engine through `reload_engine` (in a thread; an error at this step must propagate) and saves the marker read before loading to `self.loaded_marker`. The marker is read precisely before loading: if the downloader manages to put a new snapshot while the first engine is starting, polling will see the new marker and reload, instead of taking it for the one already loaded. Then it creates `update_stop_event` and starts `poll_for_updates` as a background task through `asyncio.create_task` (`update_task`).

### `stop`
Sets the stop signal, waits for the polling task to finish (its exception is logged, not propagated), removes the current `engine` from `self.engine` and, if there was one, sends it to `retire`. Then it waits for all tasks in `drain_tasks`, so after `stop` all engines are shut down; thanks to the timeout this takes no longer than `DRAIN_TIMEOUT_SEC` after the last request. If a reload is in progress at that moment and the archive is not there yet, `stop` will wait for the `reload_engine` timeout (up to `DEFAULT_TIMEOUT_INTERVAL_SEC`): a thread from `asyncio.to_thread` cannot be interrupted.

## Implementation details
- The archive path is taken from `REC_ENGINE_LOCAL_ARCHIVE_PATH`, otherwise `DEFAULT_LOCAL_ARCHIVE_PATH` is used; the marker path from `REC_ENGINE_UPDATE_MARKER_PATH`.
- The tests replace the constants `DEFAULT_POLL_INTERVAL_SEC`, `DEFAULT_TIMEOUT_INTERVAL_SEC`, `DRAIN_TIMEOUT_SEC` through `monkeypatch`, so read them as module globals at call time, not as default argument values.
- Use `asyncio.to_thread` for blocking operations (waiting for the archive, waiting for requests).
- The tests do not use `sleep`: requests in the fake engine block on a `threading.Event`, and polling iterations are driven by replacing `wait_for_stop`. Test helper code: `engine_server/tests/runtime_harness.py`.

## Testing
```bash
cd engine_server
uv run pytest tests/test_runtime.py
```

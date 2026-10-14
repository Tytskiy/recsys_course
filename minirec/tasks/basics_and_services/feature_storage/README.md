# Feature Storage

## About the task
The feature storage holds any features, embeddings, history and in general any information needed at runtime. It should be fast, memory-efficient in how it lays out objects and preferably shardable, but a simple key-value database installation is enough for us. You can use Valkey, which is the truly open-source Redis.

## Statement
Implement a wrapper over the Valkey database (in this context the same as Redis) in `engine/storages/feature_storage.py` that implements the `FeatureStorage` protocol from the interfaces, specifically its methods `start`, `extract`, `bulk_extract`, `update`, `delete`, `close`. You can find the exact requirements for them in the tests: each variant of the `(user_id, item_id)` pair is a separate key, `update` adds features to the key (without erasing the others), `delete` returns the number of features actually deleted.

## Tips
- Use `build_hash_key` to build a key from a user/item ID.
- For simplicity, serialize feature values to JSON with orjson. Think about why this is bad.
- Use pipelining and hash groups; you can take inspiration from [the DoorDash article](https://careersatdoordash.com/blog/building-a-gigascale-ml-feature-store-with-redis/).

## Testing
```bash
cd engine
uv run pytest tests/unit/storages/test_feature_storage.py
```

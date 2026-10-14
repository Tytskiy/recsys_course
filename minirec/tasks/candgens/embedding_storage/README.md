# Embedding Storage

## About the task
The embedding storage is responsible for storing and working with embeddings: getting values, computing distances between given objects and, probably most importantly, vector search, approximate or exact. Two main options:
- Either an in-memory index, for example a FAISS ANN index saved to a file.
- Or a vector database, for example Qdrant / Milvus.

GPU indexes are appearing now too, but that is out of scope for this task :)

## Statement
Implement the `EmbeddingStorage` protocol with the Qdrant vector DB client in [`engine/storages/embedding_storage.py`](../../../engine/src/engine/storages/embedding_storage.py). You need the protocol methods, but you are free to create any additional ones you need; for example, `__init__` will come in handy. You can find the exact implementation requirements (the expected behaviour of the methods) in the tests.

## Tips
- The `search` and `get` methods simply wrap the corresponding client methods and convert the arguments / results.
- The `score` method is implemented through an exhaustive search filtered by a list of IDs.
- The ANN index build parameters are not part of the interface: you can experiment (it should not affect the tests).

## Testing
```bash
(cd engine && uv run pytest tests/unit/storages/test_embedding_storage.py)
```

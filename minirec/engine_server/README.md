# engine server
A recommendation microservice that wraps the [`engine`](../engine/README.md) framework.

- It is a FastAPI application with a simple runtime.
- It implements one method, `POST /recommend`, which returns the user's feed with pagination support; recommendations are recomputed on every request for the next page.
- It implements periodic asynchronous loading of new service files (models, data for the CGs) from the file system and replacement of the internal recommender.


## How to run the service locally
To run the service locally in a container, you first need to build its snapshot in [`engine_builder`](../engine_builder/README.md).
Then simply start the containers:
- A sidecar service that downloads the archive from object storage
- The recommender API
- (Optional) a web interface for interactive browsing of the feed.

```bash
docker compose up -d engine-downloader engine-server viewer
```

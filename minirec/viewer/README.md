# Viewer

Small React+Vite SPA for exercising the `/recommend` endpoint of the `engine_server` microservice. Lets you build a request, fetch recommendations, page through results, and inspect raw responses against the local item catalog.

## Development

```sh
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and proxies `/recommend` to the backend.

### Pointing at a backend

- **Dev (proxy):** set `VITE_API_PROXY` to the engine_server URL (default `http://localhost:8000`).
- **Built artifact:** set `VITE_API_BASE` at build time to the absolute backend URL. Leave empty when the viewer is served behind the same origin or reverse-proxy as the backend.

```sh
VITE_API_PROXY=http://localhost:8000 npm run dev
VITE_API_BASE=https://api.example.com npm run build
```

### Catalog

Item metadata is served as a static `public/catalog.json` file (array of `{ item_id, title, description, image_url }`). It is loaded once at startup and used to enrich `/recommend` candidate IDs into cards. Items not present in the catalog still render with an `unknown` badge.

Regenerate `public/catalog.json` from the dataset whenever the underlying items change.

## Layout

```
src/
  api.ts                 # fetch wrapper + ApiError with FastAPI 422 parsing
  types.ts               # shared types (SourceType, DraftRequest, …)
  constants.ts           # SOURCE_TYPES enum, localStorage key
  hooks/
    useCatalog.ts        # loads /catalog.json once
    useRecommend.ts      # request lifecycle, pagination, AbortController
  components/
    RequestForm.tsx      # validated form, persists to localStorage
    ResultsList.tsx      # summary chips, cards grid, "Load more" button
    ItemCard.tsx
    RawResponse.tsx
    ErrorBanner.tsx
  App.tsx                # composition shell
  main.tsx
  styles.css
```

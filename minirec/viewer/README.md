# Viewer

Small React+Vite SPA for exercising the `/recommend` endpoint of the `engine_server` microservice. Lets you build a request, fetch recommendations, page through results (BloomToken `page_token`; items repeated across pages get a `repeat` badge), inspect raw responses against the local item catalog, and copy the last request as `curl`. Item and user IDs are uint64, so the viewer keeps them as decimal strings and never rounds them through JavaScript numbers.

Submitted params are kept in the URL query (shareable) and in `localStorage`.

## Development

```sh
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and proxies `/recommend` to the backend. `npm run typecheck` and `npm test` (vitest) check the code.

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
  api.ts                 # lossless JSON, request encoding, curl, error mapping (422/HTTP/network)
  draft.ts               # form draft: URL/localStorage persistence, validation into a request
  types.ts               # SOURCE_TYPES, request/response types
  hooks/
    useCatalog.ts        # loads /catalog.json once
    useRecommend.ts      # request lifecycle, pagination, AbortController
  components/
    RequestForm.tsx
    ResultsList.tsx      # summary, cards grid, "Load more", copy as curl, raw response
    ItemCard.tsx
    ErrorBanner.tsx
  App.tsx                # composition shell
  main.tsx
  styles.css
```

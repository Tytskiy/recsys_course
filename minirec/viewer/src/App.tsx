import { ApiError, apiBase } from "./api";
import { ErrorBanner } from "./components/ErrorBanner";
import { RequestForm } from "./components/RequestForm";
import { ResultsList } from "./components/ResultsList";
import { useCatalog } from "./hooks/useCatalog";
import { useRecommend } from "./hooks/useRecommend";

export default function App() {
  const catalog = useCatalog();
  const recommend = useRecommend();
  const { error } = recommend;

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="kicker">Lavka datasets viewer</p>
          <h1>Rec Engine Explorer</h1>
          <p className="sub">
            Tune request params, call the API, and browse top items with images.
          </p>
        </div>
        <div className="chip">API: {apiBase || "(proxied)"}</div>
      </header>

      <main className="layout">
        <section className="panel form-panel">
          <RequestForm
            onSubmit={recommend.submit}
            submitting={recommend.status === "loading" || recommend.status === "loading-more"}
          />
          {recommend.status === "error" && error ? (
            <ErrorBanner
              title={error.message}
              details={error instanceof ApiError ? error.details : []}
            />
          ) : null}
        </section>

        <section className="panel results-panel">
          <ResultsList
            {...catalog}
            pages={recommend.pages}
            lastRequest={recommend.lastRequest}
            nextPageToken={recommend.nextPageToken}
            rawResponse={recommend.rawResponse}
            latencyMs={recommend.latencyMs}
            loadingMore={recommend.status === "loading-more"}
            onLoadMore={recommend.loadMore}
          />
        </section>
      </main>
    </div>
  );
}

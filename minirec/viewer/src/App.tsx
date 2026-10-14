import { api } from "./api";
import { ErrorBanner } from "./components/ErrorBanner";
import { RawResponse } from "./components/RawResponse";
import { RequestForm } from "./components/RequestForm";
import { ResultsList } from "./components/ResultsList";
import { useCatalog } from "./hooks/useCatalog";
import { useRecommend } from "./hooks/useRecommend";

export default function App() {
  const { catalog, status: catalogStatus } = useCatalog();
  const recommend = useRecommend(catalog);

  const submitting =
    recommend.status === "loading" || recommend.status === "loading-more";

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
        <div className="chip">API: {api.apiBase || "(proxied)"}</div>
      </header>

      <main className="layout">
        <section className="panel form-panel">
          <RequestForm
            onSubmit={recommend.submit}
            submitting={submitting}
            catalogReady={catalogStatus === "success"}
          />
          {recommend.status === "error" ? (
            <ErrorBanner error={recommend.error} />
          ) : null}
        </section>

        <section className="panel results-panel">
          <ResultsList
            items={recommend.items}
            catalogSize={catalog.size}
            catalogStatus={catalogStatus}
            latencyMs={recommend.latencyMs}
            canLoadMore={recommend.canLoadMore}
            loadingMore={recommend.status === "loading-more"}
            onLoadMore={recommend.loadMore}
          />
          <RawResponse data={recommend.rawResponse} />
        </section>
      </main>
    </div>
  );
}

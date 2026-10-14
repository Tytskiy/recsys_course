import type { CatalogStatus } from "../hooks/useCatalog";
import type { ResultItem } from "../types";
import { ItemCard } from "./ItemCard";

type Props = {
  items: ResultItem[];
  catalogSize: number;
  catalogStatus: CatalogStatus;
  latencyMs: number | null;
  canLoadMore: boolean;
  loadingMore: boolean;
  onLoadMore: () => void;
};

function renderCatalogSummary(status: CatalogStatus, size: number) {
  switch (status) {
    case "loading":
      return "Loading…";
    case "error":
      return "Failed";
    case "success":
      return size.toLocaleString();
    case "idle":
    default:
      return "—";
  }
}

export function ResultsList({
  items,
  catalogSize,
  catalogStatus,
  latencyMs,
  canLoadMore,
  loadingMore,
  onLoadMore,
}: Props) {
  return (
    <>
      <div className="summary">
        <div>
          <span className="label">Items</span>
          <strong>{items.length}</strong>
        </div>
        <div>
          <span className="label">Catalog</span>
          <strong>{renderCatalogSummary(catalogStatus, catalogSize)}</strong>
        </div>
        <div>
          <span className="label">Latency</span>
          <strong>{latencyMs !== null ? `${Math.round(latencyMs)} ms` : "—"}</strong>
        </div>
      </div>

      <div className="cards">
        {items.map((result, index) => (
          <ItemCard
            key={`${result.item.item_id}-${index}`}
            result={result}
            rank={index + 1}
          />
        ))}
      </div>

      {canLoadMore ? (
        <button
          type="button"
          className="load-more"
          onClick={onLoadMore}
          disabled={loadingMore}
        >
          {loadingMore ? "Loading…" : "Load more"}
        </button>
      ) : null}
    </>
  );
}

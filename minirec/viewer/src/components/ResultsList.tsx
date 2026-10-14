import { useEffect, useMemo, useState } from "react";
import { toCurl } from "../api";
import type { CatalogState } from "../hooks/useCatalog";
import type { Id, RecommendRequest } from "../types";
import { ItemCard } from "./ItemCard";

type Props = CatalogState & {
  pages: Id[][];
  lastRequest: RecommendRequest | null;
  nextPageToken: string | null;
  rawResponse: string | null;
  latencyMs: number | null;
  loadingMore: boolean;
  onLoadMore: () => void;
};

function shortenToken(raw: string, token: string | null): string {
  if (!token || token.length <= 40) return raw;
  return raw.replace(token, `${token.slice(0, 24)}… (${token.length} chars)`);
}

export function ResultsList({
  catalog,
  status: catalogStatus,
  pages,
  lastRequest,
  nextPageToken,
  rawResponse,
  latencyMs,
  loadingMore,
  onLoadMore,
}: Props) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const rows = useMemo(() => {
    const seen = new Set<Id>();
    return pages.flatMap((ids, pageIndex) =>
      ids.map((id) => {
        const repeat = seen.has(id);
        seen.add(id);
        return { id, page: pageIndex + 1, repeat };
      })
    );
  }, [pages]);

  const repeats = rows.filter((row) => row.repeat).length;
  const canLoadMore = Boolean(nextPageToken) && pages[pages.length - 1]?.length > 0;

  const copyCurl = async () => {
    if (!lastRequest) return;
    const curl = toCurl(lastRequest);
    try {
      await navigator.clipboard.writeText(curl);
      setCopied(true);
    } catch {
      window.prompt("Copy the request:", curl);
    }
  };

  return (
    <>
      <div className="summary">
        <div>
          <span className="label">Items</span>
          <strong>{rows.length}</strong>
        </div>
        <div>
          <span className="label">Pages</span>
          <strong>{pages.length}</strong>
        </div>
        <div>
          <span className="label">Repeats</span>
          <strong>{repeats}</strong>
        </div>
        <div>
          <span className="label">Catalog</span>
          <strong>
            {catalogStatus === "success"
              ? catalog.size.toLocaleString()
              : catalogStatus === "loading"
                ? "Loading…"
                : "Failed"}
          </strong>
        </div>
        <div>
          <span className="label">Latency</span>
          <strong>{latencyMs !== null ? `${Math.round(latencyMs)} ms` : "—"}</strong>
        </div>
        {lastRequest ? (
          <button type="button" className="ghost" onClick={copyCurl}>
            {copied ? "Copied" : "Copy as curl"}
          </button>
        ) : null}
      </div>

      <div className="cards">
        {rows.map((row, index) => {
          const item = catalog.get(row.id);
          return (
            <ItemCard
              key={index}
              id={row.id}
              item={item}
              rank={index + 1}
              page={row.page}
              unknown={catalogStatus === "success" && !item}
              repeat={row.repeat}
            />
          );
        })}
      </div>

      {canLoadMore ? (
        <button type="button" className="load-more" onClick={onLoadMore} disabled={loadingMore}>
          {loadingMore ? "Loading…" : "Load more"}
        </button>
      ) : null}

      {rawResponse ? (
        <details className="raw" open>
          <summary className="label">Raw response (last page)</summary>
          <pre>{shortenToken(rawResponse, nextPageToken)}</pre>
        </details>
      ) : null}
    </>
  );
}

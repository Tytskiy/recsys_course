import { useEffect, useState } from "react";
import type { ItemMetadata } from "../types";

export type CatalogStatus = "idle" | "loading" | "error" | "success";

export type CatalogState = {
  catalog: Map<number, ItemMetadata>;
  status: CatalogStatus;
};

export function useCatalog(): CatalogState {
  const [catalog, setCatalog] = useState<Map<number, ItemMetadata>>(new Map());
  const [status, setStatus] = useState<CatalogStatus>("idle");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setStatus("loading");
      try {
        const response = await fetch("/catalog.json");
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = (await response.json()) as ItemMetadata[];
        if (cancelled) return;
        const next = new Map<number, ItemMetadata>();
        for (const item of data) {
          next.set(item.item_id, item);
        }
        setCatalog(next);
        setStatus("success");
      } catch {
        if (!cancelled) setStatus("error");
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return { catalog, status };
}

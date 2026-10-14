import { useEffect, useState } from "react";
import { parseJson } from "../api";
import type { Id, ItemMetadata } from "../types";

type CatalogStatus = "loading" | "error" | "success";

export type CatalogState = {
  catalog: Map<Id, ItemMetadata>;
  status: CatalogStatus;
};

export function useCatalog(): CatalogState {
  const [state, setState] = useState<CatalogState>({ catalog: new Map(), status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    fetch("/catalog.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then((text) => {
        if (controller.signal.aborted) return;
        const items = parseJson(text) as ItemMetadata[];
        setState({
          catalog: new Map(items.map((item) => [item.item_id, item])),
          status: "success",
        });
      })
      .catch(() => {
        if (!controller.signal.aborted) setState((prev) => ({ ...prev, status: "error" }));
      });
    return () => controller.abort();
  }, []);

  return state;
}

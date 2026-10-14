import { useCallback, useEffect, useRef, useState } from "react";
import { recommend } from "../api";
import type { Id, RecommendRequest } from "../types";

type RecommendStatus = "idle" | "loading" | "loading-more" | "error" | "success";

type RecommendState = {
  status: RecommendStatus;
  error: Error | null;
  pages: Id[][];
  firstRequest: RecommendRequest | null;
  lastRequest: RecommendRequest | null;
  nextPageToken: string | null;
  rawResponse: string | null;
  latencyMs: number | null;
};

const INITIAL: RecommendState = {
  status: "idle",
  error: null,
  pages: [],
  firstRequest: null,
  lastRequest: null,
  nextPageToken: null,
  rawResponse: null,
  latencyMs: null,
};

export function useRecommend() {
  const [state, setState] = useState<RecommendState>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = useCallback(async (request: RecommendRequest, append: boolean) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setState((prev) =>
      append
        ? { ...prev, status: "loading-more", error: null, lastRequest: request }
        : { ...INITIAL, status: "loading", firstRequest: request, lastRequest: request }
    );

    const startedAt = performance.now();
    try {
      const { response, text } = await recommend(request, controller.signal);
      if (controller.signal.aborted) return;
      const latencyMs = performance.now() - startedAt;
      setState((prev) => ({
        ...prev,
        status: "success",
        pages: [...prev.pages, response.candidates],
        nextPageToken: response.next_page_token,
        rawResponse: text,
        latencyMs,
      }));
    } catch (err) {
      if (controller.signal.aborted) return;
      setState((prev) => ({
        ...prev,
        status: "error",
        error: err instanceof Error ? err : new Error(String(err)),
      }));
    }
  }, []);

  const submit = useCallback((request: RecommendRequest) => run(request, false), [run]);

  const { firstRequest, nextPageToken } = state;
  const loadMore = useCallback(() => {
    if (firstRequest && nextPageToken) run({ ...firstRequest, page_token: nextPageToken }, true);
  }, [firstRequest, nextPageToken, run]);

  return { ...state, submit, loadMore };
}

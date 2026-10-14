import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import type {
  DraftRequest,
  ItemMetadata,
  RecommendRequest,
  RecommendResponse,
  ResultItem,
} from "../types";

export type RecommendStatus = "idle" | "loading" | "loading-more" | "error" | "success";

export type RecommendState = {
  status: RecommendStatus;
  error: ApiError | Error | null;
  items: ResultItem[];
  rawResponse: object | null;
  nextPageToken: string;
  latencyMs: number | null;
  canLoadMore: boolean;
};

export type RecommendActions = {
  submit: (draft: DraftRequest) => Promise<void>;
  loadMore: () => Promise<void>;
};

type ValidatedPayload = {
  payload: RecommendRequest;
  numCandidates: number;
};

function validate(draft: DraftRequest): ValidatedPayload {
  const userId =
    typeof draft.user_id === "number" && Number.isInteger(draft.user_id)
      ? draft.user_id
      : null;
  if (userId === null || userId < 1) {
    throw new Error("User ID must be a positive integer.");
  }
  const numCandidates =
    typeof draft.num_candidates === "number" && Number.isInteger(draft.num_candidates)
      ? draft.num_candidates
      : null;
  if (numCandidates === null || numCandidates < 1) {
    throw new Error("Num candidates must be a positive integer.");
  }
  const payload: RecommendRequest = {
    user_id: userId,
    num_candidates: numCandidates,
    source_page: draft.source_page,
    current_time: draft.current_time || null,
    context_item_ids: draft.context_item_ids,
  };
  return { payload, numCandidates };
}

function mapCandidates(
  candidates: number[],
  catalog: Map<number, ItemMetadata>
): ResultItem[] {
  return candidates.map((itemId) => {
    const found = catalog.get(itemId);
    if (found) {
      return { item: found, inCatalog: true };
    }
    return {
      item: { item_id: itemId, title: null, description: null, image_url: null },
      inCatalog: false,
    };
  });
}

function stripToken(response: RecommendResponse): object {
  const { next_page_token: _t, ...rest } = response;
  return rest;
}

export function useRecommend(catalog: Map<number, ItemMetadata>): RecommendState & RecommendActions {
  const [status, setStatus] = useState<RecommendStatus>("idle");
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [items, setItems] = useState<ResultItem[]>([]);
  const [rawResponse, setRawResponse] = useState<object | null>(null);
  const [nextPageToken, setNextPageToken] = useState<string>("");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [canLoadMore, setCanLoadMore] = useState<boolean>(false);

  const abortRef = useRef<AbortController | null>(null);
  const lastPayloadRef = useRef<RecommendRequest | null>(null);
  const lastNumCandidatesRef = useRef<number>(0);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const runRequest = useCallback(
    async (
      payload: RecommendRequest,
      mode: "submit" | "more",
      numCandidates: number
    ) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus(mode === "submit" ? "loading" : "loading-more");
      setError(null);
      if (mode === "submit") {
        setItems([]);
        setRawResponse(null);
        setLatencyMs(null);
        setCanLoadMore(false);
      }

      const startedAt = performance.now();
      try {
        const response = await api.recommend(payload, controller.signal);
        if (controller.signal.aborted) return;

        const elapsed = performance.now() - startedAt;
        const mapped = mapCandidates(response.candidates, catalog);

        setLatencyMs(elapsed);
        setRawResponse(stripToken(response));
        setNextPageToken(response.next_page_token ?? "");
        setCanLoadMore(
          Boolean(response.next_page_token) &&
            response.candidates.length === numCandidates
        );
        setItems((prev) => (mode === "submit" ? mapped : [...prev, ...mapped]));
        setStatus("success");
      } catch (err) {
        if ((err as Error)?.name === "AbortError") return;
        setStatus("error");
        setError(err instanceof Error ? err : new Error("Unexpected error"));
      }
    },
    [catalog]
  );

  const submit = useCallback(
    async (draft: DraftRequest) => {
      let validated: ValidatedPayload;
      try {
        validated = validate(draft);
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err : new Error("Invalid input"));
        setItems([]);
        setRawResponse(null);
        setLatencyMs(null);
        setCanLoadMore(false);
        return;
      }
      lastPayloadRef.current = validated.payload;
      lastNumCandidatesRef.current = validated.numCandidates;
      await runRequest(validated.payload, "submit", validated.numCandidates);
    },
    [runRequest]
  );

  const loadMore = useCallback(async () => {
    const base = lastPayloadRef.current;
    if (!base || !nextPageToken) return;
    const payload: RecommendRequest = { ...base, page_token: nextPageToken };
    await runRequest(payload, "more", lastNumCandidatesRef.current);
  }, [nextPageToken, runRequest]);

  return {
    status,
    error,
    items,
    rawResponse,
    nextPageToken,
    latencyMs,
    canLoadMore,
    submit,
    loadMore,
  };
}

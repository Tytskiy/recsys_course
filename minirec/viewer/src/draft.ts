import { SOURCE_TYPES, type FieldError, type RecommendRequest, type SourceType } from "./types";

export type Draft = {
  user_id: string;
  num_candidates: string;
  source_page: SourceType;
  current_time: string;
  context_item_ids: string;
};

export const DEFAULT_DRAFT: Draft = {
  user_id: "",
  num_candidates: "10",
  source_page: "ST_Upsale",
  current_time: "",
  context_item_ids: "",
};

const STORAGE_KEY = "rec-engine-viewer.draft.v2";
const INTEGER = /^-?\d+$/;

function isSourceType(value: string): value is SourceType {
  return (SOURCE_TYPES as readonly string[]).includes(value);
}

export function readDraft(): Draft {
  let query = window.location.search;
  if (!query) {
    try {
      query = window.localStorage.getItem(STORAGE_KEY) ?? "";
    } catch {
      query = "";
    }
  }
  const params = new URLSearchParams(query);
  const draft = { ...DEFAULT_DRAFT };
  for (const key of ["user_id", "num_candidates", "current_time", "context_item_ids"] as const) {
    draft[key] = params.get(key) ?? draft[key];
  }
  const sourcePage = params.get("source_page");
  if (sourcePage && isSourceType(sourcePage)) draft.source_page = sourcePage;
  return draft;
}

export function saveDraft(draft: Draft): void {
  const params = new URLSearchParams(Object.entries(draft).filter(([, value]) => value !== ""));
  const query = params.toString();
  window.history.replaceState(null, "", query ? `?${query}` : window.location.pathname);
  try {
    window.localStorage.setItem(STORAGE_KEY, query);
  } catch {
    // storage can throw in private mode; the URL still holds the draft
  }
}

export function splitIds(raw: string): string[] {
  return raw.split(/[\s,]+/).filter(Boolean);
}

export function toRequest(draft: Draft): RecommendRequest | FieldError[] {
  const errors: FieldError[] = [];
  const userId = draft.user_id.trim();
  if (!INTEGER.test(userId)) errors.push({ field: "user_id", msg: "must be an integer" });

  const numCandidates = Number(draft.num_candidates);
  if (!Number.isSafeInteger(numCandidates) || numCandidates <= 0) {
    errors.push({ field: "num_candidates", msg: "must be a positive integer" });
  }

  const tokens = splitIds(draft.context_item_ids);
  const invalid = tokens.filter((token) => !INTEGER.test(token));
  if (invalid.length) {
    errors.push({ field: "context_item_ids", msg: `not integers: ${invalid.join(", ")}` });
  }

  const time = draft.current_time ? new Date(draft.current_time) : null;
  if (time && Number.isNaN(time.getTime())) {
    errors.push({ field: "current_time", msg: "invalid date" });
  }

  if (errors.length) return errors;
  return {
    user_id: BigInt(userId).toString(),
    num_candidates: numCandidates,
    source_page: draft.source_page,
    current_time: time ? time.toISOString() : null,
    context_item_ids: tokens.length ? tokens.map((token) => BigInt(token).toString()) : null,
  };
}

import { useEffect, useMemo, useState, type FormEvent } from "react";
import { SOURCE_TYPES, STORAGE_KEY } from "../constants";
import type { DraftRequest, SourceType } from "../types";

type Props = {
  onSubmit: (draft: DraftRequest) => void;
  submitting: boolean;
  catalogReady: boolean;
};

const DEFAULT_DRAFT: DraftRequest = {
  user_id: "",
  num_candidates: "",
  source_page: "ST_Upsale",
  current_time: "",
  context_item_ids: null,
};

type PersistedDraft = Omit<DraftRequest, "current_time">;

function loadPersisted(): Partial<DraftRequest> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<PersistedDraft>;
    const draft: Partial<DraftRequest> = {};
    if (typeof parsed.user_id === "number" || parsed.user_id === "") {
      draft.user_id = parsed.user_id;
    }
    if (typeof parsed.num_candidates === "number" || parsed.num_candidates === "") {
      draft.num_candidates = parsed.num_candidates;
    }
    if (
      typeof parsed.source_page === "string" &&
      (SOURCE_TYPES as readonly string[]).includes(parsed.source_page)
    ) {
      draft.source_page = parsed.source_page as SourceType;
    }
    if (Array.isArray(parsed.context_item_ids) || parsed.context_item_ids === null) {
      draft.context_item_ids = parsed.context_item_ids ?? null;
    }
    return draft;
  } catch {
    return {};
  }
}

function persist(draft: DraftRequest) {
  if (typeof window === "undefined") return;
  try {
    const persisted: PersistedDraft = {
      user_id: draft.user_id,
      num_candidates: draft.num_candidates,
      source_page: draft.source_page,
      context_item_ids: draft.context_item_ids,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
  } catch {
    // ignore quota / serialization errors
  }
}

function toDatetimeLocal(iso: string): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(
    date.getDate()
  )}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function RequestForm({ onSubmit, submitting, catalogReady }: Props) {
  const [draft, setDraft] = useState<DraftRequest>(() => ({
    ...DEFAULT_DRAFT,
    ...loadPersisted(),
  }));
  const [contextInput, setContextInput] = useState<string>(() =>
    (draft.context_item_ids ?? []).join(", ")
  );

  useEffect(() => {
    persist(draft);
  }, [draft]);

  const datetimeLocal = useMemo(
    () => toDatetimeLocal(draft.current_time),
    [draft.current_time]
  );

  const update = <K extends keyof DraftRequest>(key: K, value: DraftRequest[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }));
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit(draft);
  };

  const reset = () => {
    setDraft(DEFAULT_DRAFT);
    setContextInput("");
  };

  return (
    <form onSubmit={submit} className="form">
      <label>
        User ID
        <input
          type="number"
          value={draft.user_id}
          onChange={(event) => {
            const value = event.target.value;
            if (value === "") {
              update("user_id", "");
              return;
            }
            const parsed = Number(value);
            update("user_id", Number.isNaN(parsed) ? "" : parsed);
          }}
          min={1}
          placeholder="Required"
          required
        />
      </label>

      <label>
        Context item IDs
        <input
          type="text"
          value={contextInput}
          onChange={(event) => {
            const raw = event.target.value;
            setContextInput(raw);
            const ids = raw
              .split(",")
              .map((value) => value.trim())
              .filter(Boolean)
              .map((value) => Number(value))
              .filter((value) => Number.isFinite(value) && value > 0);
            update("context_item_ids", ids.length ? ids : null);
          }}
          placeholder="Comma-separated (optional)"
        />
        {draft.context_item_ids && draft.context_item_ids.length > 0 ? (
          <span className="hint">{draft.context_item_ids.length} parsed</span>
        ) : null}
      </label>

      <label>
        Num candidates
        <input
          type="number"
          value={draft.num_candidates}
          onChange={(event) => {
            const value = event.target.value;
            if (value === "") {
              update("num_candidates", "");
              return;
            }
            const parsed = Number(value);
            update("num_candidates", Number.isNaN(parsed) ? "" : parsed);
          }}
          min={1}
          placeholder="e.g. 10"
          required
        />
      </label>

      <label>
        Source page
        <select
          value={draft.source_page}
          onChange={(event) => update("source_page", event.target.value as SourceType)}
        >
          {SOURCE_TYPES.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>

      <label>
        Current time
        <input
          type="datetime-local"
          value={datetimeLocal}
          onChange={(event) =>
            update(
              "current_time",
              event.target.value ? new Date(event.target.value).toISOString() : ""
            )
          }
        />
        <span className="hint">Leave empty to use server's current time</span>
      </label>

      <div className="actions">
        <button type="submit" disabled={submitting || !catalogReady}>
          {submitting ? "Fetching..." : "Fetch recommendations"}
        </button>
        <button type="button" className="ghost" onClick={reset} disabled={submitting}>
          Reset
        </button>
      </div>
      {!catalogReady ? (
        <span className="hint">Waiting for catalog to load…</span>
      ) : null}
    </form>
  );
}

import { useState, type FormEvent } from "react";
import { DEFAULT_DRAFT, readDraft, saveDraft, splitIds, toRequest, type Draft } from "../draft";
import { SOURCE_TYPES, type FieldError, type RecommendRequest, type SourceType } from "../types";
import { ErrorBanner } from "./ErrorBanner";

type Props = {
  onSubmit: (request: RecommendRequest) => void;
  submitting: boolean;
};

export function RequestForm({ onSubmit, submitting }: Props) {
  const [draft, setDraft] = useState<Draft>(readDraft);
  const [errors, setErrors] = useState<FieldError[]>([]);

  const update = <K extends keyof Draft>(key: K, value: Draft[K]) => {
    setDraft((prev) => ({ ...prev, [key]: value }));
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const result = toRequest(draft);
    if (Array.isArray(result)) {
      setErrors(result);
      return;
    }
    setErrors([]);
    saveDraft(draft);
    onSubmit(result);
  };

  const reset = () => {
    setDraft(DEFAULT_DRAFT);
    setErrors([]);
    saveDraft(DEFAULT_DRAFT);
  };

  const contextCount = splitIds(draft.context_item_ids).length;

  return (
    <form onSubmit={submit} className="form">
      <label>
        User ID
        <input
          inputMode="numeric"
          value={draft.user_id}
          onChange={(event) => update("user_id", event.target.value)}
          placeholder="Required"
          required
        />
      </label>

      <label>
        Context item IDs
        <input
          value={draft.context_item_ids}
          onChange={(event) => update("context_item_ids", event.target.value)}
          placeholder="Comma-separated (optional)"
        />
        {contextCount > 0 ? <span className="hint">{contextCount} IDs</span> : null}
      </label>

      <label>
        Num candidates
        <input
          type="number"
          value={draft.num_candidates}
          onChange={(event) => update("num_candidates", event.target.value)}
          min={1}
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
          value={draft.current_time}
          onChange={(event) => update("current_time", event.target.value)}
        />
        <span className="hint">Local time, sent as UTC. Empty: server time</span>
      </label>

      <div className="actions">
        <button type="submit" disabled={submitting}>
          {submitting ? "Fetching..." : "Fetch recommendations"}
        </button>
        <button type="button" className="ghost" onClick={reset} disabled={submitting}>
          Reset
        </button>
      </div>
      {errors.length > 0 ? <ErrorBanner title="Invalid request" details={errors} /> : null}
    </form>
  );
}

import type { FieldError, RecommendRequest, RecommendResponse } from "./types";

export class ApiError extends Error {
  details: FieldError[];

  constructor(message: string, details: FieldError[] = []) {
    super(message);
    this.name = "ApiError";
    this.details = details;
  }
}

export const apiBase = import.meta.env.VITE_API_BASE ?? "";

const JSON_TOKEN = /"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
const INTEGER = /^-?\d+$/;

export function parseJson(text: string): unknown {
  return JSON.parse(
    text.replace(JSON_TOKEN, (token) => (INTEGER.test(token) ? `"${token}"` : token))
  );
}

export function encodeRequest(request: RecommendRequest): string {
  const { user_id, context_item_ids, ...rest } = request;
  const ids = context_item_ids ? `[${context_item_ids.join(",")}]` : "null";
  return `{"user_id":${user_id},"context_item_ids":${ids},${JSON.stringify(rest).slice(1)}`;
}

export function toCurl(request: RecommendRequest): string {
  const url = new URL(`${apiBase}/recommend`, window.location.href).href;
  return `curl -sS -X POST '${url}' -H 'Content-Type: application/json' -d '${encodeRequest(request)}'`;
}

function tryParseJson(text: string): unknown {
  try {
    return parseJson(text);
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function errorFromBody(status: number, text: string): ApiError {
  const body = tryParseJson(text);
  const detail = isRecord(body) ? body.detail : undefined;
  if (Array.isArray(detail)) {
    const details = detail.filter(isRecord).map((d) => ({
      field: (Array.isArray(d.loc) ? d.loc.filter((part) => part !== "body").join(".") : "") || "request",
      msg: String(d.msg),
    }));
    return new ApiError(`Validation failed (HTTP ${status})`, details);
  }
  const reason = typeof detail === "string" ? detail : text;
  return new ApiError(reason ? `HTTP ${status}: ${reason}` : `HTTP ${status}`);
}

function isRecommendResponse(body: unknown): body is RecommendResponse {
  return (
    isRecord(body) &&
    Array.isArray(body.candidates) &&
    body.candidates.every((id) => typeof id === "string") &&
    typeof body.next_page_token === "string"
  );
}

export async function recommend(
  request: RecommendRequest,
  signal: AbortSignal
): Promise<{ response: RecommendResponse; text: string }> {
  let res: Response;
  try {
    res = await fetch(`${apiBase}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: encodeRequest(request),
      signal,
    });
  } catch (err) {
    if (signal.aborted) throw err;
    throw new ApiError(`Network error: ${err instanceof Error ? err.message : String(err)}`);
  }
  const text = await res.text();
  if (!res.ok) throw errorFromBody(res.status, text);
  const body = tryParseJson(text);
  if (!isRecommendResponse(body)) {
    throw new ApiError(
      `HTTP ${res.status}: expected {candidates: int[], next_page_token: str}, got ${text.slice(0, 200)}`
    );
  }
  return { response: body, text };
}

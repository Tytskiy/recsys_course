import type { FastAPIDetail, RecommendRequest, RecommendResponse } from "./types";

export class ApiError extends Error {
  status: number;
  details: FastAPIDetail[] | null;

  constructor(message: string, status: number, details: FastAPIDetail[] | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

const apiBase = import.meta.env.VITE_API_BASE ?? "";

function isFastAPIErrorBody(body: unknown): body is { detail: FastAPIDetail[] } {
  if (!body || typeof body !== "object") return false;
  const detail = (body as { detail?: unknown }).detail;
  if (!Array.isArray(detail)) return false;
  return detail.every(
    (d) =>
      d &&
      typeof d === "object" &&
      Array.isArray((d as FastAPIDetail).loc) &&
      typeof (d as FastAPIDetail).msg === "string"
  );
}

async function postJson<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal
): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    let parsed: unknown = null;
    const text = await response.text();
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        // not JSON, fall through
      }
    }
    if (isFastAPIErrorBody(parsed)) {
      throw new ApiError(
        `Validation failed (HTTP ${response.status})`,
        response.status,
        parsed.detail
      );
    }
    if (parsed && typeof parsed === "object" && "detail" in parsed) {
      const detail = (parsed as { detail: unknown }).detail;
      if (typeof detail === "string") {
        throw new ApiError(detail, response.status, null);
      }
    }
    throw new ApiError(
      text || `HTTP ${response.status}`,
      response.status,
      null
    );
  }

  return response.json() as Promise<T>;
}

export const api = {
  recommend: (payload: RecommendRequest, signal?: AbortSignal) =>
    postJson<RecommendResponse>("/recommend", payload, signal),
  apiBase,
};

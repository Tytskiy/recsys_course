import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, encodeRequest, parseJson, recommend } from "./api";
import type { RecommendRequest } from "./types";

const BIG = "18445492888512500135";

const REQUEST: RecommendRequest = {
  user_id: "1",
  num_candidates: 2,
  source_page: "ST_Feed",
  current_time: null,
  context_item_ids: [BIG],
};

function stubFetch(status: number, body: string) {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { status })));
}

async function recommendError(): Promise<ApiError> {
  const err = await recommend(REQUEST, new AbortController().signal).catch((e: unknown) => e);
  if (!(err instanceof ApiError)) throw new Error(`expected ApiError, got ${String(err)}`);
  return err;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("parseJson", () => {
  it("keeps uint64 integers exact as decimal strings", () => {
    expect(parseJson(`{"candidates":[${BIG},7]}`)).toEqual({ candidates: [BIG, "7"] });
  });

  it("leaves digits inside strings and non-integer numbers alone", () => {
    expect(parseJson('{"t":"a\\"1,2","x":1.5,"e":1e3}')).toEqual({ t: 'a"1,2', x: 1.5, e: 1000 });
  });
});

describe("encodeRequest", () => {
  it("writes ids as bare JSON integers without rounding", () => {
    const body = encodeRequest({ ...REQUEST, page_token: "abc=" });
    expect(body).toBe(
      `{"user_id":1,"context_item_ids":[${BIG}],"num_candidates":2,"source_page":"ST_Feed","current_time":null,"page_token":"abc="}`
    );
    expect(parseJson(body)).toMatchObject({ user_id: "1", context_item_ids: [BIG] });
  });

  it("sends null context ids and omits a missing page token", () => {
    expect(encodeRequest({ ...REQUEST, context_item_ids: null })).toBe(
      '{"user_id":1,"context_item_ids":null,"num_candidates":2,"source_page":"ST_Feed","current_time":null}'
    );
  });
});

describe("recommend", () => {
  it("returns exact candidate ids and the raw text", async () => {
    const text = `{"candidates":[${BIG},3],"next_page_token":"CQAA"}`;
    stubFetch(200, text);
    const result = await recommend(REQUEST, new AbortController().signal);
    expect(result).toEqual({ response: { candidates: [BIG, "3"], next_page_token: "CQAA" }, text });
  });

  it("maps a FastAPI 422 body to field errors without the body prefix", async () => {
    stubFetch(
      422,
      '{"detail":[{"type":"literal_error","loc":["body","source_page"],"msg":"Input should be \'ST_Feed\'","input":"x"},' +
        '{"type":"int_parsing","loc":["body","context_item_ids",0],"msg":"Input should be a valid integer"}]}'
    );
    const err = await recommendError();
    expect(err.message).toBe("Validation failed (HTTP 422)");
    expect(err.details).toEqual([
      { field: "source_page", msg: "Input should be 'ST_Feed'" },
      { field: "context_item_ids.0", msg: "Input should be a valid integer" },
    ]);
  });

  it.each([
    [500, "Internal Server Error", "HTTP 500: Internal Server Error"],
    [404, '{"detail":"Not Found"}', "HTTP 404: Not Found"],
    [502, "", "HTTP 502"],
  ])("reports HTTP %i with its status", async (status, body, message) => {
    stubFetch(status, body);
    expect((await recommendError()).message).toBe(message);
  });

  it("rejects a 200 response that breaks the contract", async () => {
    stubFetch(200, '{"items":[1]}');
    expect((await recommendError()).message).toMatch(/^HTTP 200: expected \{candidates/);
  });

  it("reports a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    expect((await recommendError()).message).toBe("Network error: Failed to fetch");
  });
});

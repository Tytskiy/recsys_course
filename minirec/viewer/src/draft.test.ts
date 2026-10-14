import { describe, expect, it } from "vitest";
import { DEFAULT_DRAFT, toRequest } from "./draft";

describe("toRequest", () => {
  it("builds a request with normalized ids and an ISO UTC time", () => {
    const local = "2026-01-02T12:30";
    expect(
      toRequest({
        ...DEFAULT_DRAFT,
        user_id: " 007 ",
        context_item_ids: "18445492888512500135, 0042\n-3",
        current_time: local,
      })
    ).toEqual({
      user_id: "7",
      num_candidates: 10,
      source_page: "ST_Upsale",
      current_time: new Date(local).toISOString(),
      context_item_ids: ["18445492888512500135", "42", "-3"],
    });
  });

  it("sends nulls for empty optional fields", () => {
    expect(toRequest({ ...DEFAULT_DRAFT, user_id: "0" })).toMatchObject({
      current_time: null,
      context_item_ids: null,
    });
  });

  it("lists every invalid field", () => {
    expect(
      toRequest({
        ...DEFAULT_DRAFT,
        user_id: "1.5",
        num_candidates: "0",
        context_item_ids: "12, abc, 1.5",
        current_time: "not a date",
      })
    ).toEqual([
      { field: "user_id", msg: "must be an integer" },
      { field: "num_candidates", msg: "must be a positive integer" },
      { field: "context_item_ids", msg: "not integers: abc, 1.5" },
      { field: "current_time", msg: "invalid date" },
    ]);
  });
});

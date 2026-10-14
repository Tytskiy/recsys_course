import type { SourceType } from "./types";

export const SOURCE_TYPES: readonly SourceType[] = [
  "ST_CheckoutUpsale",
  "ST_Upsale",
  "ST_Feed",
  "ST_SearchComplementRec",
  "ST_Search",
  "ST_PreviousBuyCarousel",
  "ST_OnClickDelivery",
  "ST_SearchStartRec",
  "ST_Uplift",
  "ST_Catalog",
  "ST_PreviousBuyHub",
  "ST_ItemPageCarousel",
] as const;

export const STORAGE_KEY = "rec-engine-viewer.draft.v1";

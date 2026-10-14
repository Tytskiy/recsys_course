export const SOURCE_TYPES = [
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

export type SourceType = (typeof SOURCE_TYPES)[number];

// IDs are uint64 and exceed Number.MAX_SAFE_INTEGER, so they stay decimal strings end to end.
export type Id = string;

export type RecommendRequest = {
  user_id: Id;
  num_candidates: number;
  source_page: SourceType;
  current_time: string | null;
  context_item_ids: Id[] | null;
  page_token?: string;
};

export type RecommendResponse = {
  candidates: Id[];
  next_page_token: string;
};

export type ItemMetadata = {
  item_id: Id;
  title: string | null;
  description: string | null;
  image_url: string | null;
};

export type FieldError = {
  field: string;
  msg: string;
};

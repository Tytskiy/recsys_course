export type SourceType =
  | "ST_CheckoutUpsale"
  | "ST_Upsale"
  | "ST_Feed"
  | "ST_SearchComplementRec"
  | "ST_Search"
  | "ST_PreviousBuyCarousel"
  | "ST_OnClickDelivery"
  | "ST_SearchStartRec"
  | "ST_Uplift"
  | "ST_Catalog"
  | "ST_PreviousBuyHub"
  | "ST_ItemPageCarousel";

export type RecommendRequest = {
  user_id: number;
  num_candidates: number;
  source_page: SourceType;
  current_time?: string | null;
  context_item_ids?: number[] | null;
  page_token?: string | null;
};

export type RecommendResponse = {
  candidates: number[];
  next_page_token: string;
};

export type ItemMetadata = {
  item_id: number;
  title: string | null;
  description: string | null;
  image_url: string | null;
};

export type DraftRequest = {
  user_id: number | "";
  num_candidates: number | "";
  source_page: SourceType;
  current_time: string;
  context_item_ids: number[] | null;
};

export type ResultItem = {
  item: ItemMetadata;
  inCatalog: boolean;
};

export type FastAPIDetail = {
  loc: (string | number)[];
  msg: string;
  type: string;
};

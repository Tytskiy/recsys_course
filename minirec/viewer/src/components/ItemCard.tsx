import { memo } from "react";
import type { Id, ItemMetadata } from "../types";

type Props = {
  id: Id;
  item: ItemMetadata | undefined;
  rank: number;
  page: number;
  unknown: boolean;
  repeat: boolean;
};

export const ItemCard = memo(function ItemCard({ id, item, rank, page, unknown, repeat }: Props) {
  const title = item?.title ?? `Item ${id}`;
  const imageUrl = item?.image_url;

  return (
    <article className="card">
      <div className="card-media">
        {imageUrl ? (
          <img src={imageUrl} alt={title} loading="lazy" />
        ) : (
          <div className="placeholder">No image</div>
        )}
        <span className="rank">
          #{rank}
          {page > 1 ? ` · p${page}` : ""}
        </span>
        <div className="badges">
          {unknown ? (
            <span className="badge missing" title="Item not found in local catalog">
              unknown
            </span>
          ) : null}
          {repeat ? (
            <span className="badge repeat" title="Already returned on an earlier page">
              repeat
            </span>
          ) : null}
        </div>
      </div>
      <div className="card-body">
        <h3>{title}</h3>
        <p>{item?.description ?? "No description available."}</p>
        <span className="item-id">ID {id}</span>
        {imageUrl ? (
          <a href={imageUrl} target="_blank" rel="noreferrer">
            Open image
          </a>
        ) : null}
      </div>
    </article>
  );
});

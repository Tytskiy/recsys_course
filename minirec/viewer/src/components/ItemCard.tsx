import type { ResultItem } from "../types";

type Props = {
  result: ResultItem;
  rank: number;
};

export function ItemCard({ result, rank }: Props) {
  const { item, inCatalog } = result;
  const title = item.title ?? `Item ${item.item_id}`;

  return (
    <article className="card">
      <div className="card-media">
        {item.image_url ? (
          <img src={item.image_url} alt={title} loading="lazy" />
        ) : (
          <div className="placeholder">No image</div>
        )}
        <span className="rank">#{rank}</span>
        {!inCatalog ? (
          <span className="badge missing" title="Item not found in local catalog">
            unknown
          </span>
        ) : null}
      </div>
      <div className="card-body">
        <h3>{title}</h3>
        <p>{item.description ?? "No description available."}</p>
        <span className="item-id">ID {item.item_id}</span>
        {item.image_url ? (
          <a href={item.image_url} target="_blank" rel="noreferrer">
            Open image
          </a>
        ) : null}
      </div>
    </article>
  );
}

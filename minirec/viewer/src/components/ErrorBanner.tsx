import type { FieldError } from "../types";

type Props = {
  title: string;
  details?: FieldError[];
};

export function ErrorBanner({ title, details = [] }: Props) {
  return (
    <div className="error" role="alert">
      <p className="error-title">{title}</p>
      {details.length > 0 ? (
        <ul className="error-list">
          {details.map((detail, index) => (
            <li key={index}>
              <code>{detail.field}</code> — {detail.msg}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

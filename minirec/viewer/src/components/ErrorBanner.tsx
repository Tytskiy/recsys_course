import { ApiError } from "../api";

type Props = {
  error: ApiError | Error | null;
};

function formatLoc(loc: (string | number)[]): string {
  return loc.filter((part) => part !== "body").join(".") || "request";
}

export function ErrorBanner({ error }: Props) {
  if (!error) return null;

  if (error instanceof ApiError && error.details && error.details.length > 0) {
    return (
      <div className="error" role="alert">
        <p className="error-title">
          {error.message}
        </p>
        <ul className="error-list">
          {error.details.map((detail, index) => (
            <li key={index}>
              <code>{formatLoc(detail.loc)}</code> — {detail.msg}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <p className="error" role="alert">
      {error.message}
    </p>
  );
}

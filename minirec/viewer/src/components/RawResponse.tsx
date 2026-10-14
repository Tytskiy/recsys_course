type Props = {
  data: object | null;
};

export function RawResponse({ data }: Props) {
  if (!data) return null;
  return (
    <details className="raw" open>
      <summary className="label">Raw response</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

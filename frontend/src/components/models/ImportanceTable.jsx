export default function ImportanceTable({ title, data, valueKey }) {
  if (!data) return null;

  const entries = Object.entries(data)
    .map(([key, value]) => ({
      key,
      val: typeof value === "object" && valueKey ? value[valueKey] : value,
    }))
    .filter((entry) => typeof entry.val === "number" && !Number.isNaN(entry.val))
    .sort((a, b) => b.val - a.val)
    .slice(0, 10);

  if (entries.length === 0) return null;

  return (
    <div className="rounded border bg-white p-3 shadow-sm">
      <h6 className="mb-2 border-b pb-1 text-[10px] font-bold uppercase tracking-wider text-gray-500">
        {title}
      </h6>
      <ul className="space-y-1 text-xs">
        {entries.map(({ key, val }) => (
          <li key={key} className="group flex items-center justify-between rounded px-1 hover:bg-gray-50">
            <span className="mr-2 max-w-[120px] truncate" title={key}>
              {key}
            </span>
            <span className="font-mono text-gray-700">{val.toFixed(4)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

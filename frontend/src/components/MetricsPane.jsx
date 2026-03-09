const METRIC_LABELS = {
  rmse: "RMSE",
  mae: "MAE",
  nmae: "NMAE",
  r_2: "R²",
};

function MetricBadge({ label, value }) {
  const color =
    label === "R²"
      ? value >= 0.8
        ? "text-green-700 bg-green-50"
        : value >= 0.5
        ? "text-yellow-700 bg-yellow-50"
        : "text-red-700 bg-red-50"
      : "text-gray-700 bg-gray-50";

  return (
    <div className={`flex justify-between items-center px-3 py-2 rounded-lg ${color}`}>
      <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
      <span className="text-sm font-semibold tabular-nums">
        {typeof value === "number" ? value.toFixed(4) : "—"}
      </span>
    </div>
  );
}

export default function MetricsPane({ chart }) {
  if (!chart?.metrics) return null;

  const parts = chart.title.split(" \u2014 ");
  const subtitle = parts.slice(1).join(" — ");

  return (
    <div className="w-80 flex-shrink-0 rounded-xl border bg-white shadow-sm p-4 flex flex-col h-[500px]">
      <h3 className="text-sm font-semibold text-gray-800 border-b pb-2 mb-3">
        {subtitle}
      </h3>
      <div className="flex flex-col gap-1.5 flex-1 justify-center">
        {Object.entries(METRIC_LABELS).map(([key, label]) => (
          <MetricBadge key={key} label={label} value={chart.metrics[key]} />
        ))}
      </div>
    </div>
  );
}

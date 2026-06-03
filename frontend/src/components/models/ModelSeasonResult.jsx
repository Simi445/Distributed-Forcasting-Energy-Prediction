import Chart from "@/components/Chart.jsx";
import MetricsPane from "@/components/MetricsPane.jsx";

export default function ModelSeasonResult({ season, data }) {
  return (
    <div className="mb-10 flex flex-col gap-5 last:mb-0">
      <div className="w-full">{data.metrics && <MetricsPane chart={data} season={season} />}</div>
      <div className="w-full">
        <div className="flex flex-col rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-600">
            {season} — Predictions
          </p>
          {data.html && <Chart html={data.html} />}
        </div>
      </div>
    </div>
  );
}

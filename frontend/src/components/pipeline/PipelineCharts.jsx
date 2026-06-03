import { useEffect, useMemo, useState } from "react";
import Chart from "@/components/Chart.jsx";

function parseEnergyPlot(key, html) {
  const parts = key.split("__");
  if (parts[0] !== "energy" || parts.length < 3) return null;

  const category = parts
    .slice(2)
    .join("__")
    .replace(/^primary_use_/, "")
    .replaceAll("_", " ");

  return {
    key,
    html,
    buildingId: parts[1],
    category,
    label: `${category} · ${parts[1]}`,
  };
}

export default function PipelineCharts({ plots }) {
  const energyPlots = useMemo(
    () =>
      Object.entries(plots ?? {})
        .map(([key, html]) => parseEnergyPlot(key, html))
        .filter(Boolean),
    [plots],
  );
  const [selectedEnergyKey, setSelectedEnergyKey] = useState("");

  useEffect(() => {
    if (energyPlots.length === 0) {
      setSelectedEnergyKey("");
      return;
    }

    if (!energyPlots.some((plot) => plot.key === selectedEnergyKey)) {
      setSelectedEnergyKey(energyPlots[0].key);
    }
  }, [energyPlots, selectedEnergyKey]);

  const selectedEnergyPlot = energyPlots.find((plot) => plot.key === selectedEnergyKey) ?? energyPlots[0];

  return (
    <div className="flex w-full flex-col gap-8">
      {energyPlots.length > 0 && (
        <section className="flex min-w-0 flex-col gap-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-base font-semibold text-slate-700">Energy Consumption</h2>
            <div className="flex flex-wrap gap-2">
              {energyPlots.map((plot) => (
                <button
                  key={plot.key}
                  type="button"
                  onClick={() => setSelectedEnergyKey(plot.key)}
                  className={`h-9 rounded-lg border px-3 text-xs font-semibold transition-colors ${
                    plot.key === selectedEnergyPlot?.key
                      ? "border-slate-950 bg-slate-950 text-white"
                      : "border-slate-300 bg-white text-slate-700 hover:bg-slate-100"
                  }`}
                  title={`Building ${plot.buildingId}`}
                >
                  {plot.label}
                </button>
              ))}
            </div>
          </div>

          {selectedEnergyPlot && <Chart html={selectedEnergyPlot.html} />}
        </section>
      )}

      {plots?.acf && <Chart title="ACF (lag 24)" html={plots.acf} />}
    </div>
  );
}

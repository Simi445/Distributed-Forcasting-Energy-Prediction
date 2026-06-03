import { ArrowLeft } from "lucide-react";
import ModelModeSection from "./ModelModeSection.jsx";

export default function ModelDetails({ selectedModel, modelCharts, onBack }) {
  const charts = modelCharts.filter((chart) => chart.title.startsWith(`${selectedModel} —`));

  return (
    <div className="mx-auto w-full max-w-7xl">
      <button
        onClick={onBack}
        className="mb-6 flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Dashboard
      </button>
      <h2 className="mb-6 text-xl font-bold text-slate-900">{selectedModel} — Seasonal vs Global</h2>
      <div className="flex w-full flex-col gap-10">
        {charts.map(({ title, modes }) => {
          const buildingId = title.split(" — ")[1];

          return (
            <div key={title} className="border-b border-slate-300 pb-10 last:border-0">
              <h3 className="mb-6 border-b-2 pb-2 text-2xl font-bold text-slate-800">{buildingId}</h3>
              <ModelModeSection mode="global" seasons={modes.global} />
              <ModelModeSection mode="seasonal" seasons={modes.seasonal} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

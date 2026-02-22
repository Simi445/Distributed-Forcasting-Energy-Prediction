import { useState } from "react";
import Chart from "./components/Chart.jsx";
import { FadeLoader } from "react-spinners";

const STEPS = [
  { label: "Load Data",        endpoint: "/api/load_data" },
  { label: "Merge Data",       endpoint: "/api/merge_data" },
  { label: "Feature Engineer", endpoint: "/api/feature_engineer" },
  { label: "ACF",              endpoint: "/api/acf" },
  { label: "Final Selection",  endpoint: "/api/final_selection" },
];

const stepStyles = {
  done:    "bg-green-100 text-green-800",
  error:   "bg-red-100 text-red-800",
  running: "bg-yellow-100 text-yellow-800",
  pending: "bg-gray-100 text-gray-500",
};

const stepIcons = { done: "✓", error: "✗", pending: "○" };

function StepIcon({ state }) {
  if (state === "running") return <FadeLoader color="#854d0e" height={2} width={2} margin={1} />;
  return <span>{stepIcons[state]}</span>;
}

export default function App() {
  const [stepStatus, setStepStatus] = useState({});
  const [plots, setPlots] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runStep(endpoint, label) {
    setStepStatus((s) => ({ ...s, [label]: "running" }));
    const res = await fetch(endpoint);
    if (!res.ok) {
      setStepStatus((s) => ({ ...s, [label]: "error" }));
      throw new Error(`${label} failed: ${res.status}`);
    }
    setStepStatus((s) => ({ ...s, [label]: "done" }));
  }

  async function runPipeline() {
    setLoading(true);
    setError(null);
    setPlots(null);
    setStepStatus({});
    try {
      for (const { label, endpoint } of STEPS) {
        await runStep(endpoint, label);
      }
      const res = await fetch("/api/plot");
      if (!res.ok) throw new Error(`Plot generation failed: ${res.status}`);
      const data = await res.json();
      setPlots(data.response);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      <div className="max-w-5xl mx-auto px-6 py-10">

        <header className="flex items-center gap-6 mb-8">
          <h1 className="text-2xl font-bold tracking-tight">Licenta Dashboard</h1>
          <button
            onClick={runPipeline}
            disabled={loading}
            className="px-5 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Running..." : "Run Pipeline"}
          </button>
        </header>

        {Object.keys(stepStatus).length > 0 && (
          <div className="flex flex-wrap gap-2 mb-6">
            {STEPS.map(({ label }) => {
              const state = stepStatus[label] ?? "pending";
              return (
                <span
                  key={label}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${stepStyles[state]}`}
                >
                  <StepIcon state={state} />
                  {label}
                </span>
              );
            })}
          </div>
        )}

        {error && (
          <div className="bg-red-50 text-red-700 border border-red-200 rounded-lg px-4 py-3 mb-6 text-sm">
            {error}
          </div>
        )}

        {plots && (
          <div className="flex flex-col gap-8">
            <Chart title="Energy Consumption" html={plots.energy} />
            <Chart title="ACF (lag 24)"        html={plots.acf} />
          </div>
        )}
      </div>
    </div>
  );
}

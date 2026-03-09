import { useState } from "react";
import Chart from "./components/Chart.jsx";
import { Check, X, Circle, Loader2, ArrowLeft } from 'lucide-react';
import { DottedSurface } from "@/components/ui/dotted-surface";
import IdleComponent from "@/components/IdleComponent.jsx";
import MetricsPane from "@/components/MetricsPane.jsx";
import pipelineConfig from "@/config/pipeline.json";

const { steps: STEPS, modelSteps: MODEL_STEPS, stepStyles } = pipelineConfig;

const stepIcons = { 
  running: <Loader2 className="animate-spin" />,
  done: <Check />, 
  error: <X/>, 
  pending: <Circle/> 
};

function StepIcon({ state }) {
      return <span>{stepIcons[state]}</span>;
  }

export default function App() {
  const [stepStatus, setStepStatus] = useState({});
  const [plots, setPlots] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pipelineDone, setPipelineDone] = useState(false);
  const [modelStepStatus, setModelStepStatus] = useState({});
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState(null);


  const [modelCharts, setModelCharts] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);

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
      setPipelineDone(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runModelPipeline() {
    setModelLoading(true);
    setModelError(null);
    setModelStepStatus({});
    setModelCharts([]);
    try {
      for (const { cardTitle, modes } of MODEL_STEPS) {
        setModelStepStatus((s) => ({ ...s, [cardTitle]: "running" }));
        for (const [mode, endpoint] of Object.entries(modes)) {
          const res = await fetch(endpoint);
          if (!res.ok) {
            setModelStepStatus((s) => ({ ...s, [cardTitle]: "error" }));
            throw new Error(`${cardTitle} (${mode}) failed: ${res.status}`);
          }
          const data = await res.json();
          const charts = Object.entries(data.response).map(([key, info]) => ({
            title: `${cardTitle} — ${mode} — ${key}`,
            html: info.html,
            metrics: info.metrics ?? null,
            pImportance: info.p_importance ?? null,
            fImportance: info.f_importance ?? null,
          }));
          setModelCharts((prev) => [...prev, ...charts]);
        }
        setModelStepStatus((s) => ({ ...s, [cardTitle]: "done" }));
      }
    } catch (e) {
      setModelError(e.message);
    } finally {
      setModelLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-start text-gray-800 overflow-x-auto">
      <DottedSurface />
      <div className="max-w-7xl w-full mx-auto px-6 py-10">
        <header className="flex items-center justify-center gap-6 mb-8">
          <h1 className="text-2xl font-bold tracking-tight">Licenta Dashboard</h1>
          <button
            onClick={runPipeline}
            disabled={loading}
            className="px-5 py-2 bg-black text-white rounded-lg font-medium hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Running..." : "Run Pipeline"}
          </button>
          <button
            onClick={runModelPipeline}
            disabled={!pipelineDone || modelLoading}
            className="px-5 py-2 bg-black text-white rounded-lg font-medium hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {modelLoading ? "Training..." : "Run Models"}
          </button>
        </header>

        {Object.keys(stepStatus).length > 0 && (
          <div className="flex items-center justify-center gap-2 mb-6">
            {STEPS.map(({ label }) => {
              const state = stepStatus[label] ?? "pending";
              return (
                <span
                  key={label}
                  className={`flex items-center justify-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${stepStyles[state]}`}
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

        {Object.keys(modelStepStatus).length > 0 && (
          <div className="flex items-center justify-center gap-2 mb-6">
            {MODEL_STEPS.map(({ cardTitle }) => {
              const state = modelStepStatus[cardTitle] ?? "pending";
              return (
                <span
                  key={cardTitle}
                  className={`flex items-center justify-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${stepStyles[state]}`}
                >
                  <StepIcon state={state} />
                  {cardTitle}
                </span>
              );
            })}
          </div>
        )}

        {modelError && (
          <div className="bg-red-50 text-red-700 border border-red-200 rounded-lg px-4 py-3 mb-6 text-sm">
            {modelError}
          </div>
        )}

        {selectedModel ? (
          <div className="w-full">
            <button
              onClick={() => setSelectedModel(null)}
              className="flex items-center gap-2 px-4 py-2 mb-6 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Dashboard
            </button>
            <h2 className="text-xl font-bold mb-6">{selectedModel} — Seasonal vs Global</h2>
            <div className="flex flex-col gap-8 w-full">
              {modelCharts
                .filter((c) => c.title.startsWith(selectedModel + " \u2014"))
                .map((c, i) => (
                  <div key={i} className="flex gap-4 items-end">
                    <div className="flex-1 min-w-0">
                      <Chart title={c.title} html={c.html} />
                    </div>
                    {c.metrics && <MetricsPane chart={c} />}
                  </div>
                ))}
            </div>
          </div>
        ) : (
          (plots || Object.keys(modelStepStatus).length > 0) && (
            <div className="flex gap-8">
              <div className="flex-1 min-w-0 transition-all duration-500 ease-in-out">
                {plots && (
                  <div className="flex flex-col gap-8 w-full">
                    <Chart title="Energy Consumption" html={plots.energy} />
                    <Chart title="ACF (lag 24)"        html={plots.acf} />
                  </div>
                )}
              </div>

              {MODEL_STEPS.every(step => modelStepStatus[step.cardTitle] === "done") && (
                <div className="w-64 flex-shrink-0 transition-all duration-500 ease-in-out">
                  <IdleComponent changeState={(name) => setSelectedModel(name)} models={MODEL_STEPS}/>
                </div>
              )}
            </div>
          )
        )}
      </div>
    </div>
  );
}

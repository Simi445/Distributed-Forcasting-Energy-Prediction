import { useState } from "react";
import Chart from "./components/Chart.jsx";
import { Check, X, Circle, Loader2, ArrowLeft, Baseline } from 'lucide-react';
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

function ImportanceTable({ title, data, valueKey }) {
  if (!data) return null;
  const entries = Object.entries(data)
    .map(([k, v]) => ({ key: k, val: typeof v === 'object' && valueKey ? v[valueKey] : v }))
    .filter(e => typeof e.val === 'number' && !isNaN(e.val))
    .sort((a, b) => b.val - a.val)
    .slice(0, 10);
    
  if (entries.length === 0) return null;

  return (
    <div className="bg-white border rounded shadow-sm p-3">
      <h6 className="text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2 border-b pb-1">{title}</h6>
      <ul className="space-y-1 text-xs">
        {entries.map(({ key, val }) => (
          <li key={key} className="flex justify-between items-center hover:bg-gray-50 rounded px-1 group">
            <span className="truncate mr-2 max-w-[120px]" title={key}>{key}</span>
            <span className="font-mono text-gray-700">{val.toFixed(4)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
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
    
    const groupedCharts = {};
    
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
          
          Object.entries(data.response).forEach(([key, info]) => {
            const [build_id, season, mode] = key.split(",");

            const groupKey = `${cardTitle} — ${build_id}`;
            
            const usedMode = typeof(mode) !== "undefined" ? mode :'seasonal';

            if (!groupedCharts[groupKey]) {
              groupedCharts[groupKey] = {};
            }
            if (!groupedCharts[groupKey][usedMode]) {
              groupedCharts[groupKey][usedMode] = {};
            }
            if (!groupedCharts[groupKey][usedMode][season]) {
              groupedCharts[groupKey][usedMode][season] = {};
            }
            
            groupedCharts[groupKey][usedMode][season] = {
              html: info.html,
              metrics: info.metrics ?? null,
              pImportance: info.p_importance ?? null,
              fImportance: info.f_importance ?? null,
            };
          });
        }
        
        setModelStepStatus((s) => ({ ...s, [cardTitle]: "done" }));
      }
      
      Object.entries(groupedCharts).map(([key, value], id) => {
        console.log(key, value);
      });
      setModelCharts(Object.entries(groupedCharts).map(([title, modes]) => ({ title, modes })));
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
        {!selectedModel && (
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
        )}

        {!selectedModel && Object.keys(stepStatus).length > 0 && (
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

        {!selectedModel && error && (
          <div className="bg-red-50 text-red-700 border border-red-200 rounded-lg px-4 py-3 mb-6 text-sm">
            {error}
          </div>
        )}

        {!selectedModel && Object.keys(modelStepStatus).length > 0 && (
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

        {!selectedModel && modelError && (
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
            <div className="flex flex-col gap-12 w-full">
              {modelCharts
                .filter((c) => c.title.startsWith(selectedModel + " \u2014"))
                .map(({ title, modes }, i) => {
                  const buildingId = title.split(" \u2014 ")[1];
                  
                  return (
                    <div key={i} className="border-b border-gray-300 pb-12 last:border-0">
                      <h3 className="text-2xl font-bold text-gray-800 border-b-2 pb-2 mb-8">{buildingId}</h3>
                      
                      <div className="mb-12">
                        {modes.global && <h4 className="font-semibold text-center bg-blue-100 p-2 rounded text-blue-800 uppercase text-sm tracking-wider mb-4">Global Model</h4>}
                        {modes.global && Object.entries(modes.global).map(([season, data]) => (
                          <div key={`global-${season}`} className="flex flex-col gap-6 mb-12 last:mb-0">
                            <div className="w-full">
                              {data.metrics && <MetricsPane chart={data} season={season} />}
                            </div>
                            <div className="w-full">
                              <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col">
                                <p className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">{season} — Predictions</p>
                                {data.html && <Chart html={data.html} />}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div>
                        {modes.seasonal && <h4 className="font-semibold text-center bg-amber-100 p-2 rounded text-amber-800 uppercase text-sm tracking-wider mb-4">Seasonal Model</h4>}
                        {modes.seasonal && Object.entries(modes.seasonal).map(([season, data]) => (
                          <div key={`seasonal-${season}`} className="flex flex-col gap-6 mb-12 last:mb-0">
                            <div className="w-full">
                              {data.metrics && <MetricsPane chart={data} season={season} />}
                            </div>
                            <div className="w-full">
                              <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col">
                                <p className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">{season} — Predictions</p>
                                {data.html && <Chart html={data.html} />}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        ) : (
          (plots || Object.keys(modelStepStatus).length > 0) && (
            <div className="flex flex-col gap-12 w-full max-w-5xl mx-auto">
              <div className="w-full transition-all duration-500 ease-in-out">
                {plots && (
                  <div className="flex flex-col gap-8 w-full">
                    <Chart title="Energy Consumption" html={plots.energy} />
                    <Chart title="ACF (lag 24)"        html={plots.acf} />
                  </div>
                )}
              </div>

              {MODEL_STEPS.every(step => modelStepStatus[step.cardTitle] === "done") && (
                <div className="w-full flex justify-center transition-all duration-500 ease-in-out">
                  <div className="w-full max-w-4xl">
                    <IdleComponent changeState={(name) => setSelectedModel(name)} models={MODEL_STEPS}/>
                  </div>
                </div>
              )}

              <div className="w-full transition-all duration-500 ease-in-out flex flex-col gap-6">
                {modelCharts
                  .filter((c) => c.title.startsWith("Random Forest \u2014"))
                  .map(({ title, modes }, i) => {
                    const buildingId = title.split(" \u2014 ")[1];
                    return (
                      <div key={i} className="bg-slate-50 border border-slate-200 rounded p-4 shadow-sm flex flex-col gap-3">
                        <h4 className="font-bold text-center text-slate-800 text-sm border-b border-slate-300 pb-2">
                          Building {buildingId} Importances
                        </h4>
                        
                        {["global", "seasonal"].map((mode) => {
                          if (!modes[mode]) return null;
                          return (
                            <div key={mode} className="flex flex-col gap-2 mb-2">
                              <h5 className="font-semibold text-center uppercase text-[10px] tracking-wider text-slate-600 bg-slate-200 py-1 rounded">
                                {mode}
                              </h5>
                              <div className="flex flex-col gap-3">
                                {Object.entries(modes[mode]).map(([season, data]) => (
                                  <div key={season} className="flex flex-col gap-2">
                                    <h6 className="text-[10px] font-bold text-slate-500 uppercase">{season}</h6>
                                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-2">
                                      <ImportanceTable 
                                        title="Permutation" 
                                        data={data.pImportance} 
                                        valueKey="mae_rel_increase" 
                                      />
                                      <ImportanceTable 
                                        title="Feature" 
                                        data={data.fImportance} 
                                      />
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}
              </div>
            </div>

            
          )
        )}
      </div>
    </div>
  );
}

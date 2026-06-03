import { BarChart3, Loader2, Play } from "lucide-react";

export default function PipelineHeader({
  loading,
  modelLoading,
  pipelineDone,
  onRunPipeline,
  onRunModels,
}) {
  return (
    <header className="mb-6 flex flex-col gap-4 rounded-lg border border-slate-200 bg-white/95 px-4 py-4 shadow-sm sm:flex-row sm:items-center sm:justify-between sm:px-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Forecasting Pipeline</p>
        <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
          Time-Aware Energy Forecasting
        </h1>
      </div>
      <div className="flex flex-wrap gap-3">
        <button
          onClick={onRunPipeline}
          disabled={loading}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-slate-950 px-4 text-sm font-semibold text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {loading ? "Running" : "Run Pipeline"}
        </button>
        <button
          onClick={onRunModels}
          disabled={!pipelineDone || modelLoading}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-slate-950 px-4 text-sm font-semibold text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {modelLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <BarChart3 className="h-4 w-4" />}
          {modelLoading ? "Training" : "Run Models"}
        </button>
      </div>
    </header>
  );
}

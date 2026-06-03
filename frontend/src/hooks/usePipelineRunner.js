import { useState } from "react";

export function usePipelineRunner(steps) {
  const [stepStatus, setStepStatus] = useState({});
  const [plots, setPlots] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pipelineDone, setPipelineDone] = useState(false);

  async function runStep(endpoint, label) {
    setStepStatus((status) => ({ ...status, [label]: "running" }));

    const res = await fetch(endpoint);
    if (!res.ok) {
      setStepStatus((status) => ({ ...status, [label]: "error" }));
      throw new Error(`${label} failed: ${res.status}`);
    }

    setStepStatus((status) => ({ ...status, [label]: "done" }));
  }

  async function runPipeline() {
    setLoading(true);
    setError(null);
    setPlots(null);
    setStepStatus({});

    try {
      for (const { label, endpoint } of steps) {
        await runStep(endpoint, label);
      }

      const res = await fetch("/api/plot");
      if (!res.ok) {
        throw new Error(`Plot generation failed: ${res.status}`);
      }

      const data = await res.json();
      setPlots(data.response);
      setPipelineDone(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return {
    stepStatus,
    plots,
    loading,
    error,
    pipelineDone,
    runPipeline,
  };
}

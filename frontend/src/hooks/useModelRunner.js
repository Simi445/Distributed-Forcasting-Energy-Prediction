import { useState } from "react";

function addModelResponse(groupedCharts, cardTitle, response) {
  Object.entries(response).forEach(([key, info]) => {
    const [buildId, season, mode] = key.split(",");
    const groupKey = `${cardTitle} — ${buildId}`;
    const usedMode = typeof mode !== "undefined" ? mode : "seasonal";

    groupedCharts[groupKey] ??= {};
    groupedCharts[groupKey][usedMode] ??= {};
    groupedCharts[groupKey][usedMode][season] = {
      html: info.html,
      metrics: info.metrics ?? null,
      pImportance: info.p_importance ?? null,
      fImportance: info.f_importance ?? null,
    };
  });
}

export function useModelRunner(modelSteps) {
  const [modelStepStatus, setModelStepStatus] = useState({});
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [modelCharts, setModelCharts] = useState([]);

  async function runModelPipeline() {
    setModelLoading(true);
    setModelError(null);
    setModelStepStatus({});
    setModelCharts([]);

    const groupedCharts = {};

    try {
      for (const { cardTitle, modes } of modelSteps) {
        setModelStepStatus((status) => ({ ...status, [cardTitle]: "running" }));

        for (const [mode, endpoint] of Object.entries(modes)) {
          const res = await fetch(endpoint);
          if (!res.ok) {
            setModelStepStatus((status) => ({ ...status, [cardTitle]: "error" }));
            throw new Error(`${cardTitle} (${mode}) failed: ${res.status}`);
          }

          const data = await res.json();
          addModelResponse(groupedCharts, cardTitle, data.response);
        }

        setModelStepStatus((status) => ({ ...status, [cardTitle]: "done" }));
      }

      setModelCharts(Object.entries(groupedCharts).map(([title, modes]) => ({ title, modes })));
    } catch (e) {
      setModelError(e.message);
    } finally {
      setModelLoading(false);
    }
  }

  return {
    modelStepStatus,
    modelLoading,
    modelError,
    modelCharts,
    runModelPipeline,
  };
}

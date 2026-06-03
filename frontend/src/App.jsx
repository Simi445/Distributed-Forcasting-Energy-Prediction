import { useState } from "react";
import { DottedSurface } from "@/components/ui/dotted-surface";
import Dashboard from "@/components/Dashboard.jsx";
import ModelDetails from "@/components/models/ModelDetails.jsx";
import ErrorMessage from "@/components/pipeline/ErrorMessage.jsx";
import PipelineHeader from "@/components/pipeline/PipelineHeader.jsx";
import StepStatusList from "@/components/pipeline/StepStatusList.jsx";
import pipelineConfig from "@/config/pipeline.json";
import { useModelRunner } from "@/hooks/useModelRunner.js";
import { usePipelineRunner } from "@/hooks/usePipelineRunner.js";

const { steps: STEPS, modelSteps: MODEL_STEPS, stepStyles } = pipelineConfig;

export default function App() {
  const [selectedModel, setSelectedModel] = useState(null);
  const pipeline = usePipelineRunner(STEPS);
  const models = useModelRunner(MODEL_STEPS);

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-slate-50 text-slate-800">
      <DottedSurface className="z-0 opacity-30" />
      <div className="relative z-10 mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        {!selectedModel && (
          <PipelineHeader
            loading={pipeline.loading}
            modelLoading={models.modelLoading}
            pipelineDone={pipeline.pipelineDone}
            onRunPipeline={pipeline.runPipeline}
            onRunModels={models.runModelPipeline}
          />
        )}

        {!selectedModel && (
          <>
            <StepStatusList steps={STEPS} status={pipeline.stepStatus} stepStyles={stepStyles} />
            <ErrorMessage message={pipeline.error} />
            <StepStatusList
              steps={MODEL_STEPS}
              status={models.modelStepStatus}
              stepStyles={stepStyles}
              getLabel={(step) => step.cardTitle}
            />
            <ErrorMessage message={models.modelError} />
          </>
        )}

        {selectedModel ? (
          <ModelDetails
            selectedModel={selectedModel}
            modelCharts={models.modelCharts}
            onBack={() => setSelectedModel(null)}
          />
        ) : (
          <Dashboard
            plots={pipeline.plots}
            modelStepStatus={models.modelStepStatus}
            modelCharts={models.modelCharts}
            modelSteps={MODEL_STEPS}
            onSelectModel={setSelectedModel}
          />
        )}
      </div>
    </div>
  );
}

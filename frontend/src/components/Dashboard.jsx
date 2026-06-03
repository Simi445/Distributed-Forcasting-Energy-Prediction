import IdleComponent from "@/components/IdleComponent.jsx";
import ModelImportances from "@/components/models/ModelImportances.jsx";
import PipelineCharts from "@/components/pipeline/PipelineCharts.jsx";

export default function Dashboard({
  plots,
  modelStepStatus,
  modelCharts,
  modelSteps,
  onSelectModel,
}) {
  const modelsReady = modelSteps.every((step) => modelStepStatus[step.cardTitle] === "done");
  const hasContent = plots || Object.keys(modelStepStatus).length > 0;

  if (!hasContent) return null;

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
      <div className="w-full transition-all duration-500 ease-in-out">
        {plots && <PipelineCharts plots={plots} />}
      </div>

      {modelsReady && (
        <div className="flex w-full justify-center transition-all duration-500 ease-in-out">
          <div className="w-full max-w-4xl">
            <IdleComponent changeState={onSelectModel} models={modelSteps} />
          </div>
        </div>
      )}

      <ModelImportances modelCharts={modelCharts} />
    </div>
  );
}

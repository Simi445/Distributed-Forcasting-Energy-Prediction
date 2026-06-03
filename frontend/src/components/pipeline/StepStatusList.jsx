import StepIcon from "./StepIcon.jsx";

export default function StepStatusList({ steps, status, stepStyles, getLabel = (step) => step.label }) {
  if (Object.keys(status).length === 0) return null;

  return (
    <div className="mb-4 flex flex-wrap items-center justify-center gap-2">
      {steps.map((step) => {
        const label = getLabel(step);
        const state = status[label] ?? "pending";

        return (
          <span
            key={label}
            className={`flex items-center justify-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium shadow-sm ring-1 ring-inset ring-black/5 ${stepStyles[state]}`}
          >
            <StepIcon state={state} />
            {label}
          </span>
        );
      })}
    </div>
  );
}

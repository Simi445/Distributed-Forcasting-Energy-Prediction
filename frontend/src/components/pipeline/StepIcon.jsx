import { Check, Circle, Loader2, X } from "lucide-react";

export default function StepIcon({ state }) {
  const className = "h-4 w-4 shrink-0";
  const icons = {
    running: <Loader2 className={`${className} animate-spin`} />,
    done: <Check className={className} />,
    error: <X className={className} />,
    pending: <Circle className={className} />,
  };

  return <span className="flex items-center">{icons[state]}</span>;
}

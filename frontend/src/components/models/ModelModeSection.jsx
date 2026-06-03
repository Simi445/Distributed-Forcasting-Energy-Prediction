import ModelSeasonResult from "./ModelSeasonResult.jsx";

const MODE_STYLES = {
  global: "bg-sky-100 text-sky-800",
  seasonal: "bg-amber-100 text-amber-800",
};

const MODE_LABELS = {
  global: "Global Model",
  seasonal: "Seasonal Model",
};

export default function ModelModeSection({ mode, seasons }) {
  if (!seasons) return null;

  return (
    <div className={mode === "global" ? "mb-10" : undefined}>
      <h4 className={`mb-4 rounded-lg p-2 text-center text-sm font-semibold uppercase tracking-wider ${MODE_STYLES[mode]}`}>
        {MODE_LABELS[mode]}
      </h4>
      {Object.entries(seasons).map(([season, data]) => (
        <ModelSeasonResult key={`${mode}-${season}`} season={season} data={data} />
      ))}
    </div>
  );
}

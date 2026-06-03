import ImportanceTable from "./ImportanceTable.jsx";

export default function ModelImportances({ modelCharts, modelName = "Random Forest" }) {
  const charts = modelCharts.filter((chart) => chart.title.startsWith(`${modelName} —`));

  if (charts.length === 0) return null;

  return (
    <div className="flex w-full flex-col gap-6 transition-all duration-500 ease-in-out">
      {charts.map(({ title, modes }) => {
        const buildingId = title.split(" — ")[1];

        return (
          <div key={title} className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <h4 className="border-b border-slate-300 pb-2 text-center text-sm font-bold text-slate-800">
              Building {buildingId} Importances
            </h4>

            {["global", "seasonal"].map((mode) => {
              if (!modes[mode]) return null;

              return (
                <div key={mode} className="mb-2 flex flex-col gap-2">
                  <h5 className="rounded bg-slate-200 py-1 text-center text-[10px] font-semibold uppercase tracking-wider text-slate-600">
                    {mode}
                  </h5>
                  <div className="flex flex-col gap-3">
                    {Object.entries(modes[mode]).map(([season, data]) => (
                      <div key={season} className="flex flex-col gap-2">
                        <h6 className="text-[10px] font-bold uppercase text-slate-500">{season}</h6>
                        <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
                          <ImportanceTable
                            title="Permutation"
                            data={data.pImportance}
                            valueKey="mae_rel_increase"
                          />
                          <ImportanceTable title="Feature" data={data.fImportance} />
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
  );
}

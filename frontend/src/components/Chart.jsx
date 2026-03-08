import { useEffect, useState, useRef } from "react";

export default function Chart({ title, html }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const prevUrl = useRef(null);

  useEffect(() => {
    if (!html) return;
    const centeredHtml = `
      <html><head><style>
        body { margin: 0; padding: 10px; overflow: scroll; }
        .mpld3-figure { display: block; margin: 0 auto; }
      </style></head><body>${html}</body></html>`;
    const blob = new Blob([centeredHtml], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    setBlobUrl(url);
    prevUrl.current = url;
    return () => URL.revokeObjectURL(url);
  }, [html]);

  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-lg font-semibold text-gray-700">{title}</h2>
      {blobUrl ? (
        <iframe
          src={blobUrl}
          title={title}
          className="w-full h-[500px] rounded-lg border border-gray-200 bg-white shadow-sm"
        />
      ) : (
        <div className="h-48 flex items-center justify-center rounded-lg border border-dashed border-gray-300 text-gray-400">
          Waiting for data...
        </div>
      )}
    </div>
  );
}

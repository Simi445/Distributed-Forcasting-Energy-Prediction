import { useEffect, useRef, useState } from "react";

export default function Chart({ title, html }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [frameHeight, setFrameHeight] = useState(540);
  const chartId = useRef(`chart-${Math.random().toString(36).slice(2)}`);

  useEffect(() => {
    function handleMessage(event) {
      const data = event.data;
      if (data?.source !== "licenta-mpld3-chart" || data.id !== chartId.current) {
        return;
      }

      setFrameHeight(Math.min(760, Math.max(420, Math.ceil(data.height))));
    }

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  useEffect(() => {
    if (!html) return;
    setFrameHeight(540);

    const responsiveHtml = `
      <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <style>
            html,
            body {
              margin: 0;
              padding: 0;
              width: 100%;
              min-width: 0;
              overflow: hidden;
              background: transparent;
            }

            html {
              font-family: Arial, Helvetica, sans-serif;
            }

            body {
              box-sizing: border-box;
              min-height: 420px;
            }

            #chart-root {
              width: 100%;
              min-width: 0;
              overflow-x: hidden;
              overflow-y: hidden;
              padding: 8px;
            }

            *,
            *::before,
            *::after {
              box-sizing: inherit;
            }

            .mpld3-figure {
              display: block !important;
              width: 100% !important;
              max-width: 100% !important;
              margin: 0 auto !important;
              overflow: visible !important;
            }

            .mpld3-figure svg {
              display: block !important;
              width: 100% !important;
              max-width: 100% !important;
              height: auto !important;
            }
          </style>
        </head>
        <body>
          <div id="chart-root">${html}</div>
          <script>
            var chartId = "${chartId.current}";
            var lastHeight = 0;

            function makeMpld3Responsive() {
              document.querySelectorAll(".mpld3-figure svg").forEach(function (svg) {
                var width = svg.getAttribute("width");
                var height = svg.getAttribute("height");
                if (width && height && !svg.getAttribute("viewBox")) {
                  svg.setAttribute("viewBox", "0 0 " + parseFloat(width) + " " + parseFloat(height));
                  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
                }
                svg.style.width = "100%";
                svg.style.maxWidth = "100%";
                svg.style.height = "auto";
              });
            }

            function getSvgHeight(svg, rootTop) {
              var rect = svg.getBoundingClientRect();
              var attrWidth = parseFloat(svg.getAttribute("width"));
              var attrHeight = parseFloat(svg.getAttribute("height"));

              if (!Number.isNaN(attrWidth) && attrWidth > 0 && !Number.isNaN(attrHeight) && rect.width > 0) {
                return rect.top - rootTop + rect.width * (attrHeight / attrWidth);
              }

              return rect.bottom - rootTop;
            }

            function postHeight() {
              makeMpld3Responsive();

              var root = document.getElementById("chart-root");
              var height = 0;
              var rootTop = root ? root.getBoundingClientRect().top : 0;
              var svgs = document.querySelectorAll(".mpld3-figure svg");

              svgs.forEach(function (svg) {
                height = Math.max(height, getSvgHeight(svg, rootTop));
              });

              if (!height && root) {
                height = root.getBoundingClientRect().height;
              }

              height = Math.min(760, Math.max(420, Math.ceil(height + 24)));
              if (Math.abs(height - lastHeight) > 2) {
                lastHeight = height;
                window.parent.postMessage({
                  source: "licenta-mpld3-chart",
                  id: chartId,
                  height: height
                }, "*");
              }
            }

            window.addEventListener("load", function () {
              postHeight();
              window.setTimeout(postHeight, 50);
              window.setTimeout(postHeight, 250);
            });
            window.addEventListener("resize", postHeight);

            new MutationObserver(postHeight).observe(document.body, {
              childList: true,
              subtree: true
            });
            postHeight();
          </script>
        </body>
      </html>`;
    const blob = new Blob([responsiveHtml], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    setBlobUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [html]);

  return (
    <section className="flex min-w-0 flex-col gap-3">
      {title && <h2 className="text-base font-semibold text-slate-700">{title}</h2>}
      {blobUrl ? (
        <iframe
          src={blobUrl}
          title={title ?? "Chart"}
          scrolling="no"
          className="block w-full shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm"
          style={{ height: frameHeight, minHeight: frameHeight }}
        />
      ) : (
        <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-slate-400">
          Waiting for data...
        </div>
      )}
    </section>
  );
}

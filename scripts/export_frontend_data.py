#!/usr/bin/env python3
"""Export the data currently queried and displayed by the frontend.

The script mirrors the frontend flow:
1. Run preprocessing endpoints in sequence.
2. Save raw responses for preprocessing, plots, and model endpoints.
3. Extract frontend-visible data into HTML/JSON/CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib import error, request


PREPROCESS_STEPS: List[Tuple[str, str]] = [
    ("load_data", "/load_data"),
    ("merge_data", "/merge_data"),
    ("feature_engineer", "/feature_engineer"),
    ("acf", "/acf"),
    ("final_selection", "/final_selection"),
]

MODEL_ENDPOINTS: List[Tuple[str, str, str]] = [
    ("random_forest", "global", "/rf"),
    ("random_forest", "autumn", "/rf?season=autumn"),
    ("random_forest", "winter", "/rf?season=winter"),
    ("svr", "global", "/svr"),
    ("svr", "autumn", "/svr?season=autumn"),
    ("svr", "winter", "/svr?season=winter"),
    ("xg_boost", "global", "/xg_boost"),
    ("xg_boost", "autumn", "/xg_boost?season=autumn"),
    ("xg_boost", "winter", "/xg_boost?season=winter"),
    ("naive", "autumn", "/naive/autumn"),
    ("naive", "winter", "/naive/winter"),
    ("mean", "autumn", "/mean/autumn"),
    ("mean", "winter", "/mean/winter"),
]


@dataclass
class EndpointResult:
    name: str
    url: str
    payload: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frontend-queried HTML and score data.")
    parser.add_argument(
        "--preprocess-base",
        default="http://localhost:8001",
        help="Base URL for the data-preprocess service.",
    )
    parser.add_argument(
        "--model-base",
        default="http://localhost:8002",
        help="Base URL for the model-training service.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/frontend_data",
        help="Directory where exported files will be written.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def http_get_json(url: str) -> Dict[str, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    try:
        with request.urlopen(req) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanitize_token(value: Any) -> str:
    token = str(value).strip().strip("'\"")
    token = token.replace("/", "_")
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    return token.strip("._") or "value"


def parse_result_key(raw_key: str) -> List[str]:
    key = raw_key.strip()
    if key.startswith("(") and key.endswith(")"):
        key = key[1:-1]
    parts = [part.strip().strip("'\"") for part in key.split(",")]
    return [part for part in parts if part]


def save_acf_outputs(output_dir: Path, acf_payload: Dict[str, Any]) -> None:
    graph_data = acf_payload.get("graph_data", {})
    categories, values = graph_data.get("acf_data", [[], []])
    rows = [
        {"category": category, "acf_value": value}
        for category, value in zip(categories, values)
    ]
    write_json(output_dir / "preprocess" / "acf_data.json", {"rows": rows})
    write_csv(output_dir / "preprocess" / "acf_data.csv", rows, ["category", "acf_value"])


def save_building_outputs(output_dir: Path, buildings_payload: Dict[str, Any]) -> None:
    buildings = buildings_payload.get("response", {})
    rows = [
        {"category": category, "building_id": building_id}
        for category, building_id in buildings.items()
    ]
    write_json(output_dir / "preprocess" / "final_buildings.json", {"rows": rows})
    write_csv(output_dir / "preprocess" / "final_buildings.csv", rows, ["category", "building_id"])


def save_plot_outputs(output_dir: Path, plot_payload: Dict[str, Any]) -> None:
    plots = plot_payload.get("response", {})
    for plot_name, html in plots.items():
        write_text(output_dir / "plots" / f"{sanitize_token(plot_name)}.html", html)


def save_selected_building_data(
    output_dir: Path,
    preprocess_base: str,
    buildings_payload: Dict[str, Any],
) -> None:
    buildings = buildings_payload.get("response", {})
    for category, building_id in buildings.items():
        url = f"{preprocess_base.rstrip('/')}/building/{building_id}"
        payload = http_get_json(url)
        category_token = sanitize_token(category)
        building_token = sanitize_token(building_id)

        write_json(
            output_dir / "raw" / "preprocess" / "buildings" / f"{building_token}.json",
            payload,
        )

        data = payload.get("data", {})
        index = payload.get("index", [])
        columns = list(data.keys())
        rows: List[Dict[str, Any]] = []
        for idx, timestamp in enumerate(index):
            row = {"timestamp": timestamp, "category": category, "building_id": building_id}
            for column in columns:
                values = data.get(column, [])
                row[column] = values[idx] if idx < len(values) else None
            rows.append(row)

        if rows:
            fieldnames = ["timestamp", "category", "building_id"] + columns
            write_csv(
                output_dir / "preprocess" / "buildings" / f"{building_token}_{category_token}.csv",
                rows,
                fieldnames,
            )


def normalize_model_entry(
    model_name: str,
    requested_mode: str,
    raw_key: str,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    parts = parse_result_key(raw_key)
    if len(parts) == 3:
        building_id, season, actual_mode = parts
    elif len(parts) == 2:
        building_id, season = parts
        actual_mode = "seasonal"
    else:
        raise ValueError(f"Unrecognized result key format: {raw_key}")

    return {
        "model_name": model_name,
        "requested_mode": requested_mode,
        "building_id": building_id,
        "season": season,
        "actual_mode": actual_mode,
        "metrics": info.get("metrics"),
        "html": info.get("html"),
        "p_importance": info.get("p_importance"),
        "f_importance": info.get("f_importance"),
    }


def save_model_outputs(output_dir: Path, model_runs: List[EndpointResult]) -> None:
    metrics_rows: List[Dict[str, Any]] = []
    permutation_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []

    for run in model_runs:
        response = run.payload.get("response", {})
        for raw_key, info in response.items():
            normalized = normalize_model_entry(run.name, run.url, raw_key, info)
            model_name = sanitize_token(normalized["model_name"])
            building_id = sanitize_token(normalized["building_id"])
            actual_mode = sanitize_token(normalized["actual_mode"])
            season = sanitize_token(normalized["season"])

            html = normalized.pop("html")
            metrics = normalized.pop("metrics") or {}
            p_importance = normalized.pop("p_importance") or {}
            f_importance = normalized.pop("f_importance") or {}

            if html:
                html_path = (
                    output_dir
                    / "models"
                    / model_name
                    / building_id
                    / actual_mode
                    / f"{season}.html"
                )
                write_text(html_path, html)

            if metrics:
                metrics_rows.append(
                    {
                        "model_name": normalized["model_name"],
                        "requested_endpoint": run.url,
                        "building_id": normalized["building_id"],
                        "season": normalized["season"],
                        "actual_mode": normalized["actual_mode"],
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "nmae": metrics.get("nmae"),
                        "r_2": metrics.get("r_2"),
                    }
                )

            for feature_name, feature_value in f_importance.items():
                feature_rows.append(
                    {
                        "model_name": normalized["model_name"],
                        "requested_endpoint": run.url,
                        "building_id": normalized["building_id"],
                        "season": normalized["season"],
                        "actual_mode": normalized["actual_mode"],
                        "feature_name": feature_name,
                        "importance": feature_value,
                    }
                )

            for group_name, group_scores in p_importance.items():
                row = {
                    "model_name": normalized["model_name"],
                    "requested_endpoint": run.url,
                    "building_id": normalized["building_id"],
                    "season": normalized["season"],
                    "actual_mode": normalized["actual_mode"],
                    "group_name": group_name,
                }
                if isinstance(group_scores, dict):
                    row.update(group_scores)
                else:
                    row["value"] = group_scores
                permutation_rows.append(row)

    if metrics_rows:
        write_json(output_dir / "models" / "metrics.json", {"rows": metrics_rows})
        write_csv(
            output_dir / "models" / "metrics.csv",
            metrics_rows,
            ["model_name", "requested_endpoint", "building_id", "season", "actual_mode", "rmse", "mae", "nmae", "r_2"],
        )

    if feature_rows:
        write_json(output_dir / "models" / "feature_importance.json", {"rows": feature_rows})
        write_csv(
            output_dir / "models" / "feature_importance.csv",
            feature_rows,
            ["model_name", "requested_endpoint", "building_id", "season", "actual_mode", "feature_name", "importance"],
        )

    if permutation_rows:
        fieldnames = sorted({key for row in permutation_rows for key in row.keys()})
        write_json(output_dir / "models" / "permutation_importance.json", {"rows": permutation_rows})
        write_csv(output_dir / "models" / "permutation_importance.csv", permutation_rows, fieldnames)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / timestamp
    ensure_dir(output_dir)
    ensure_dir(output_dir / "raw" / "preprocess")
    ensure_dir(output_dir / "raw" / "models")

    preprocess_results: Dict[str, EndpointResult] = {}
    for name, path in PREPROCESS_STEPS:
        url = f"{args.preprocess_base.rstrip('/')}{path}"
        payload = http_get_json(url)
        preprocess_results[name] = EndpointResult(name=name, url=url, payload=payload)
        write_json(output_dir / "raw" / "preprocess" / f"{name}.json", payload)

    plot_url = f"{args.preprocess_base.rstrip('/')}/plot"
    plot_payload = http_get_json(plot_url)
    write_json(output_dir / "raw" / "preprocess" / "plot.json", plot_payload)

    save_acf_outputs(output_dir, preprocess_results["acf"].payload)
    save_building_outputs(output_dir, preprocess_results["final_selection"].payload)
    save_selected_building_data(output_dir, args.preprocess_base, preprocess_results["final_selection"].payload)
    save_plot_outputs(output_dir, plot_payload)

    model_results: List[EndpointResult] = []
    for model_name, request_mode, path in MODEL_ENDPOINTS:
        url = f"{args.model_base.rstrip('/')}{path}"
        payload = http_get_json(url)
        model_results.append(EndpointResult(name=model_name, url=url, payload=payload))
        raw_name = f"{sanitize_token(model_name)}__{sanitize_token(request_mode)}.json"
        write_json(output_dir / "raw" / "models" / raw_name, payload)

    save_model_outputs(output_dir, model_results)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocess_base": args.preprocess_base,
        "model_base": args.model_base,
        "output_dir": str(output_dir),
        "preprocess_endpoints": [name for name, _ in PREPROCESS_STEPS] + ["plot"],
        "model_endpoints": [
            {"model_name": model_name, "request_mode": request_mode, "path": path}
            for model_name, request_mode, path in MODEL_ENDPOINTS
        ],
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Saved frontend data export to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

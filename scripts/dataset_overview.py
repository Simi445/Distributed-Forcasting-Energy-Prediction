#!/usr/bin/env python3
"""Print presentation-ready facts about the ASHRAE dataset.

The large meter table is processed in chunks so this can run on a laptop
without loading the full CSV into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def pct(value: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return 100.0 * value / total


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def print_section(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    return (
        pd.DataFrame(
            {
                "missing": missing,
                "missing_pct": (missing / len(df) * 100).round(2),
            }
        )
        .query("missing > 0")
        .sort_values(["missing_pct", "missing"], ascending=False)
    )


def iter_train_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        chunksize=chunksize,
        dtype={"building_id": "int32", "meter": "int8", "meter_reading": "float32"},
    )


def summarize_train(path: Path, chunksize: int) -> dict:
    rows = 0
    meter_counts: dict[int, int] = {}
    meter_buildings: dict[int, set[int]] = {}
    min_ts: str | None = None
    max_ts: str | None = None
    zero_rows = 0
    reading_min: float | None = None
    reading_max: float | None = None
    reading_sum = 0.0

    for chunk in iter_train_chunks(path, chunksize):
        rows += len(chunk)

        counts = chunk["meter"].value_counts()
        for meter, count in counts.items():
            meter_counts[int(meter)] = meter_counts.get(int(meter), 0) + int(count)

        for meter, group in chunk.groupby("meter")["building_id"]:
            meter_buildings.setdefault(int(meter), set()).update(group.unique().tolist())

        chunk_min_ts = str(chunk["timestamp"].min())
        chunk_max_ts = str(chunk["timestamp"].max())
        min_ts = chunk_min_ts if min_ts is None else min(min_ts, chunk_min_ts)
        max_ts = chunk_max_ts if max_ts is None else max(max_ts, chunk_max_ts)

        electricity = chunk[chunk["meter"] == 0]
        if not electricity.empty:
            zero_rows += int((electricity["meter_reading"] == 0).sum())
            current_min = float(electricity["meter_reading"].min())
            current_max = float(electricity["meter_reading"].max())
            reading_min = current_min if reading_min is None else min(reading_min, current_min)
            reading_max = current_max if reading_max is None else max(reading_max, current_max)
            reading_sum += float(electricity["meter_reading"].sum())

    electricity_rows = meter_counts.get(0, 0)
    return {
        "rows": rows,
        "meter_counts": meter_counts,
        "meter_buildings": {k: len(v) for k, v in meter_buildings.items()},
        "min_ts": min_ts,
        "max_ts": max_ts,
        "electricity_rows": electricity_rows,
        "electricity_zero_rows": zero_rows,
        "electricity_reading_min": reading_min,
        "electricity_reading_max": reading_max,
        "electricity_reading_mean": reading_sum / electricity_rows if electricity_rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root containing the CSV files.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per chunk for train.csv.")
    args = parser.parse_args()

    root = args.root.resolve()
    train_path = root / "train.csv"
    weather_path = root / "weather_train.csv"
    metadata_path = root / "building_metadata.csv"

    metadata = pd.read_csv(metadata_path)
    weather = pd.read_csv(weather_path)
    train = summarize_train(train_path, args.chunksize)

    print_section("Dataset Overview")
    print("Source: ASHRAE Great Energy Predictor III 2019")
    print("Raw tables:")
    print(f"- train.csv: {format_int(train['rows'])} hourly meter-reading rows")
    print(f"- weather_train.csv: {format_int(len(weather))} hourly weather rows")
    print(f"- building_metadata.csv: {format_int(len(metadata))} building rows")
    print(f"Time span in train.csv: {train['min_ts']} to {train['max_ts']}")

    print_section("Meter Table")
    meter_names = {
        0: "electricity",
        1: "chilled water",
        2: "steam",
        3: "hot water",
    }
    for meter, count in sorted(train["meter_counts"].items()):
        name = meter_names.get(meter, "unknown")
        building_count = train["meter_buildings"].get(meter, 0)
        print(
            f"- meter={meter} ({name}): {format_int(count)} rows, "
            f"{format_int(building_count)} buildings"
        )
    print()
    print("Project target:")
    print(f"- uses only meter=0 electricity rows: {format_int(train['electricity_rows'])} rows")
    print(
        f"- zero electricity readings: {format_int(train['electricity_zero_rows'])} "
        f"({pct(train['electricity_zero_rows'], train['electricity_rows']):.2f}%)"
    )
    print(
        "- electricity reading range: "
        f"{train['electricity_reading_min']:.3f} to {train['electricity_reading_max']:.3f}; "
        f"mean {train['electricity_reading_mean']:.3f}"
    )

    print_section("Building Metadata")
    print(f"Buildings: {metadata['building_id'].nunique():,}")
    print(f"Sites: {metadata['site_id'].nunique():,}")
    print()
    print("Primary-use distribution:")
    for primary_use, count in metadata["primary_use"].value_counts().items():
        print(f"- {primary_use}: {format_int(count)} buildings")
    print()
    print("Metadata missingness:")
    metadata_missing = missing_summary(metadata)
    print(metadata_missing.to_string() if not metadata_missing.empty else "No missing values.")

    print_section("Weather Table")
    print(f"Sites with weather: {weather['site_id'].nunique():,}")
    print(f"Time span in weather_train.csv: {weather['timestamp'].min()} to {weather['timestamp'].max()}")
    print()
    print("Weather columns:")
    print(", ".join(c for c in weather.columns if c not in {"site_id", "timestamp"}))
    print()
    print("Weather missingness:")
    weather_missing = missing_summary(weather)
    print(weather_missing.to_string() if not weather_missing.empty else "No missing values.")

    print_section("How The Modeling Dataset Is Constructed")
    print("1. Start from train.csv, building_metadata.csv, and weather_train.csv.")
    print("2. Keep only electricity readings: train rows where meter == 0.")
    print("3. Drop the meter column because the target is fixed to electricity.")
    print("4. Add floor_missing/year_missing flags, median-fill floor_count and year_built.")
    print("5. One-hot encode primary_use.")
    print("6. Merge train + metadata on building_id.")
    print("7. Merge weather on site_id and timestamp.")
    print("8. For each building, sort hourly by timestamp and resample to an hourly index.")
    print("9. Add calendar features, weather interpolation, lagged load, rolling load, and hour cyclic features.")
    print("10. Select representative buildings using lag-24 autocorrelation plus data-quality filters.")

    print_section("Engineered Feature Groups")
    print("- Target: meter_reading")
    print("- Calendar: hour, dayofweek, month, day, is_weekend, is_business_hour")
    print("- Load history: load_lag_1h, load_lag_24h")
    print("- Rolling load: 24h/168h rolling mean and std")
    print("- Weather: air temperature, dew point, precipitation, pressure, wind, cloud coverage")
    print("- Metadata: site_id, square_feet, year_built, floor_count, missingness flags, primary_use dummies")
    print("- Cyclic/nonlinear: hour_sin, hour_cos, hour_x_temp")


if __name__ == "__main__":
    main()

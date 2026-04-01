from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from data import (
    LOGS_DIR,
    STATISTIC_RESULTS_DIR,
    compute_solar_phase_profile,
    constant_ratio,
    dataset_missing_ratio,
    default_dataset_argument,
    ensure_project_directories,
    list_dataset_files,
    read_dataset,
    relative_path,
    safe_float,
    write_csv,
    zero_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-feature dataset statistics.")
    parser.add_argument(
        "--datasets",
        default=default_dataset_argument(),
        help="Comma-separated dataset names. Defaults to the core analysis datasets.",
    )
    return parser.parse_args()


def generate_dataset_statistics() -> None:
    args = parse_args()
    ensure_project_directories()
    dataset_filter = {item.strip() for item in args.datasets.split(",") if item.strip()}
    statistic_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    solar_phase_rows: list[dict[str, object]] = []

    for file_path in list_dataset_files(dataset_filter):
        bundle = read_dataset(file_path)
        numeric_columns = bundle.numeric_columns
        dataset_summaries.append(
            {
                "dataset_name": bundle.dataset_name,
                "file_path": relative_path(bundle.file_path),
                "freq": bundle.freq,
                "n_rows": len(bundle.frame),
                "n_vars": len(numeric_columns),
                "time_col": bundle.time_col or "",
                "missing_ratio": round(dataset_missing_ratio(bundle.frame, numeric_columns), 6),
            }
        )

        for column in numeric_columns:
            series = pd.to_numeric(bundle.frame[column], errors="coerce")
            statistic_rows.append(
                {
                    "dataset_name": bundle.dataset_name,
                    "file_path": relative_path(bundle.file_path),
                    "freq": bundle.freq,
                    "time_col": bundle.time_col or "",
                    "variable": column,
                    "n_rows": len(series),
                    "n_valid": int(series.notna().sum()),
                    "mean": round(safe_float(series.mean()), 6),
                    "std": round(safe_float(series.std(ddof=0)), 6),
                    "min": round(safe_float(series.min()), 6),
                    "max": round(safe_float(series.max()), 6),
                    "zero_ratio": round(zero_ratio(series), 6),
                    "constant_ratio": round(constant_ratio(series), 6),
                    "missing_ratio": round(safe_float(series.isna().mean()), 6),
                }
            )

        if bundle.dataset_name == "solar_AL":
            solar_phase_profile = compute_solar_phase_profile(bundle.frame, numeric_columns)
            solar_phase_rows.extend(solar_phase_profile.to_dict(orient="records"))

    csv_path = STATISTIC_RESULTS_DIR / "dataset_statistics.csv"
    log_path = LOGS_DIR / "dataset_statistics.log"
    summary_path = STATISTIC_RESULTS_DIR / "dataset_registry_snapshot.csv"
    solar_phase_path = STATISTIC_RESULTS_DIR / "solar_AL_phase_profile.csv"

    write_csv(csv_path, statistic_rows)
    write_csv(summary_path, dataset_summaries)
    if solar_phase_rows:
        write_csv(solar_phase_path, solar_phase_rows)
    write_log(log_path, statistic_rows)
    print(f"Wrote {len(statistic_rows)} feature rows to {csv_path}")


def write_log(log_path, statistic_rows: list[dict[str, object]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Dataset Statistics Log",
        f"Generated at: {generated_at}",
        f"Feature rows: {len(statistic_rows)}",
        "",
    ]
    for row in statistic_rows:
        lines.extend(
            [
                f"[{row['dataset_name']}] {row['variable']}",
                f"  path: {row['file_path']}",
                f"  freq: {row['freq']}",
                f"  mean/std: {row['mean']} / {row['std']}",
                f"  min/max: {row['min']} / {row['max']}",
                f"  zero_ratio: {row['zero_ratio']}",
                f"  constant_ratio: {row['constant_ratio']}",
                f"  missing_ratio: {row['missing_ratio']}",
                "",
            ]
        )
    log_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generate_dataset_statistics()

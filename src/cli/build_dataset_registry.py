from __future__ import annotations

import argparse

from data import (
    STATISTIC_RESULTS_DIR,
    dataset_missing_ratio,
    default_dataset_argument,
    ensure_project_directories,
    list_dataset_files,
    read_dataset,
    relative_path,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the dataset registry.")
    parser.add_argument(
        "--datasets",
        default=default_dataset_argument(),
        help="Comma-separated dataset names. Defaults to the core analysis datasets.",
    )
    return parser.parse_args()


def build_dataset_registry() -> None:
    args = parse_args()
    ensure_project_directories()
    dataset_filter = {item.strip() for item in args.datasets.split(",") if item.strip()}
    rows: list[dict[str, object]] = []

    for file_path in list_dataset_files(dataset_filter):
        bundle = read_dataset(file_path)
        numeric_columns = bundle.numeric_columns
        rows.append(
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

    output_path = STATISTIC_RESULTS_DIR / "dataset_registry.csv"
    write_rows_csv(output_path, rows)
    print(f"Wrote {len(rows)} dataset rows to {output_path}")


if __name__ == "__main__":
    build_dataset_registry()

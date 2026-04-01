from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data import REPORTS_DIR, STATISTIC_RESULTS_DIR, ensure_project_directories, write_markdown
from events.processing import (
    build_event_metadata_markdown,
    build_event_summary_markdown,
    build_final_event_table,
    ensure_event_schema,
    ensure_mapping_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge candidate-level artifacts into event-level metadata.")
    parser.add_argument(
        "--stats-dir",
        default=str(STATISTIC_RESULTS_DIR),
        help="Directory containing candidate statistics and phase annotations.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh2", "ETTm2", "solar_AL", "ETTh1", "ETTm1"],
        help="Datasets to include in the event merge stage.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(STATISTIC_RESULTS_DIR / "final_artifact_events.csv"),
        help="Primary output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        default=str(STATISTIC_RESULTS_DIR / "final_artifact_events.md"),
        help="Primary output markdown path.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def write_alias_outputs(events: pd.DataFrame, out_csv: Path, out_md: Path, metadata_md: str) -> None:
    for path in [
        out_csv.parent / "final_artifact_events.csv",
        out_csv.parent / "final_artifact_metadata.csv",
    ]:
        events.to_csv(path, index=False)

    for path in [
        out_md.parent / "final_artifact_events.md",
        out_md.parent / "final_artifact_metadata.md",
    ]:
        write_markdown(path, metadata_md)


def run_merge(
    stats_dir: Path,
    datasets: list[str],
    out_csv: Path,
    out_md: Path,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_directories()
    stats_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(stats_dir / "artifact_candidates.csv")
    dataset_registry = pd.read_csv(stats_dir / "dataset_registry.csv")
    phase_annotations_path = stats_dir / "solar_AL_phase_annotations.csv"
    phase_annotations = pd.read_csv(phase_annotations_path) if phase_annotations_path.exists() else pd.DataFrame()

    events, mappings = build_final_event_table(
        candidates=candidates,
        phase_annotations=phase_annotations,
        dataset_registry=dataset_registry,
        datasets=datasets,
        show_progress=show_progress,
    )
    events = ensure_event_schema(events)
    mappings = ensure_mapping_schema(mappings)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_csv, index=False)
    mappings.to_csv(stats_dir / "candidate_to_event_map.csv", index=False)

    metadata_md = build_event_metadata_markdown(events, mappings, dataset_registry)
    summary_md = build_event_summary_markdown(events, dataset_registry)
    write_markdown(out_md, metadata_md)
    write_markdown(REPORTS_DIR / "artifact_event_summary.md", summary_md)
    write_alias_outputs(events, out_csv=out_csv, out_md=out_md, metadata_md=metadata_md)

    for dataset_name, group in events.groupby("dataset_name", dropna=False):
        group.to_csv(stats_dir / f"{dataset_name}_event_metadata.csv", index=False)
    return events, mappings


def main() -> None:
    args = parse_args()
    run_merge(
        stats_dir=Path(args.stats_dir),
        datasets=list(args.datasets),
        out_csv=Path(args.out_csv),
        out_md=Path(args.out_md),
        show_progress=not bool(args.no_progress),
    )


if __name__ == "__main__":
    main()

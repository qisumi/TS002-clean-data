from __future__ import annotations

import argparse

import pandas as pd

from data import (
    build_solar_phase_annotations,
    compute_solar_phase_profile,
    default_dataset_argument,
    ensure_project_directories,
    list_dataset_files,
    read_dataset,
    STATISTIC_RESULTS_DIR,
    write_csv,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate review aids from artifact candidates.")
    parser.add_argument(
        "--datasets",
        default=default_dataset_argument(),
        help="Comma-separated dataset names. Defaults to the core analysis datasets.",
    )
    return parser.parse_args()


def review_artifacts() -> None:
    args = parse_args()
    ensure_project_directories()
    focus_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    candidate_path = STATISTIC_RESULTS_DIR / "artifact_candidates.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Artifact candidates not found: {candidate_path}")

    candidates = pd.read_csv(candidate_path)
    bundles = load_selected_bundles(focus_datasets)

    for dataset_name in focus_datasets:
        flagged = (
            candidates[candidates["dataset_name"] == dataset_name]
            .sort_values(["score", "length"], ascending=[False, False])
            .reset_index(drop=True)
        )
        output_path = STATISTIC_RESULTS_DIR / f"{dataset_name}_flagged.csv"
        flagged.to_csv(output_path, index=False)

    if "solar_AL" in bundles:
        solar_bundle = bundles["solar_AL"]
        phase_profile = compute_solar_phase_profile(solar_bundle.frame, solar_bundle.numeric_columns)
        phase_annotations = build_solar_phase_annotations(solar_bundle.frame, solar_bundle.numeric_columns)
        write_csv(
            STATISTIC_RESULTS_DIR / "solar_AL_phase_profile.csv",
            phase_profile.to_dict(orient="records"),
            fieldnames=list(phase_profile.columns),
        )
        write_csv(
            STATISTIC_RESULTS_DIR / "solar_AL_phase_annotations.csv",
            phase_annotations.to_dict(orient="records"),
            fieldnames=list(phase_annotations.columns),
        )

    write_markdown(STATISTIC_RESULTS_DIR / "artifact_taxonomy.md", taxonomy_markdown())
    write_markdown(
        STATISTIC_RESULTS_DIR / "manual_review_notes.md",
        build_review_template(candidates, focus_datasets),
    )
    print("Wrote dataset-specific flagged files, taxonomy, and manual review notes.")


def load_selected_bundles(focus_datasets: list[str]) -> dict[str, object]:
    selected = set(focus_datasets)
    bundles: dict[str, object] = {}
    for file_path in list_dataset_files(selected):
        bundle = read_dataset(file_path)
        bundles[bundle.dataset_name] = bundle
    return bundles


def taxonomy_markdown() -> str:
    return """# Artifact Taxonomy

## Index Convention

- `start_idx` and `end_idx` are inclusive row indices.
- `length = end_idx - start_idx + 1`.

## Classes

### `zero_block`
- Consecutive zero-valued observations longer than a dataset-specific threshold.
- Often indicates corrupted missing-value handling or valid inactive periods that require review.

### `flat_run`
- Consecutive repeated values with negligible change.
- Useful for spotting frozen sensors or implausibly stable measurements.

### `near_constant_segment`
- Low-variance regions identified by rolling standard deviation.
- Captures stable regimes that may still be exploitable shortcuts.

### `suspicious_repetition`
- Consecutive repeated windows with nearly identical values.
- Useful for detecting copied chunks or mechanical replay patterns.

## Solar Phase Labels

For `solar_AL`, the review stage also emits:

- `solar_AL_phase_profile.csv`
- `solar_AL_phase_annotations.csv`

These files provide weak labels for:

- `night_zero_band`
- `phase_transition_band`
- `active_period`
"""


def build_review_template(candidates: pd.DataFrame, focus_datasets: list[str]) -> str:
    lines = [
        "# Manual Review Notes",
        "",
        "Use this file to separate true artifacts from semantically valid but exploitable structures.",
        "",
    ]

    for dataset_name in focus_datasets:
        subset = candidates[candidates["dataset_name"] == dataset_name]
        counts = subset["artifact_type"].value_counts().to_dict()
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                f"- Total candidates: {len(subset)}",
                f"- Type counts: {counts}",
                "- Review criteria:",
                "  - corrupted artifact",
                "  - valid-but-exploitable pattern",
                "  - normal structure / false positive",
            ]
        )
        if dataset_name == "solar_AL":
            lines.extend(
                [
                    "- Solar hints:",
                    "  - cross-check `solar_AL_phase_profile.csv` for `night`, `transition`, and `active` phase bands",
                    "  - treat night-time zero bands as valid-but-exploitable before treating them as corrupted",
                ]
            )
        lines.extend(
            [
                "",
                "| variable | start_idx | end_idx | artifact_type | decision | notes |",
                "| --- | ---: | ---: | --- | --- | --- |",
                "",
            ]
        )

    return "\n".join(lines)


if __name__ == "__main__":
    review_artifacts()

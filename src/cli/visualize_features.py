from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from data import (
    SOLAR_ACTIVE_ZERO_THRESHOLD,
    SOLAR_NIGHT_ZERO_THRESHOLD,
    ETT_DATASETS,
    FIGURES_DIR,
    STATISTIC_RESULTS_DIR,
    compute_solar_phase_profile,
    default_dataset_argument,
    ensure_project_directories,
    list_dataset_files,
    read_dataset,
    row_zero_ratio,
)

FloatArray = npt.NDArray[np.float64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature-level plots for selected datasets.")
    parser.add_argument("--datasets", default=default_dataset_argument(), help="Comma-separated dataset names.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Maximum variables per dataset. Use 0 or a negative value for all variables.",
    )
    parser.add_argument(
        "--random-features",
        action="store_true",
        help="Randomly sample features when --max-features is positive.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base random seed used when --random-features is enabled.",
    )
    parser.add_argument("--max-segments", type=int, default=8, help="Maximum segment figures per dataset.")
    parser.add_argument("--rolling-window", type=int, default=96, help="Window size for rolling statistics.")
    parser.add_argument(
        "--aggregate-points",
        type=int,
        default=2048,
        help="Sample points used by ETT aggregate plots.",
    )
    return parser.parse_args()


def visualize_features() -> None:
    args = parse_args()
    ensure_project_directories()
    dataset_filter = {item.strip() for item in args.datasets.split(",") if item.strip()}
    candidates = load_artifact_candidates()
    ett_bundles: dict[str, tuple[pd.DataFrame, list[str], str]] = {}

    for file_path in list_dataset_files(dataset_filter):
        bundle = read_dataset(file_path)

        dataset_candidates = filter_artifact_candidates(candidates, bundle.dataset_name)
        selected_columns = select_columns(
            bundle.numeric_columns,
            dataset_candidates,
            args.max_features,
            random_features=args.random_features,
            random_seed=args.random_seed + dataset_seed_offset(bundle.dataset_name),
        )
        if not selected_columns:
            continue

        feature_dir = FIGURES_DIR / "feature_plots" / bundle.dataset_name
        segment_dir = FIGURES_DIR / "artifact_segments" / bundle.dataset_name
        overview_dir = FIGURES_DIR / "dataset_overview" / bundle.dataset_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        segment_dir.mkdir(parents=True, exist_ok=True)
        overview_dir.mkdir(parents=True, exist_ok=True)

        save_dataset_overview(bundle.frame, bundle.dataset_name, selected_columns, overview_dir / "overview.png")
        if bundle.dataset_name == "solar_AL":
            save_solar_phase_views(bundle.frame, bundle.numeric_columns, overview_dir)
        for column in selected_columns:
            series = get_numeric_series(bundle.frame, column)
            plot_full_series(series, column, feature_dir / f"{column}_full.png")
            plot_rolling_stats(series, column, args.rolling_window, feature_dir / f"{column}_rolling.png")
            plot_distribution(series, column, feature_dir / f"{column}_distribution.png")
            plot_diff(series, column, feature_dir / f"{column}_diff.png")
            plot_with_artifacts(series, column, dataset_candidates, feature_dir / f"{column}_overlay.png")

        save_artifact_segments(bundle.frame, dataset_candidates, selected_columns, segment_dir, args.max_segments)
        if bundle.dataset_name in ETT_DATASETS:
            ett_bundles[bundle.dataset_name] = (bundle.frame.copy(), list(selected_columns), bundle.freq)

    save_ett_aggregate_plots(ett_bundles, args.aggregate_points)


def load_artifact_candidates() -> pd.DataFrame:
    candidate_path = STATISTIC_RESULTS_DIR / "artifact_candidates.csv"
    if not candidate_path.exists():
        return pd.DataFrame(
            columns=pd.Index(["dataset_name", "variable", "start_idx", "end_idx", "artifact_type", "score", "length"])
        )
    return pd.read_csv(candidate_path)


def filter_artifact_candidates(candidates: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    dataset_names = cast(pd.Series, candidates.loc[:, "dataset_name"])
    return cast(pd.DataFrame, candidates.loc[dataset_names == dataset_name].copy())


def select_columns(
    numeric_columns: list[str],
    dataset_candidates: pd.DataFrame,
    max_features: int,
    random_features: bool,
    random_seed: int,
) -> list[str]:
    if max_features <= 0:
        return list(numeric_columns)
    if random_features:
        sample_size = min(max_features, len(numeric_columns))
        if sample_size <= 0:
            return []
        rng = np.random.default_rng(random_seed)
        selected_indices = rng.choice(len(numeric_columns), size=sample_size, replace=False)
        return [numeric_columns[int(index)] for index in selected_indices.tolist()]
    if dataset_candidates.empty:
        return numeric_columns[:max_features]

    variable_counts = cast(pd.Series, cast(pd.Series, dataset_candidates.loc[:, "variable"]).value_counts())
    ordered = [str(column) for column in variable_counts.index.tolist() if str(column) in numeric_columns]
    for column in numeric_columns:
        if column not in ordered:
            ordered.append(column)
    return ordered[:max_features]


def dataset_seed_offset(dataset_name: str) -> int:
    return sum(ord(character) for character in dataset_name)


def save_dataset_overview(frame: pd.DataFrame, dataset_name: str, columns: list[str], save_path: Path) -> None:
    fig, axes = plt.subplots(len(columns), 1, figsize=(14, 3 * len(columns)), sharex=True)
    axes_list = np.atleast_1d(axes).tolist()
    for axis, column in zip(axes_list, columns):
        axis.plot(series_to_numpy(get_numeric_series(frame, column)), linewidth=0.8)
        axis.set_title(column)
        axis.grid(alpha=0.2)
    fig.suptitle(f"{dataset_name} overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_solar_phase_views(frame: pd.DataFrame, numeric_columns: list[str], overview_dir: Path) -> None:
    row_zero = row_zero_ratio(frame, numeric_columns)
    phase_profile = compute_solar_phase_profile(frame, numeric_columns)
    plot_row_zero_ratio(row_zero, overview_dir / "row_zero_ratio.png")
    plot_solar_phase_profile(phase_profile, overview_dir / "phase_profile.png")


def plot_full_series(series: pd.Series, column: str, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(series_to_numpy(series), linewidth=0.8)
    axis.set_title(f"{column} - Full Series")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_row_zero_ratio(row_zero: pd.Series, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(series_to_numpy(row_zero), linewidth=0.8, color="tab:blue")
    axis.axhline(SOLAR_NIGHT_ZERO_THRESHOLD, linestyle="--", linewidth=1.0, color="tab:red", label="night threshold")
    axis.axhline(
        SOLAR_ACTIVE_ZERO_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        color="tab:green",
        label="active threshold",
    )
    axis.set_title("solar_AL - Row Zero Ratio")
    axis.set_xlabel("Row index")
    axis.set_ylabel("Zero ratio across variables")
    axis.set_ylim(-0.02, 1.02)
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_solar_phase_profile(phase_profile: pd.DataFrame, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(
        phase_profile["phase_idx"].tolist(),
        phase_profile["row_zero_ratio_mean"].tolist(),
        linewidth=1.2,
        color="tab:orange",
        label="mean row zero ratio",
    )
    for phase_group, color in [("night", "tab:red"), ("transition", "tab:gray"), ("active", "tab:green")]:
        subset = phase_profile[phase_profile["phase_group"] == phase_group]
        if subset.empty:
            continue
        axis.scatter(
            subset["phase_idx"].tolist(),
            subset["row_zero_ratio_mean"].tolist(),
            s=18,
            color=color,
            label=phase_group,
        )
    axis.axhline(SOLAR_NIGHT_ZERO_THRESHOLD, linestyle="--", linewidth=1.0, color="tab:red")
    axis.axhline(SOLAR_ACTIVE_ZERO_THRESHOLD, linestyle="--", linewidth=1.0, color="tab:green")
    axis.set_title("solar_AL - Phase Zero Profile")
    axis.set_xlabel("Phase index (10-minute slots)")
    axis.set_ylabel("Mean row zero ratio")
    axis.set_xlim(0, max(phase_profile["phase_idx"].tolist()) if not phase_profile.empty else 1)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_rolling_stats(series: pd.Series, column: str, window: int, save_path: Path) -> None:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(series_to_numpy(series), alpha=0.25, linewidth=0.7, label="raw")
    axis.plot(series_to_numpy(pd.Series(rolling_mean)), linewidth=1.0, label="rolling_mean")
    axis.plot(series_to_numpy(pd.Series(rolling_std)), linewidth=1.0, label="rolling_std")
    axis.set_title(f"{column} - Rolling Stats ({window})")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_distribution(series: pd.Series, column: str, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4))
    values = series.dropna().tolist()
    if not values:
        plt.close(fig)
        return
    sns.histplot(x=values, bins=100, kde=True, ax=axis)
    axis.set_title(f"{column} - Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_diff(series: pd.Series, column: str, save_path: Path) -> None:
    diff = series.diff()
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(series_to_numpy(pd.Series(diff)), linewidth=0.8)
    axis.set_title(f"{column} - First Difference")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_with_artifacts(series: pd.Series, column: str, artifacts: pd.DataFrame, save_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(series_to_numpy(series), linewidth=0.8, label="signal")
    variables = cast(pd.Series, artifacts.loc[:, "variable"])
    subset = cast(pd.DataFrame, artifacts.loc[variables == column, ["start_idx", "end_idx"]])
    for row in subset.to_dict(orient="records"):
        axis.axvspan(int(row["start_idx"]), int(row["end_idx"]), alpha=0.2, color="tab:red")
    axis.set_title(f"{column} - Artifact Overlay")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_ett_aggregate_plots(
    ett_bundles: dict[str, tuple[pd.DataFrame, list[str], str]],
    sample_points: int,
) -> None:
    if not ett_bundles:
        return

    aggregate_root = FIGURES_DIR / "feature_plots" / "aggregated"
    by_feature_dir = aggregate_root / "by_feature"
    by_feature_dir.mkdir(parents=True, exist_ok=True)

    selected_by_dataset = {dataset_name: columns for dataset_name, (_, columns, _) in ett_bundles.items()}
    shared_columns = find_shared_columns(selected_by_dataset)

    for feature in shared_columns:
        plot_ett_feature_aggregate(feature, ett_bundles, sample_points, by_feature_dir / f"{feature}_across_datasets.png")


def plot_ett_feature_aggregate(
    feature: str,
    ett_bundles: dict[str, tuple[pd.DataFrame, list[str], str]],
    sample_points: int,
    save_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(14, 5))
    plotted = 0
    for dataset_name in ETT_DATASETS:
        bundle = ett_bundles.get(dataset_name)
        if bundle is None:
            continue
        frame, selected_columns, freq = bundle
        if feature not in selected_columns:
            continue
        prepared = prepare_aggregate_series(get_numeric_series(frame, feature), sample_points)
        if prepared is None:
            continue
        progress, standardized = prepared
        axis.plot(progress, standardized, linewidth=1.1, alpha=0.9, label=f"{dataset_name} ({freq})")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    axis.set_title(f"{feature} aggregated across ETT datasets")
    axis.set_xlabel("Relative progression")
    axis.set_ylabel("Standardized value")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def prepare_aggregate_series(series: pd.Series, sample_points: int) -> tuple[FloatArray, FloatArray] | None:
    values = series_to_numpy(series)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return None

    n_points = max(2, min(max(sample_points, 2), len(values)))
    source_x = np.linspace(0.0, 1.0, num=len(values))
    target_x = np.linspace(0.0, 1.0, num=n_points)
    finite_x = source_x[finite_mask]
    finite_values = values[finite_mask]

    if finite_values.size == 1:
        sampled = np.full(n_points, finite_values[0], dtype=np.float64)
    else:
        sampled = np.interp(target_x, finite_x, finite_values)

    mean = float(np.mean(sampled))
    std = float(np.std(sampled))
    if std <= 1e-12:
        standardized = sampled - mean
    else:
        standardized = (sampled - mean) / std
    return cast(FloatArray, target_x), cast(FloatArray, standardized)


def find_shared_columns(selected_by_dataset: dict[str, list[str]]) -> list[str]:
    shared_columns: set[str] | None = None
    for columns in selected_by_dataset.values():
        if shared_columns is None:
            shared_columns = set(columns)
        else:
            shared_columns &= set(columns)
    return sorted(shared_columns or set())


def get_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.Series(pd.to_numeric(frame.loc[:, column], errors="coerce")))


def series_to_numpy(series: pd.Series) -> FloatArray:
    numeric = pd.to_numeric(series, errors="coerce")
    return cast(FloatArray, np.asarray(numeric, dtype=np.float64))


def save_artifact_segments(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_columns: list[str],
    segment_dir: Path,
    max_segments: int,
) -> None:
    variables = cast(pd.Series, candidates.loc[:, "variable"])
    focused = cast(pd.DataFrame, candidates.loc[variables.isin(selected_columns)].copy())
    if focused.empty:
        return
    focused = focused.sort_values(by=["score", "length"], ascending=[False, False]).head(max_segments)

    rows = focused.loc[:, ["variable", "length", "start_idx", "end_idx", "artifact_type"]].to_dict(orient="records")
    for index, row in enumerate(rows, start=1):
        variable = str(row["variable"])
        series = get_numeric_series(frame, variable)
        padding = max(int(row["length"]), 24)
        start = max(int(row["start_idx"]) - padding, 0)
        end = min(int(row["end_idx"]) + padding + 1, len(series))
        fig, axis = plt.subplots(figsize=(12, 4))
        x_values = list(range(start, end))
        axis.plot(x_values, series_to_numpy(series.iloc[start:end]), linewidth=0.9)
        axis.axvspan(int(row["start_idx"]), int(row["end_idx"]), alpha=0.25, color="tab:red")
        axis.set_title(f"{variable} - {row['artifact_type']} [{row['start_idx']}, {row['end_idx']}]")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(segment_dir / f"{variable}_segment_{index:03d}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    visualize_features()

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import ROOT_DIR, STATISTIC_RESULTS_DIR, ensure_project_directories, json_dumps, write_markdown
from views import load_view_spec, parse_artifact_ids


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [005/views] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset-specific evaluation views from window scores.")
    parser.add_argument("--spec", default=str(Path("configs") / "view_specs.yaml"))
    parser.add_argument("--scores-dir", default=str(STATISTIC_RESULTS_DIR / "window_scores"))
    parser.add_argument("--out-dir", default=str(STATISTIC_RESULTS_DIR / "window_views"))
    parser.add_argument("--events", default=str(STATISTIC_RESULTS_DIR / "final_artifact_events.csv"))
    parser.add_argument("--manifest-out", default=str(STATISTIC_RESULTS_DIR / "eval_view_manifest.csv"))
    parser.add_argument("--report-out", default=str(ROOT_DIR / "reports" / "eval_view_design.md"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--lookbacks", nargs="+", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    return parser.parse_args()


SCORE_FILE_RE = re.compile(r"^(?P<dataset>.+)_L(?P<lookback>\d+)_H(?P<horizon>\d+)\.csv$")


def parse_score_filename(path: Path) -> tuple[str, int, int] | None:
    match = SCORE_FILE_RE.match(path.name)
    if match is None:
        return None
    return (
        str(match.group("dataset")),
        int(match.group("lookback")),
        int(match.group("horizon")),
    )


def score_path_matches(
    path: Path,
    datasets: set[str] | None,
    lookbacks: set[int] | None,
    horizons: set[int] | None,
) -> bool:
    parsed = parse_score_filename(path)
    if parsed is None:
        return False
    dataset_name, lookback, horizon = parsed
    if datasets is not None and dataset_name not in datasets:
        return False
    if lookbacks is not None and int(lookback) not in lookbacks:
        return False
    if horizons is not None and int(horizon) not in horizons:
        return False
    return True


def decide_anchor_clean(row: pd.Series, cfg: dict[str, Any]) -> bool:
    if float(row["target_contam_score"]) > float(cfg.get("max_target_contam", 0.0)):
        return False
    if float(row["input_contam_score"]) > float(cfg.get("max_input_contam", 0.0)):
        return False

    forbidden = set(cfg.get("forbid_target_validity", []))
    if "corrupted" in forbidden and int(row.get("has_corrupted_target", 0)) == 1:
        return False
    if "suspicious" in forbidden and int(row.get("has_suspicious_target", 0)) == 1:
        return False
    if bool(cfg.get("veto_multivar_severe", False)) and int(row.get("has_multivar_severe_target", 0)) == 1:
        return False
    if bool(cfg.get("veto_ot_severe", False)) and int(row.get("has_ot_severe_target", 0)) == 1:
        return False

    if str(row["dataset_name"]) == "solar_AL":
        if str(row.get("dominant_phase_target", "NA")) != str(cfg.get("require_target_dominant_phase", "active")):
            return False
        if float(row.get("phase_share_target_active", 0.0)) < float(cfg.get("min_target_active_share", 0.0)):
            return False
        if float(row.get("phase_share_input_night", 0.0)) > float(cfg.get("max_input_night_share", 1.0)):
            return False
        if bool(cfg.get("forbid_target_suspicious", False)) and int(row.get("has_suspicious_target", 0)) == 1:
            return False
    return True


def decide_conservative_clean(row: pd.Series, cfg: dict[str, Any]) -> bool:
    if float(row["target_contam_score"]) > float(cfg.get("max_target_contam", 1.0)):
        return False
    if float(row["input_contam_score"]) > float(cfg.get("max_input_contam", 1.0)):
        return False

    forbidden = set(cfg.get("forbid_target_validity", []))
    if "corrupted" in forbidden and int(row.get("has_corrupted_target", 0)) == 1:
        return False

    if str(row["dataset_name"]) == "solar_AL":
        if float(row.get("phase_share_target_night", 0.0)) > float(cfg.get("max_target_night_share", 1.0)):
            return False
        daylike = float(row.get("phase_share_target_active", 0.0)) + float(row.get("phase_share_target_transition", 0.0))
        if daylike < float(cfg.get("min_target_daylike_share", 0.0)):
            return False
    return True


def decide_intervened(row: pd.Series, cfg: dict[str, Any]) -> bool:
    if bool(cfg.get("forbid_unrecoverable_target", False)) and int(row.get("has_unrecoverable_target", 0)) == 1:
        return False
    return True


def _valid_phase_name(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "na", "none"}


def _deterministic_take(pdf: pd.DataFrame, n_take: int) -> list[int]:
    ordered = pdf.sort_values(["target_start", "window_id"]).reset_index()
    if n_take >= len(ordered):
        return ordered["index"].astype(int).tolist()
    picked = np.linspace(0, len(ordered) - 1, num=n_take, dtype=int)
    picked = np.unique(picked)
    return ordered.iloc[picked]["index"].astype(int).tolist()


def build_phase_balance_weights(
    df: pd.DataFrame,
    phase_col: str,
    group_cols: list[str],
    weight_clip: tuple[float, float],
) -> np.ndarray:
    weights = np.ones(len(df), dtype=float)
    lower, upper = weight_clip
    for _, gdf in df.groupby(group_cols, dropna=False):
        valid = gdf[gdf[phase_col].apply(_valid_phase_name)]
        if valid.empty:
            continue
        counts = valid[phase_col].astype(str).value_counts()
        if counts.empty or len(counts) < 2:
            continue
        target_n = int(counts.max())
        for phase_name, pdf in valid.groupby(phase_col, dropna=False):
            phase_key = str(phase_name)
            if not _valid_phase_name(phase_key):
                continue
            phase_n = int(counts.get(phase_key, len(pdf)))
            weight = float(np.clip(target_n / max(phase_n, 1), lower, upper))
            weights[pdf.index.to_numpy()] = weight
    return weights


def build_phase_balanced_subset(
    df: pd.DataFrame,
    phase_col: str,
    group_cols: list[str],
    min_phases: int,
    min_per_phase: int,
) -> np.ndarray:
    kept = np.zeros(len(df), dtype=np.int64)
    for _, gdf in df.groupby(group_cols, dropna=False):
        valid = gdf[gdf[phase_col].apply(_valid_phase_name)]
        if valid.empty:
            continue
        counts = valid[phase_col].astype(str).value_counts()
        if len(counts) < min_phases:
            continue
        n_take = int(counts.min())
        if n_take < min_per_phase:
            continue
        for _, pdf in valid.groupby(phase_col, dropna=False):
            chosen = _deterministic_take(pdf, n_take)
            kept[np.asarray(chosen, dtype=int)] = 1
    return kept


def build_intervention_recipe(row: pd.Series, events_lookup: dict[str, dict[str, Any]]) -> str:
    ops: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()

    input_ids = parse_artifact_ids(row.get("artifact_ids_input"))
    target_ids = parse_artifact_ids(row.get("artifact_ids_target"))
    input_start = int(row["input_start"])
    input_end = int(row["input_end"])
    target_start = int(row["target_start"])
    target_end = int(row["target_end"])

    for artifact_id in input_ids:
        event = events_lookup.get(artifact_id)
        if event is None:
            continue
        start_idx = max(int(event["start_idx"]), input_start)
        end_idx = min(int(event["end_idx"]), input_end)
        if start_idx > end_idx:
            continue

        op = ""
        reason = str(event["artifact_group"])
        if str(row["dataset_name"]) == "solar_AL":
            if reason == "night_zero_band":
                op = "mask_phase_night"
            elif "transition" in reason:
                op = "mask_phase_transition"
            elif reason.startswith("active_suspicious"):
                op = "mask_span"
        else:
            if str(event["recoverability"]) == "repairable":
                op = "local_trend_interp_span"
            elif str(event["recoverability"]) in {"mask_only", "unrecoverable"}:
                op = "mask_span"

        if not op:
            continue
        key = (op, start_idx, end_idx, artifact_id)
        if key in seen:
            continue
        seen.add(key)
        ops.append(
            {
                "op": op,
                "start": start_idx,
                "end": end_idx,
                "artifact_id": artifact_id,
                "reason": reason,
            }
        )

    target_drop = False
    for artifact_id in target_ids:
        event = events_lookup.get(artifact_id)
        if event is None:
            continue
        if str(event["recoverability"]) in {"mask_only", "unrecoverable"}:
            target_drop = True
            reason = str(event["artifact_group"])
            key = ("drop_window", target_start, target_end, artifact_id)
            if key not in seen:
                seen.add(key)
                ops.append(
                    {
                        "op": "drop_window",
                        "start": target_start,
                        "end": target_end,
                        "artifact_id": artifact_id,
                        "reason": reason,
                    }
                )
    if not target_drop and int(row.get("has_unrecoverable_target", 0)) == 1:
        ops.append(
            {
                "op": "drop_window",
                "start": target_start,
                "end": target_end,
                "artifact_id": "",
                "reason": "unrecoverable_target_overlap",
            }
        )
    return json_dumps(ops)


def add_dataset_specific_views(df: pd.DataFrame, spec: dict[str, Any], events_lookup: dict[str, dict[str, Any]]) -> pd.DataFrame:
    dataset_name = str(df["dataset_name"].iloc[0])
    ds_cfg = spec[dataset_name]

    out = df.copy()
    out["is_raw_view"] = 1
    out["is_anchor_clean_view"] = out.apply(lambda row: int(decide_anchor_clean(row, ds_cfg["anchor_clean"])), axis=1)
    out["is_conservative_clean_view"] = out.apply(
        lambda row: int(decide_conservative_clean(row, ds_cfg["conservative_clean"])),
        axis=1,
    )
    out["is_intervened_view"] = out.apply(lambda row: int(decide_intervened(row, ds_cfg.get("intervened", {}))), axis=1)
    out["is_group_controlled_view"] = 1
    out["is_phase_balanced_view"] = 0
    out["is_active_only_view"] = 0
    out["is_daytime_only_view"] = 0
    out["phase_balance_weight"] = 1.0

    if dataset_name == "solar_AL":
        phase_cfg = ds_cfg.get("phase_balanced", {})
        phase_col = str(phase_cfg.get("stratify_by", "dominant_phase_target"))
        balance_group_cols = ["dataset_name", "split_name", "lookback", "horizon"]
        weight_clip_cfg = phase_cfg.get("weight_clip", (0.5, 3.0))
        if isinstance(weight_clip_cfg, list):
            weight_clip = (float(weight_clip_cfg[0]), float(weight_clip_cfg[1]))
        else:
            weight_clip = tuple(weight_clip_cfg)
        out["phase_balance_weight"] = build_phase_balance_weights(
            out,
            phase_col=phase_col,
            group_cols=balance_group_cols,
            weight_clip=(float(weight_clip[0]), float(weight_clip[1])),
        )
        out["is_phase_balanced_view"] = build_phase_balanced_subset(
            out,
            phase_col=phase_col,
            group_cols=balance_group_cols,
            min_phases=int(phase_cfg.get("min_phases", 2)),
            min_per_phase=int(phase_cfg.get("min_per_phase", 100)),
        )
        active_cfg = ds_cfg.get("active_only", {})
        daytime_cfg = ds_cfg.get("daytime_only", {})
        out["is_active_only_view"] = (
            (out["phase_share_target_active"] >= float(active_cfg.get("min_target_active_share", 0.95)))
            & (out["phase_share_input_night"] <= float(active_cfg.get("max_input_night_share", 0.10)))
        ).astype(int)
        out["is_daytime_only_view"] = (
            (out["phase_share_target_night"] <= float(daytime_cfg.get("max_target_night_share", 0.0)))
            & (out["phase_share_input_night"] <= float(daytime_cfg.get("max_input_night_share", 0.05)))
        ).astype(int)

    out["view_status"] = "ok"
    out["primary_group_key"] = np.where(
        out["dataset_name"].eq("solar_AL"),
        out["dominant_phase_target"].astype(str),
        out["artifact_group_major"].astype(str),
    )
    out["drop_reason"] = np.where(out["is_intervened_view"].eq(0), "unrecoverable_target_overlap", "")
    out["intervention_recipe"] = out.apply(lambda row: build_intervention_recipe(row, events_lookup), axis=1)
    out["notes"] = np.where(
        out["dataset_name"].eq("solar_AL"),
        "phase-aware views; night zeros are not repaired",
        "",
    )
    return out


def apply_support_fallback(all_views: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    support_cfg = spec["defaults"]["support"]
    out = all_views.copy()
    summary = (
        out[out["split_name"] == "train"]
        .groupby(["dataset_name", "lookback", "horizon"], dropna=False)
        .agg(
            raw_train_windows=("is_raw_view", "sum"),
            anchor_train_windows=("is_anchor_clean_view", "sum"),
            conservative_train_windows=("is_conservative_clean_view", "sum"),
        )
        .reset_index()
    )

    for row in summary.itertuples(index=False):
        anchor_min = max(
            int(support_cfg["min_anchor_train_windows"]),
            int(np.ceil(float(support_cfg["min_anchor_train_ratio"]) * int(row.raw_train_windows))),
        )
        conservative_min = max(
            int(support_cfg["min_conservative_train_windows"]),
            int(np.ceil(float(support_cfg["min_conservative_train_ratio"]) * int(row.raw_train_windows))),
        )
        mask = (
            (out["dataset_name"] == row.dataset_name)
            & (out["lookback"] == int(row.lookback))
            & (out["horizon"] == int(row.horizon))
        )
        if int(row.anchor_train_windows) < anchor_min:
            out.loc[mask, "view_status"] = "anchor_only"
        if int(row.conservative_train_windows) < conservative_min:
            fallback_key = (
                "fallback_to_intervention"
                if bool(spec[str(row.dataset_name)]["conservative_clean"].get("fallback_to_intervention_if_support_low", False))
                else "intervention_primary"
            )
            out.loc[mask, "view_status"] = fallback_key
    return out


def first_intervention_type(recipe: str) -> str:
    try:
        ops = json.loads(recipe)
    except json.JSONDecodeError:
        return ""
    if not ops:
        return ""
    return str(ops[0].get("op", ""))


def build_manifest_rows(view_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    view_specs = [
        ("raw", "raw", "is_raw_view"),
        ("anchor_clean", "anchor_clean", "is_anchor_clean_view"),
        ("conservative_clean", "conservative_clean", "is_conservative_clean_view"),
        ("clean_like", "conservative_clean", "is_conservative_clean_view"),
        ("intervened", "intervened", "is_intervened_view"),
        ("flagged_group", "group_controlled", "is_group_controlled_view"),
        ("balanced", "phase_balanced", "is_phase_balanced_view"),
        ("active_only", "active_only", "is_active_only_view"),
        ("daytime_only", "daytime_only", "is_daytime_only_view"),
    ]

    for row in view_df.itertuples(index=False):
        for view_name, semantics, flag_col in view_specs:
            if int(getattr(row, flag_col, 0)) != 1:
                continue
            if row.dataset_name != "solar_AL" and view_name in {"balanced", "active_only", "daytime_only"}:
                continue
            rows.append(
                {
                    "dataset_name": row.dataset_name,
                    "split_name": row.split_name,
                    "source_split": row.source_split,
                    "lookback": int(row.lookback),
                    "horizon": int(row.horizon),
                    "window_id": row.window_id,
                    "view_name": view_name,
                    "view_semantics": semantics,
                    "artifact_ids": row.artifact_ids,
                    "phase_group": row.dominant_phase_target,
                    "is_flagged": int(row.is_flagged),
                    "intervention_type": first_intervention_type(row.intervention_recipe),
                    "view_status": row.view_status,
                    "primary_group_key": row.primary_group_key,
                    "sample_weight": float(getattr(row, "phase_balance_weight", 1.0)),
                }
            )
    return rows


def build_eval_view_design_markdown(view_df: pd.DataFrame) -> str:
    lines = [
        "# Eval View Design",
        "",
        "以下设计把事件级 metadata 变成可执行的窗口视图系统。",
        "",
    ]
    for dataset_name, group in view_df.groupby("dataset_name", dropna=False):
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                f"- raw: {int(group['is_raw_view'].sum())} 窗口",
                f"- anchor_clean: {int(group['is_anchor_clean_view'].sum())} 窗口",
                f"- conservative_clean: {int(group['is_conservative_clean_view'].sum())} 窗口",
                f"- intervened: {int(group['is_intervened_view'].sum())} 窗口",
                f"- 当前 view_status: {dict(group['view_status'].value_counts())}",
            ]
        )
        if dataset_name == "solar_AL":
            balanced_subset = group[group["is_phase_balanced_view"] == 1]
            lines.extend(
                [
                    f"- balanced: {int(group['is_phase_balanced_view'].sum())} 窗口",
                    f"- active_only: {int(group['is_active_only_view'].sum())} 窗口",
                    f"- daytime_only: {int(group['is_daytime_only_view'].sum())} 窗口",
                    f"- balanced_weight range: {group['phase_balance_weight'].min():.3f} ~ {group['phase_balance_weight'].max():.3f}",
                    (
                        f"- balanced phase counts: {dict(balanced_subset['dominant_phase_target'].astype(str).value_counts())}"
                        if not balanced_subset.empty
                        else "- balanced phase counts: {}"
                    ),
                    "- 约束: night zeros 只做 phase-aware masking / stratify，不做 blind interpolation。",
                ]
            )
        else:
            lines.append("- clean_like 语义默认绑定 conservative_clean；anchor_clean 主要用于更严格 sanity check。")
        lines.append("")
    return "\n".join(lines)


def run_build_eval_views(
    spec_path: Path,
    scores_dir: Path,
    out_dir: Path,
    events_path: Path,
    manifest_out: Path,
    report_out: Path,
    datasets: list[str] | None = None,
    lookbacks: list[int] | None = None,
    horizons: list[int] | None = None,
) -> tuple[list[Path], Path]:
    ensure_project_directories()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    spec = load_view_spec(spec_path)
    events = pd.read_csv(events_path)
    events_lookup = {str(row["artifact_id"]): row.to_dict() for _, row in events.iterrows()}
    dataset_filter = {str(item) for item in datasets} if datasets else None
    lookback_filter = {int(item) for item in lookbacks} if lookbacks else None
    horizon_filter = {int(item) for item in horizons} if horizons else None
    score_paths = [
        path
        for path in sorted(scores_dir.glob("*.csv"))
        if score_path_matches(
            path,
            datasets=dataset_filter,
            lookbacks=lookback_filter,
            horizons=horizon_filter,
        )
    ]
    log_progress(f"start score_files={len(score_paths)} events={len(events)}")

    wide_frames: list[pd.DataFrame] = []
    written_paths: list[Path] = []
    for idx, score_path in enumerate(score_paths, start=1):
        log_progress(f"processing {idx}/{len(score_paths)} {score_path.name}")
        score_df = pd.read_csv(score_path)
        if score_df.empty:
            log_progress(f"skip empty score file {score_path.name}")
            continue
        view_df = add_dataset_specific_views(score_df, spec=spec, events_lookup=events_lookup)
        wide_frames.append(view_df)

    if not wide_frames:
        empty_manifest = pd.DataFrame()
        empty_manifest.to_csv(manifest_out, index=False)
        write_markdown(report_out, "# Eval View Design\n\n无可用窗口分数文件。")
        return written_paths, manifest_out

    all_views = apply_support_fallback(pd.concat(wide_frames, ignore_index=True), spec=spec)
    log_progress(f"combined rows={len(all_views)}")

    for (dataset_name, lookback, horizon), group in all_views.groupby(["dataset_name", "lookback", "horizon"], dropna=False):
        out_path = out_dir / f"{dataset_name}_L{lookback}_H{horizon}.csv"
        group.to_csv(out_path, index=False)
        written_paths.append(out_path)
        log_progress(
            f"wrote {out_path.name} rows={len(group)} balanced={int(group.get('is_phase_balanced_view', pd.Series(dtype=int)).sum())}"
        )

    manifest_rows = build_manifest_rows(all_views)
    pd.DataFrame(manifest_rows).to_csv(manifest_out, index=False)
    write_markdown(report_out, build_eval_view_design_markdown(all_views))
    log_progress(f"wrote manifest rows={len(manifest_rows)} -> {manifest_out}")
    return written_paths, manifest_out


def main() -> None:
    args = parse_args()
    run_build_eval_views(
        spec_path=Path(args.spec),
        scores_dir=Path(args.scores_dir),
        out_dir=Path(args.out_dir),
        events_path=Path(args.events),
        manifest_out=Path(args.manifest_out),
        report_out=Path(args.report_out),
        datasets=args.datasets,
        lookbacks=args.lookbacks,
        horizons=args.horizons,
    )


if __name__ == "__main__":
    main()

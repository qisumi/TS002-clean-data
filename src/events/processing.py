from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils.progress import progress


MERGE_GAP = {
    "ETTh1": 3,
    "ETTh2": 3,
    "ETTm1": 12,
    "ETTm2": 12,
    "weather": 6,
}

ETT_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}
SOLAR_DATASET = "solar_AL"
WEATHER_DATASET = "weather"

SOLAR_BASE_LABELS = {
    "night_zero_band": {
        "validity": "valid",
        "exploitability": "high",
        "recoverability": "mask_only",
        "recommended_policy": "stratify",
        "recommended_eval_view": "balanced/active_only/masked_phase_eval",
        "severity": 0.40,
    },
    "phase_transition_band": {
        "validity": "valid",
        "exploitability": "medium",
        "recoverability": "mask_only",
        "recommended_policy": "stratify",
        "recommended_eval_view": "balanced/masked_phase_eval",
        "severity": 0.20,
    },
    "active_period": {
        "validity": "valid",
        "exploitability": "low",
        "recoverability": "keep",
        "recommended_policy": "keep",
        "recommended_eval_view": "raw/active_only_eval",
        "severity": 0.00,
    },
}

SOLAR_AUGMENTED_LABELS = {
    "active_suspicious_zero": {
        "validity": "suspicious",
        "exploitability": "high",
        "recoverability": "repairable",
        "recommended_policy": "repair",
        "recommended_eval_view": "active_only_eval/intervened",
    },
    "active_suspicious_constant": {
        "validity": "suspicious",
        "exploitability": "high",
        "recoverability": "repairable",
        "recommended_policy": "repair",
        "recommended_eval_view": "active_only_eval/intervened",
    },
    "transition_suspicious_zero": {
        "validity": "suspicious",
        "exploitability": "medium",
        "recoverability": "mask_only",
        "recommended_policy": "mask",
        "recommended_eval_view": "balanced_eval/intervened",
    },
    "transition_suspicious_constant": {
        "validity": "suspicious",
        "exploitability": "medium",
        "recoverability": "mask_only",
        "recommended_policy": "mask",
        "recommended_eval_view": "balanced_eval/intervened",
    },
}

FINAL_EVENT_COLUMNS = [
    "artifact_id",
    "cluster_event_id",
    "raw_event_id",
    "raw_event_ids",
    "dataset_name",
    "source_dataset",
    "source_kind",
    "source_file",
    "start_idx",
    "end_idx",
    "length",
    "scope",
    "variables",
    "n_variables",
    "artifact_type",
    "artifact_types",
    "artifact_group",
    "validity",
    "validity_label",
    "exploitability",
    "exploitability_label",
    "confidence",
    "severity",
    "recoverability",
    "recommended_policy",
    "recommended_action",
    "recommended_eval_view",
    "phase_group",
    "phase_overlap_active",
    "phase_overlap_transition",
    "phase_overlap_night",
    "n_raw_candidates",
    "n_candidate_rows",
    "max_score",
    "score_max",
    "mean_score",
    "score_mean",
    "notes",
]

EVENT_MAPPING_COLUMNS = [
    "candidate_row_id",
    "dataset_name",
    "variable",
    "artifact_type",
    "start_idx",
    "end_idx",
    "score",
    "raw_event_id",
    "cluster_event_id",
    "artifact_id",
    "mapping_status",
    "mapped_source_kind",
]


@dataclass
class PhaseDetails:
    phase_group: str
    phase_overlap_active: float
    phase_overlap_transition: float
    phase_overlap_night: float
    dominant_phase_event_id: str
    dominant_phase_confidence: float


def stable_id(prefix: str, dataset_name: str, start_idx: int, end_idx: int, key: str) -> str:
    raw = f"{prefix}|{dataset_name}|{start_idx}|{end_idx}|{key}"
    digest = sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}_{prefix}_{digest}"


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def normalize_candidates(candidates: pd.DataFrame, datasets: Iterable[str]) -> pd.DataFrame:
    df = candidates.copy()
    df = df[df["dataset_name"].isin(set(datasets))].copy()
    if df.empty:
        return df
    df["candidate_row_id"] = np.arange(len(df), dtype=int)
    df["start_idx"] = df["start_idx"].astype(int)
    df["end_idx"] = df["end_idx"].astype(int)
    df["length"] = (df["end_idx"] - df["start_idx"] + 1).astype(int)
    df["score"] = df["score"].astype(float)
    df["variable"] = df["variable"].astype(str)
    df["artifact_type"] = df["artifact_type"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df = df[df["start_idx"] <= df["end_idx"]].copy()
    return df.sort_values(
        ["dataset_name", "variable", "artifact_type", "start_idx", "end_idx", "score"],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)


def merge_candidate_group(group: pd.DataFrame, gap: int, prefix: str) -> list[dict[str, Any]]:
    group = group.sort_values(["start_idx", "end_idx", "score"], ascending=[True, True, False]).reset_index(drop=True)
    merged_rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in group.to_dict(orient="records"):
        if current is None:
            current = {
                "dataset_name": row["dataset_name"],
                "variable": row["variable"],
                "artifact_type": row["artifact_type"],
                "start_idx": int(row["start_idx"]),
                "end_idx": int(row["end_idx"]),
                "scores": [float(row["score"])],
                "candidate_row_ids": [int(row["candidate_row_id"])],
                "source_files": {str(row["file_path"])},
            }
            continue
        if int(row["start_idx"]) <= int(current["end_idx"]) + gap:
            current["end_idx"] = max(int(current["end_idx"]), int(row["end_idx"]))
            current["scores"].append(float(row["score"]))
            current["candidate_row_ids"].append(int(row["candidate_row_id"]))
            current["source_files"].add(str(row["file_path"]))
            continue
        merged_rows.append(finalize_merged_candidate(current, prefix))
        current = {
            "dataset_name": row["dataset_name"],
            "variable": row["variable"],
            "artifact_type": row["artifact_type"],
            "start_idx": int(row["start_idx"]),
            "end_idx": int(row["end_idx"]),
            "scores": [float(row["score"])],
            "candidate_row_ids": [int(row["candidate_row_id"])],
            "source_files": {str(row["file_path"])},
        }
    if current is not None:
        merged_rows.append(finalize_merged_candidate(current, prefix))
    return merged_rows


def finalize_merged_candidate(current: dict[str, Any], prefix: str) -> dict[str, Any]:
    start_idx = int(current["start_idx"])
    end_idx = int(current["end_idx"])
    raw_event_id = stable_id(
        prefix=prefix,
        dataset_name=str(current["dataset_name"]),
        start_idx=start_idx,
        end_idx=end_idx,
        key=f"{current['variable']}|{current['artifact_type']}",
    )
    scores = list(current["scores"])
    return {
        "dataset_name": str(current["dataset_name"]),
        "variable": str(current["variable"]),
        "artifact_type": str(current["artifact_type"]),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "length": end_idx - start_idx + 1,
        "score_max": float(max(scores)),
        "score_mean": float(np.mean(scores)),
        "candidate_row_ids": list(current["candidate_row_ids"]),
        "source_files": sorted(current["source_files"]),
        "raw_event_id": raw_event_id,
    }


def cluster_overlapping_records(
    records: list[dict[str, Any]],
    allow_touching_gap: int = 0,
) -> list[list[dict[str, Any]]]:
    if not records:
        return []

    sorted_records = sorted(
        records,
        key=lambda record: (int(record["start_idx"]), int(record["end_idx"])),
    )
    clusters: list[list[dict[str, Any]]] = []
    current_cluster: list[dict[str, Any]] = []
    current_end = -1

    for record in sorted_records:
        start_idx = int(record["start_idx"])
        end_idx = int(record["end_idx"])
        if not current_cluster or start_idx > current_end + allow_touching_gap:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [record]
            current_end = end_idx
            continue
        current_cluster.append(record)
        current_end = max(current_end, end_idx)

    if current_cluster:
        clusters.append(current_cluster)
    return clusters


def confidence_for_ett(n_raw_candidates: int, n_variables: int) -> float:
    if n_raw_candidates >= 3 or n_variables >= 2:
        return 0.90
    if n_raw_candidates == 2:
        return 0.75
    return 0.60


def confidence_for_rule_dataset(dataset_name: str, n_raw_candidates: int, n_variables: int) -> float:
    # Weather is much denser and its rule-based candidates are noisier than the
    # ETT family, so we intentionally lower confidence to avoid collapsing all
    # windows into "corrupted/unrecoverable" by default.
    if dataset_name == WEATHER_DATASET:
        if n_raw_candidates >= 3 or n_variables >= 2:
            return 0.70
        if n_raw_candidates == 2:
            return 0.55
        return 0.45
    return confidence_for_ett(n_raw_candidates=n_raw_candidates, n_variables=n_variables)


def severity_from_distribution(lengths: pd.Series, scores: pd.Series, fixed: float | None = None) -> pd.Series:
    if fixed is not None:
        return pd.Series(np.full(len(lengths), fixed, dtype=float), index=lengths.index)
    length_p95 = max(float(lengths.quantile(0.95)), 1.0)
    score_p95 = max(float(scores.quantile(0.95)), 1e-6)
    norm_length = np.minimum(1.0, lengths.astype(float) / length_p95)
    norm_score = np.minimum(1.0, scores.astype(float) / score_p95)
    return pd.Series(np.clip(0.5 * norm_length + 0.5 * norm_score, 0.0, 1.0), index=lengths.index)


def annotate_severity(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    severities = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    for (dataset_name, artifact_group), group in out.groupby(["dataset_name", "artifact_group"], dropna=False):
        fixed = SOLAR_BASE_LABELS.get(str(artifact_group), {}).get("severity") if dataset_name == SOLAR_DATASET else None
        severities.loc[group.index] = severity_from_distribution(group["length"], group["score_max"], fixed=fixed).to_numpy()
    out["severity"] = severities.round(6)
    return out


def determine_ett_artifact_group(artifact_types: list[str], n_variables: int) -> str:
    unique_types = sorted(set(artifact_types))
    if unique_types == ["suspicious_repetition"]:
        return "suspicious_repetition_event"
    if set(unique_types).issubset({"flat_run", "near_constant_segment"}):
        return "local_near_constant_event"
    if "zero_block" in unique_types:
        return "multi_var_plateau_or_zero_block"
    if n_variables >= 2 and set(unique_types).issubset({"flat_run", "near_constant_segment", "zero_block"}):
        return "multi_var_plateau_or_zero_block"
    return "mixed_suspicious_event"


def label_ett_cluster(artifact_group: str, artifact_types: list[str]) -> dict[str, str]:
    if "suspicious_repetition" in set(artifact_types) or artifact_group == "suspicious_repetition_event":
        return {
            "validity": "corrupted",
            "exploitability": "high",
            "recoverability": "unrecoverable",
            "recommended_policy": "drop",
            "recommended_eval_view": "clean_like/intervened/flagged_group",
        }
    if artifact_group in {"multi_var_plateau_or_zero_block", "local_near_constant_event"}:
        return {
            "validity": "suspicious",
            "exploitability": "high",
            "recoverability": "repairable",
            "recommended_policy": "repair",
            "recommended_eval_view": "clean_like/intervened",
        }
    return {
        "validity": "suspicious",
        "exploitability": "medium",
        "recoverability": "mask_only",
        "recommended_policy": "mask",
        "recommended_eval_view": "intervened/flagged_group",
    }


def label_weather_cluster(artifact_group: str, artifact_types: list[str]) -> dict[str, str]:
    unique_types = set(artifact_types)
    if "suspicious_repetition" in unique_types:
        return {
            "validity": "suspicious",
            "exploitability": "medium",
            "recoverability": "repairable",
            "recommended_policy": "repair",
            "recommended_eval_view": "clean_like/intervened",
        }
    if artifact_group in {"multi_var_plateau_or_zero_block", "local_near_constant_event"}:
        return {
            "validity": "suspicious",
            "exploitability": "medium",
            "recoverability": "repairable",
            "recommended_policy": "repair",
            "recommended_eval_view": "clean_like/intervened",
        }
    return {
        "validity": "suspicious",
        "exploitability": "low",
        "recoverability": "repairable",
        "recommended_policy": "repair",
        "recommended_eval_view": "clean_like/intervened",
    }


def label_rule_cluster(dataset_name: str, artifact_group: str, artifact_types: list[str]) -> dict[str, str]:
    if dataset_name == WEATHER_DATASET:
        return label_weather_cluster(artifact_group=artifact_group, artifact_types=artifact_types)
    return label_ett_cluster(artifact_group=artifact_group, artifact_types=artifact_types)


def build_ett_events(candidates: pd.DataFrame, show_progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    # This merge path is also used for rule-based non-solar datasets such as
    # weather, where we do not have solar-style phase annotations.
    merged_candidates = candidates[candidates["dataset_name"] != SOLAR_DATASET].copy()
    if merged_candidates.empty:
        return (
            pd.DataFrame(columns=FINAL_EVENT_COLUMNS),
            pd.DataFrame(columns=EVENT_MAPPING_COLUMNS),
        )

    merged_variable_rows: list[dict[str, Any]] = []
    grouped = list(
        merged_candidates.groupby(
            ["dataset_name", "variable", "artifact_type"],
            dropna=False,
        )
    )
    for (dataset_name, variable, artifact_type), group in progress(
        grouped,
        total=len(grouped),
        desc="004/ETT variable-merge",
        disable=not show_progress,
    ):
        gap = MERGE_GAP.get(str(dataset_name), 3)
        merged_variable_rows.extend(merge_candidate_group(group, gap=gap, prefix="RAW"))

    variable_df = pd.DataFrame(merged_variable_rows)
    final_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    candidate_lookup = merged_candidates.set_index("candidate_row_id")

    dataset_groups = list(variable_df.groupby("dataset_name", dropna=False))
    for dataset_name, group in progress(
        dataset_groups,
        total=len(dataset_groups),
        desc="004/ETT cluster-merge",
        disable=not show_progress,
    ):
        records = group.to_dict(orient="records")
        for cluster_records in cluster_overlapping_records(records, allow_touching_gap=0):
            start_idx = min(int(rec["start_idx"]) for rec in cluster_records)
            end_idx = max(int(rec["end_idx"]) for rec in cluster_records)
            variables = sorted({str(rec["variable"]) for rec in cluster_records})
            artifact_types = sorted({str(rec["artifact_type"]) for rec in cluster_records})
            artifact_group = determine_ett_artifact_group(artifact_types, n_variables=len(variables))
            labels = label_rule_cluster(str(dataset_name), artifact_group, artifact_types)
            raw_event_ids = sorted(str(rec["raw_event_id"]) for rec in cluster_records)
            artifact_id = stable_id(
                prefix="EVT",
                dataset_name=str(dataset_name),
                start_idx=start_idx,
                end_idx=end_idx,
                key=f"{artifact_group}|{','.join(variables)}|{','.join(artifact_types)}",
            )
            scores = [float(rec["score_max"]) for rec in cluster_records]
            n_raw_candidates = int(sum(len(rec["candidate_row_ids"]) for rec in cluster_records))
            final_rows.append(
                {
                    "artifact_id": artifact_id,
                    "cluster_event_id": artifact_id,
                    "raw_event_id": raw_event_ids[0],
                    "raw_event_ids": ",".join(raw_event_ids),
                    "dataset_name": str(dataset_name),
                    "source_dataset": str(dataset_name),
                    "source_kind": "merged_candidate",
                    "source_file": ",".join(sorted({sf for rec in cluster_records for sf in rec["source_files"]})),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "length": end_idx - start_idx + 1,
                    "scope": "multivariate" if len(variables) >= 2 else "univariate",
                    "variables": ",".join(variables),
                    "n_variables": len(variables),
                    "artifact_type": artifact_types[0] if len(artifact_types) == 1 else ",".join(artifact_types),
                    "artifact_types": ",".join(artifact_types),
                    "artifact_group": artifact_group,
                    "validity": labels["validity"],
                    "validity_label": labels["validity"],
                    "exploitability": labels["exploitability"],
                    "exploitability_label": labels["exploitability"],
                    "confidence": round(
                        confidence_for_rule_dataset(
                            dataset_name=str(dataset_name),
                            n_raw_candidates=n_raw_candidates,
                            n_variables=len(variables),
                        ),
                        6,
                    ),
                    "severity": 0.0,
                    "recoverability": labels["recoverability"],
                    "recommended_policy": labels["recommended_policy"],
                    "recommended_action": labels["recommended_policy"],
                    "recommended_eval_view": labels["recommended_eval_view"],
                    "phase_group": "NA",
                    "phase_overlap_active": 0.0,
                    "phase_overlap_transition": 0.0,
                    "phase_overlap_night": 0.0,
                    "n_raw_candidates": n_raw_candidates,
                    "n_candidate_rows": n_raw_candidates,
                    "max_score": round(float(max(scores)), 6),
                    "score_max": round(float(max(scores)), 6),
                    "mean_score": round(float(np.mean(scores)), 6),
                    "score_mean": round(float(np.mean(scores)), 6),
                    "notes": (
                        "weather softened rule label"
                        if str(dataset_name) == WEATHER_DATASET
                        else "mixed merged ETT artifact types" if artifact_group == "mixed_suspicious_event" else ""
                    ),
                }
            )
            for record in cluster_records:
                for candidate_row_id in record["candidate_row_ids"]:
                    candidate = candidate_lookup.loc[int(candidate_row_id)]
                    mapping_rows.append(
                        {
                            "candidate_row_id": int(candidate_row_id),
                            "dataset_name": str(candidate["dataset_name"]),
                            "variable": str(candidate["variable"]),
                            "artifact_type": str(candidate["artifact_type"]),
                            "start_idx": int(candidate["start_idx"]),
                            "end_idx": int(candidate["end_idx"]),
                            "score": round(float(candidate["score"]), 6),
                            "raw_event_id": str(record["raw_event_id"]),
                            "cluster_event_id": artifact_id,
                            "artifact_id": artifact_id,
                            "mapping_status": "mapped",
                            "mapped_source_kind": "merged_candidate",
                        }
                    )

    final_df = annotate_severity(pd.DataFrame(final_rows))
    mapping_df = pd.DataFrame(mapping_rows).sort_values(["dataset_name", "candidate_row_id"]).reset_index(drop=True)
    return final_df.loc[:, FINAL_EVENT_COLUMNS].copy(), mapping_df.loc[:, EVENT_MAPPING_COLUMNS].copy()


def dominant_phase_details(start_idx: int, end_idx: int, base_events: pd.DataFrame) -> PhaseDetails:
    overlaps = base_events[
        (base_events["start_idx"] <= end_idx) & (base_events["end_idx"] >= start_idx)
    ].copy()
    if overlaps.empty:
        return PhaseDetails("unknown", 0.0, 0.0, 0.0, "", 0.0)
    overlaps["overlap_len"] = overlaps.apply(
        lambda row: overlap_len(start_idx, end_idx, int(row["start_idx"]), int(row["end_idx"])),
        axis=1,
    )
    overlaps = overlaps[overlaps["overlap_len"] > 0].copy()
    total_overlap = max(int(overlaps["overlap_len"].sum()), 1)
    phase_sums = overlaps.groupby("phase_group", dropna=False)["overlap_len"].sum().to_dict()
    dominant = overlaps.sort_values(["overlap_len", "confidence"], ascending=[False, False]).iloc[0]
    return PhaseDetails(
        phase_group=str(dominant["phase_group"]),
        phase_overlap_active=round(float(phase_sums.get("active", 0) / total_overlap), 6),
        phase_overlap_transition=round(float(phase_sums.get("transition", 0) / total_overlap), 6),
        phase_overlap_night=round(float(phase_sums.get("night", 0) / total_overlap), 6),
        dominant_phase_event_id=str(dominant["artifact_id"]),
        dominant_phase_confidence=float(dominant["confidence"]),
    )


def build_phase_segments(base_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[int]]:
    segments = []
    for row in base_df.sort_values(["start_idx", "end_idx"]).itertuples(index=False):
        segments.append(
            {
                "start_idx": int(row.start_idx),
                "end_idx": int(row.end_idx),
                "phase_group": str(row.phase_group),
                "artifact_id": str(row.artifact_id),
                "confidence": float(row.confidence),
            }
        )
    starts = [segment["start_idx"] for segment in segments]
    return segments, starts


def dominant_phase_details_fast(
    start_idx: int,
    end_idx: int,
    phase_segments: list[dict[str, Any]],
    phase_starts: list[int],
) -> PhaseDetails:
    if not phase_segments:
        return PhaseDetails("unknown", 0.0, 0.0, 0.0, "", 0.0)

    scan_idx = max(0, bisect_right(phase_starts, start_idx) - 1)
    overlaps: list[tuple[int, dict[str, Any]]] = []
    while scan_idx < len(phase_segments) and int(phase_segments[scan_idx]["start_idx"]) <= end_idx:
        segment = phase_segments[scan_idx]
        ov = overlap_len(start_idx, end_idx, int(segment["start_idx"]), int(segment["end_idx"]))
        if ov > 0:
            overlaps.append((ov, segment))
        scan_idx += 1

    if not overlaps:
        return PhaseDetails("unknown", 0.0, 0.0, 0.0, "", 0.0)

    total_overlap = max(sum(item[0] for item in overlaps), 1)
    phase_sums = {"active": 0, "transition": 0, "night": 0}
    for ov, segment in overlaps:
        phase_group = str(segment["phase_group"])
        if phase_group in phase_sums:
            phase_sums[phase_group] += ov

    dominant_overlap, dominant_segment = max(overlaps, key=lambda item: (item[0], float(item[1]["confidence"])))
    _ = dominant_overlap
    return PhaseDetails(
        phase_group=str(dominant_segment["phase_group"]),
        phase_overlap_active=round(float(phase_sums.get("active", 0) / total_overlap), 6),
        phase_overlap_transition=round(float(phase_sums.get("transition", 0) / total_overlap), 6),
        phase_overlap_night=round(float(phase_sums.get("night", 0) / total_overlap), 6),
        dominant_phase_event_id=str(dominant_segment["artifact_id"]),
        dominant_phase_confidence=float(dominant_segment["confidence"]),
    )


def build_solar_events(
    candidates: pd.DataFrame,
    phase_annotations: pd.DataFrame,
    dataset_registry: pd.DataFrame,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solar_candidates = candidates[candidates["dataset_name"] == SOLAR_DATASET].copy()
    if phase_annotations.empty:
        return (
            pd.DataFrame(columns=FINAL_EVENT_COLUMNS),
            pd.DataFrame(columns=EVENT_MAPPING_COLUMNS),
        )

    registry_row = dataset_registry.loc[dataset_registry["dataset_name"] == SOLAR_DATASET].iloc[0]
    n_variables = int(registry_row["n_vars"])
    base_rows: list[dict[str, Any]] = []
    for idx, row in phase_annotations.reset_index(drop=True).iterrows():
        artifact_group = str(row["artifact_group"])
        labels = SOLAR_BASE_LABELS[artifact_group]
        phase_group = str(row["phase_group"])
        artifact_id = stable_id(
            prefix="PHASE",
            dataset_name=SOLAR_DATASET,
            start_idx=int(row["start_idx"]),
            end_idx=int(row["end_idx"]),
            key=artifact_group,
        )
        base_rows.append(
            {
                "artifact_id": artifact_id,
                "cluster_event_id": artifact_id,
                "raw_event_id": f"{SOLAR_DATASET}_phase_{idx:04d}",
                "raw_event_ids": f"{SOLAR_DATASET}_phase_{idx:04d}",
                "dataset_name": SOLAR_DATASET,
                "source_dataset": SOLAR_DATASET,
                "source_kind": "phase_annotation",
                "source_file": "statistic_results/solar_AL_phase_annotations.csv",
                "start_idx": int(row["start_idx"]),
                "end_idx": int(row["end_idx"]),
                "length": int(row["length"]),
                "scope": "phase_segment",
                "variables": "ALL",
                "n_variables": n_variables,
                "artifact_type": artifact_group,
                "artifact_types": artifact_group,
                "artifact_group": artifact_group,
                "validity": labels["validity"],
                "validity_label": labels["validity"],
                "exploitability": labels["exploitability"],
                "exploitability_label": labels["exploitability"],
                "confidence": round(float(row["confidence"]), 6),
                "severity": labels["severity"],
                "recoverability": labels["recoverability"],
                "recommended_policy": labels["recommended_policy"],
                "recommended_action": labels["recommended_policy"],
                "recommended_eval_view": labels["recommended_eval_view"],
                "phase_group": phase_group,
                "phase_overlap_active": 1.0 if phase_group == "active" else 0.0,
                "phase_overlap_transition": 1.0 if phase_group == "transition" else 0.0,
                "phase_overlap_night": 1.0 if phase_group == "night" else 0.0,
                "n_raw_candidates": 0,
                "n_candidate_rows": 0,
                "max_score": round(float(row["confidence"]), 6),
                "score_max": round(float(row["confidence"]), 6),
                "mean_score": round(float(row["confidence"]), 6),
                "score_mean": round(float(row["confidence"]), 6),
                "notes": "phase-derived base event",
            }
        )
    base_df = pd.DataFrame(base_rows).loc[:, FINAL_EVENT_COLUMNS].copy()
    phase_segments, phase_starts = build_phase_segments(base_df)

    if solar_candidates.empty:
        return base_df, pd.DataFrame(columns=EVENT_MAPPING_COLUMNS)

    score_threshold = float(solar_candidates["score"].quantile(0.90))
    merged_rows: list[dict[str, Any]] = []
    grouped = list(solar_candidates.groupby(["variable", "artifact_type"], dropna=False))
    for (variable, artifact_type), group in progress(
        grouped,
        total=len(grouped),
        desc="004/Solar candidate-merge",
        disable=not show_progress,
    ):
        merged_rows.extend(merge_candidate_group(group, gap=2, prefix="SOLRAW"))

    candidate_lookup = solar_candidates.set_index("candidate_row_id")
    high_signal_records: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for merged in progress(
        merged_rows,
        total=len(merged_rows),
        desc="004/Solar phase-map",
        disable=not show_progress,
    ):
        phase_details = dominant_phase_details_fast(int(merged["start_idx"]), int(merged["end_idx"]), phase_segments, phase_starts)
        if phase_details.phase_group == "night" or int(merged["length"]) < 6 or float(merged["score_max"]) < score_threshold:
            for candidate_row_id in merged["candidate_row_ids"]:
                candidate = candidate_lookup.loc[int(candidate_row_id)]
                mapping_rows.append(
                    {
                        "candidate_row_id": int(candidate_row_id),
                        "dataset_name": SOLAR_DATASET,
                        "variable": str(candidate["variable"]),
                        "artifact_type": str(candidate["artifact_type"]),
                        "start_idx": int(candidate["start_idx"]),
                        "end_idx": int(candidate["end_idx"]),
                        "score": round(float(candidate["score"]), 6),
                        "raw_event_id": str(merged["raw_event_id"]),
                        "cluster_event_id": phase_details.dominant_phase_event_id,
                        "artifact_id": phase_details.dominant_phase_event_id,
                        "mapping_status": "mapped_to_phase_annotation",
                        "mapped_source_kind": "phase_annotation",
                    }
                )
            continue
        merged["phase_group"] = phase_details.phase_group
        merged["phase_overlap_active"] = phase_details.phase_overlap_active
        merged["phase_overlap_transition"] = phase_details.phase_overlap_transition
        merged["phase_overlap_night"] = phase_details.phase_overlap_night
        merged["phase_confidence"] = phase_details.dominant_phase_confidence
        high_signal_records.append(merged)

    augmented_rows: list[dict[str, Any]] = []
    clusters = cluster_overlapping_records(high_signal_records, allow_touching_gap=2)
    for cluster_records in progress(
        clusters,
        total=len(clusters),
        desc="004/Solar augment",
        disable=not show_progress,
    ):
        start_idx = min(int(rec["start_idx"]) for rec in cluster_records)
        end_idx = max(int(rec["end_idx"]) for rec in cluster_records)
        variables = sorted({str(rec["variable"]) for rec in cluster_records})
        artifact_types = sorted({str(rec["artifact_type"]) for rec in cluster_records})
        phase_group = str(cluster_records[0]["phase_group"])
        artifact_group = (
            "active_suspicious_zero"
            if phase_group == "active" and "zero_block" in artifact_types
            else "active_suspicious_constant"
            if phase_group == "active"
            else "transition_suspicious_zero"
            if "zero_block" in artifact_types
            else "transition_suspicious_constant"
        )
        labels = SOLAR_AUGMENTED_LABELS[artifact_group]
        artifact_id = stable_id(
            prefix="AUG",
            dataset_name=SOLAR_DATASET,
            start_idx=start_idx,
            end_idx=end_idx,
            key=f"{artifact_group}|{','.join(variables)}",
        )
        score_max = max(float(rec["score_max"]) for rec in cluster_records)
        score_mean = float(np.mean([float(rec["score_mean"]) for rec in cluster_records]))
        phase_details = dominant_phase_details_fast(start_idx, end_idx, phase_segments, phase_starts)
        confidence = min(
            max(float(np.mean([float(rec["phase_confidence"]) for rec in cluster_records])), 0.1),
            min(1.0, score_max / max(score_threshold, 1e-6)),
        )
        raw_event_ids = sorted(str(rec["raw_event_id"]) for rec in cluster_records)
        candidate_ids = [cid for rec in cluster_records for cid in rec["candidate_row_ids"]]
        augmented_rows.append(
            {
                "artifact_id": artifact_id,
                "cluster_event_id": artifact_id,
                "raw_event_id": raw_event_ids[0],
                "raw_event_ids": ",".join(raw_event_ids),
                "dataset_name": SOLAR_DATASET,
                "source_dataset": SOLAR_DATASET,
                "source_kind": "phase_augmented",
                "source_file": str(registry_row["file_path"]),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length": end_idx - start_idx + 1,
                "scope": "multivariate" if len(variables) >= 2 else "univariate",
                "variables": ",".join(variables),
                "n_variables": len(variables),
                "artifact_type": artifact_types[0] if len(artifact_types) == 1 else ",".join(artifact_types),
                "artifact_types": ",".join(artifact_types),
                "artifact_group": artifact_group,
                "validity": labels["validity"],
                "validity_label": labels["validity"],
                "exploitability": labels["exploitability"],
                "exploitability_label": labels["exploitability"],
                "confidence": round(float(confidence), 6),
                "severity": 0.0,
                "recoverability": labels["recoverability"],
                "recommended_policy": labels["recommended_policy"],
                "recommended_action": labels["recommended_policy"],
                "recommended_eval_view": labels["recommended_eval_view"],
                "phase_group": phase_group,
                "phase_overlap_active": phase_details.phase_overlap_active,
                "phase_overlap_transition": phase_details.phase_overlap_transition,
                "phase_overlap_night": phase_details.phase_overlap_night,
                "n_raw_candidates": len(candidate_ids),
                "n_candidate_rows": len(candidate_ids),
                "max_score": round(score_max, 6),
                "score_max": round(score_max, 6),
                "mean_score": round(score_mean, 6),
                "score_mean": round(score_mean, 6),
                "notes": "phase-aware augmented suspicious event",
            }
        )
        for record in cluster_records:
            for candidate_row_id in record["candidate_row_ids"]:
                candidate = candidate_lookup.loc[int(candidate_row_id)]
                mapping_rows.append(
                    {
                        "candidate_row_id": int(candidate_row_id),
                        "dataset_name": SOLAR_DATASET,
                        "variable": str(candidate["variable"]),
                        "artifact_type": str(candidate["artifact_type"]),
                        "start_idx": int(candidate["start_idx"]),
                        "end_idx": int(candidate["end_idx"]),
                        "score": round(float(candidate["score"]), 6),
                        "raw_event_id": str(record["raw_event_id"]),
                        "cluster_event_id": artifact_id,
                        "artifact_id": artifact_id,
                        "mapping_status": "mapped_to_phase_augmented",
                        "mapped_source_kind": "phase_augmented",
                    }
                )

    if augmented_rows:
        augmented_df = annotate_severity(pd.DataFrame(augmented_rows)).loc[:, FINAL_EVENT_COLUMNS].copy()
        final_df = pd.concat([base_df, augmented_df], ignore_index=True)
    else:
        final_df = base_df.copy()
    mapping_df = pd.DataFrame(mapping_rows).sort_values(["candidate_row_id"]).reset_index(drop=True)
    return final_df.loc[:, FINAL_EVENT_COLUMNS].copy(), mapping_df.loc[:, EVENT_MAPPING_COLUMNS].copy()


def build_final_event_table(
    candidates: pd.DataFrame,
    phase_annotations: pd.DataFrame,
    dataset_registry: pd.DataFrame,
    datasets: Iterable[str],
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = normalize_candidates(candidates, datasets=datasets)
    phase_annotations = phase_annotations.copy()
    if not phase_annotations.empty:
        phase_annotations["start_idx"] = phase_annotations["start_idx"].astype(int)
        phase_annotations["end_idx"] = phase_annotations["end_idx"].astype(int)
        phase_annotations["length"] = phase_annotations["length"].astype(int)
        phase_annotations["confidence"] = phase_annotations["confidence"].astype(float)
        phase_annotations["phase_group"] = phase_annotations["phase_group"].astype(str)
        phase_annotations["artifact_group"] = phase_annotations["artifact_group"].astype(str)

    ett_df, ett_mapping = build_ett_events(normalized, show_progress=show_progress)
    solar_df, solar_mapping = build_solar_events(normalized, phase_annotations, dataset_registry, show_progress=show_progress)
    events = pd.concat([ett_df, solar_df], ignore_index=True)
    mappings = pd.concat([ett_mapping, solar_mapping], ignore_index=True)
    events = events.sort_values(["dataset_name", "start_idx", "end_idx", "artifact_group"]).reset_index(drop=True)
    mappings = mappings.sort_values(["dataset_name", "candidate_row_id"]).reset_index(drop=True)
    return events.loc[:, FINAL_EVENT_COLUMNS].copy(), mappings.loc[:, EVENT_MAPPING_COLUMNS].copy()


def interval_coverage_ratio(events: pd.DataFrame, n_rows: int) -> float:
    if events.empty or n_rows <= 0:
        return 0.0
    merged: list[tuple[int, int]] = []
    for row in events.sort_values(["start_idx", "end_idx"]).itertuples(index=False):
        start_idx = int(row.start_idx)
        end_idx = int(row.end_idx)
        if not merged or start_idx > merged[-1][1] + 1:
            merged.append((start_idx, end_idx))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end_idx))
    covered = sum(end_idx - start_idx + 1 for start_idx, end_idx in merged)
    return covered / max(n_rows, 1)


def summarize_top_variables(events: pd.DataFrame, limit: int = 5) -> str:
    if events.empty:
        return "NA"
    counts: dict[str, int] = {}
    for value in events["variables"].fillna(""):
        for variable in [item.strip() for item in str(value).split(",") if item.strip() and item.strip() != "ALL"]:
            counts[variable] = counts.get(variable, 0) + 1
    if not counts:
        return "ALL"
    top_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{name}({count})" for name, count in top_items)


def build_event_summary_markdown(events: pd.DataFrame, dataset_registry: pd.DataFrame) -> str:
    lines = [
        "# Artifact Event Summary",
        "",
        "本报告汇总 candidate -> event 收束后的事件级 metadata。",
        "",
    ]
    for dataset_name, group in events.groupby("dataset_name", dropna=False):
        registry_row = dataset_registry.loc[dataset_registry["dataset_name"] == dataset_name].iloc[0]
        coverage = interval_coverage_ratio(group, int(registry_row["n_rows"]))
        by_group = group.groupby("artifact_group", dropna=False).size().sort_values(ascending=False)
        by_phase = group[group["phase_group"].ne("NA")].groupby("phase_group", dropna=False).size().sort_values(ascending=False)
        by_validity = group.groupby("validity", dropna=False).size().sort_values(ascending=False)
        lines.extend(
            [
                f"## {dataset_name}",
                "",
                f"- 事件数: {len(group)}",
                f"- 覆盖率: {coverage:.2%}",
                f"- 主导变量: {summarize_top_variables(group)}",
                f"- validity 分布: {dict(by_validity)}",
                f"- artifact_group 分布: {dict(by_group.head(8))}",
            ]
        )
        if not by_phase.empty:
            lines.append(f"- 主导相位分布: {dict(by_phase)}")
        lines.append("")
    lines.extend(
        [
            "## 字段说明",
            "",
            "- `artifact_id`: 事件级稳定 ID。",
            "- `source_kind`: `merged_candidate / phase_annotation / phase_augmented`。",
            "- `recommended_policy`: 后续 view / intervention 的默认策略。",
            "- `phase_overlap_*`: solar phase-aware 事件的相位重叠比例。",
            "",
        ]
    )
    return "\n".join(lines)


def build_event_metadata_markdown(events: pd.DataFrame, mappings: pd.DataFrame, dataset_registry: pd.DataFrame) -> str:
    lines = [
        "# Final Artifact Events",
        "",
        f"- 总事件数: {len(events)}",
        f"- 总候选映射数: {len(mappings)}",
        "",
        "## 数据集摘要",
        "",
    ]
    for dataset_name, group in events.groupby("dataset_name", dropna=False):
        registry_row = dataset_registry.loc[dataset_registry["dataset_name"] == dataset_name].iloc[0]
        coverage = interval_coverage_ratio(group, int(registry_row["n_rows"]))
        lines.extend(
            [
                f"### {dataset_name}",
                "",
                f"- 事件数: {len(group)}",
                f"- 覆盖率: {coverage:.2%}",
                f"- 平均置信度: {group['confidence'].mean():.3f}",
                f"- 平均严重度: {group['severity'].mean():.3f}",
                f"- artifact_group 分布: {dict(group['artifact_group'].value_counts().head(8))}",
                "",
            ]
        )
    return "\n".join(lines)


def ensure_event_schema(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    for column in FINAL_EVENT_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out.loc[:, FINAL_EVENT_COLUMNS].copy()


def ensure_mapping_schema(mapping: pd.DataFrame) -> pd.DataFrame:
    out = mapping.copy()
    for column in EVENT_MAPPING_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out.loc[:, EVENT_MAPPING_COLUMNS].copy()

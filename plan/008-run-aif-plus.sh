#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_arrays
load_plan_gpu_arrays

if [[ "${#PLAN_GPU_IDS_ARR[@]}" -eq 0 ]]; then
  plan_log "no GPUs detected; AIF-Plus requires at least one GPU"
  exit 1
fi

PLAN_AIF_PLUS_DATASETS=("${PLAN_DATASETS_ARR[@]}")
PLAN_AIF_PLUS_WORKER_COUNT="${#PLAN_GPU_IDS_ARR[@]}"
if (( PLAN_AIF_PLUS_WORKER_COUNT > ${#PLAN_AIF_PLUS_DATASETS[@]} )); then
  PLAN_AIF_PLUS_WORKER_COUNT="${#PLAN_AIF_PLUS_DATASETS[@]}"
fi
if (( PLAN_AIF_PLUS_WORKER_COUNT <= 0 )); then
  plan_log "no datasets selected for AIF-Plus"
  exit 1
fi

if [[ "${RESULTS_DIR}" == /* ]]; then
  RESULTS_DIR_ABS="${RESULTS_DIR}"
else
  RESULTS_DIR_ABS="${ROOT_DIR}/${RESULTS_DIR}"
fi
if [[ "${REPORTS_DIR}" == /* ]]; then
  REPORTS_DIR_ABS="${REPORTS_DIR}"
else
  REPORTS_DIR_ABS="${ROOT_DIR}/${REPORTS_DIR}"
fi

PLAN_AIF_PLUS_THREADS="$(resolve_plan_worker_threads "${PLAN_AIF_PLUS_WORKER_COUNT}")"
PLAN_AIF_PLUS_NUM_WORKERS="$(resolve_plan_dataloader_workers)"
PLAN_AIF_PLUS_TMP_DIR="$(plan_make_temp_dir "008-aif-plus")"
trap 'cleanup_plan_temp_dir "${PLAN_AIF_PLUS_TMP_DIR}"' EXIT

partition_items_round_robin PLAN_AIF_PLUS_DATASETS "${PLAN_AIF_PLUS_WORKER_COUNT}" PLAN_AIF_PLUS_DATASET_GROUPS

plan_log "datasets=$(join_by_comma "${PLAN_AIF_PLUS_DATASETS[@]}")"
plan_log "gpus=$(join_by_comma "${PLAN_GPU_IDS_ARR[@]}")"
plan_log "worker_count=${PLAN_AIF_PLUS_WORKER_COUNT} threads_per_worker=${PLAN_AIF_PLUS_THREADS} dataloader_workers=${PLAN_AIF_PLUS_NUM_WORKERS}"

plan_bg_reset
for ((worker_idx = 0; worker_idx < PLAN_AIF_PLUS_WORKER_COUNT; worker_idx++)); do
  dataset_group="${PLAN_AIF_PLUS_DATASET_GROUPS[worker_idx]}"
  if [[ -z "${dataset_group}" ]]; then
    continue
  fi
  gpu_id="${PLAN_GPU_IDS_ARR[worker_idx]}"
  shard_dir="${PLAN_AIF_PLUS_TMP_DIR}/shard_${worker_idx}"
  shard_config="${PLAN_AIF_PLUS_TMP_DIR}/aif_plus_shard_${worker_idx}.yaml"
  mkdir -p "${shard_dir}"
  write_dataset_shard_config "${AIF_PLUS_CONFIG}" "${shard_config}" "${dataset_group}" "${PLAN_AIF_PLUS_NUM_WORKERS}"
  run_module_on_gpu_bg \
    "aif-plus-shard-${worker_idx}" \
    "${gpu_id}" \
    "${PLAN_AIF_PLUS_THREADS}" \
    "$(plan_child_log_path "008-aif-plus-shard-${worker_idx}")" \
    cli.run_aif_plus \
    --config "${shard_config}" \
    --views-dir "${STATS_DIR}/window_views" \
    --registry "${STATS_DIR}/dataset_registry.csv" \
    --events "${STATS_DIR}/final_artifact_events.csv" \
    --support-summary "${REPORTS_DIR}/clean_view_support_summary.csv" \
    --baseline-results "${RESULTS_DIR}/counterfactual_2x2.csv" \
    --results-out "${shard_dir}/aif_plus_results.csv" \
    --window-errors-out "${shard_dir}/aif_plus_window_errors.csv" \
    --arg-out "${shard_dir}/aif_plus_artifact_reliance_gap.csv" \
    --wgr-out "${shard_dir}/aif_plus_worst_group_risk.csv" \
    --ri-out "${shard_dir}/aif_plus_ranking_instability.csv" \
    --report-out "${shard_dir}/aif_plus_summary.md"
done

if ! plan_wait_bg; then
  plan_log "one or more AIF-Plus shards failed; skip merge"
  exit 1
fi

plan_log "merge AIF-Plus shard outputs"
PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
  "${PLAN_AIF_PLUS_TMP_DIR}" \
  "${RESULTS_DIR_ABS}" \
  "${REPORTS_DIR_ABS}/aif_plus_summary.md" \
  "${RESULTS_DIR_ABS}/counterfactual_2x2.csv" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cli.run_aif_plus import build_summary_markdown
from data import write_markdown
from experiments.aif_shared import compute_aif_arg_table, compute_aif_ri_table, compute_aif_wgr_table


tmp_dir = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
report_out = Path(sys.argv[3])
baseline_path = Path(sys.argv[4])
results_dir.mkdir(parents=True, exist_ok=True)
report_out.parent.mkdir(parents=True, exist_ok=True)


def read_many(name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shard_dir in sorted(tmp_dir.glob("shard_*")):
        path = shard_dir / name
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


results_df = read_many("aif_plus_results.csv")
window_errors_df = read_many("aif_plus_window_errors.csv")
if not results_df.empty:
    results_df = results_df.sort_values(
        ["dataset_name", "backbone", "lookback", "horizon", "train_view_name", "eval_view_name", "seed"]
    ).reset_index(drop=True)
if not window_errors_df.empty:
    window_errors_df = window_errors_df.sort_values(
        ["dataset_name", "backbone", "lookback", "horizon", "train_view_name", "eval_view_name", "window_id"]
    ).reset_index(drop=True)

arg_df = compute_aif_arg_table(results_df)
wgr_df = compute_aif_wgr_table(window_errors_df)
ri_df = compute_aif_ri_table(results_df)

results_df.to_csv(results_dir / "aif_plus_results.csv", index=False)
window_errors_df.to_csv(results_dir / "aif_plus_window_errors.csv", index=False)
arg_df.to_csv(results_dir / "aif_plus_artifact_reliance_gap.csv", index=False)
wgr_df.to_csv(results_dir / "aif_plus_worst_group_risk.csv", index=False)
ri_df.to_csv(results_dir / "aif_plus_ranking_instability.csv", index=False)

baseline_df = pd.read_csv(baseline_path, low_memory=False) if baseline_path.exists() else pd.DataFrame()
write_markdown(report_out, build_summary_markdown(results_df, baseline_df, arg_df, wgr_df, ri_df))
PY

plan_log "done"

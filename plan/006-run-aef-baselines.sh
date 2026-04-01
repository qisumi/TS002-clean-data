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
  plan_log "no GPUs detected; AEF baselines require at least one GPU"
  exit 1
fi

PLAN_AEF_DATASETS=("${PLAN_DATASETS_ARR[@]}")
PLAN_AEF_WORKER_COUNT="${#PLAN_GPU_IDS_ARR[@]}"
if (( PLAN_AEF_WORKER_COUNT > ${#PLAN_AEF_DATASETS[@]} )); then
  PLAN_AEF_WORKER_COUNT="${#PLAN_AEF_DATASETS[@]}"
fi
if (( PLAN_AEF_WORKER_COUNT <= 0 )); then
  plan_log "no datasets selected for AEF baselines"
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

PLAN_AEF_THREADS="$(resolve_plan_worker_threads "${PLAN_AEF_WORKER_COUNT}")"
PLAN_AEF_NUM_WORKERS="$(resolve_plan_dataloader_workers)"
PLAN_AEF_TMP_DIR="$(plan_make_temp_dir "006-aef-baselines")"
trap 'cleanup_plan_temp_dir "${PLAN_AEF_TMP_DIR}"' EXIT

partition_items_round_robin PLAN_AEF_DATASETS "${PLAN_AEF_WORKER_COUNT}" PLAN_AEF_DATASET_GROUPS

plan_log "datasets=$(join_by_comma "${PLAN_AEF_DATASETS[@]}")"
plan_log "gpus=$(join_by_comma "${PLAN_GPU_IDS_ARR[@]}")"
plan_log "worker_count=${PLAN_AEF_WORKER_COUNT} threads_per_worker=${PLAN_AEF_THREADS} dataloader_workers=${PLAN_AEF_NUM_WORKERS}"

plan_bg_reset
for ((worker_idx = 0; worker_idx < PLAN_AEF_WORKER_COUNT; worker_idx++)); do
  dataset_group="${PLAN_AEF_DATASET_GROUPS[worker_idx]}"
  if [[ -z "${dataset_group}" ]]; then
    continue
  fi
  gpu_id="${PLAN_GPU_IDS_ARR[worker_idx]}"
  shard_dir="${PLAN_AEF_TMP_DIR}/shard_${worker_idx}"
  shard_config="${PLAN_AEF_TMP_DIR}/aef_shard_${worker_idx}.yaml"
  mkdir -p "${shard_dir}"
  write_dataset_shard_config "${AEF_CONFIG}" "${shard_config}" "${dataset_group}" "${PLAN_AEF_NUM_WORKERS}"
  run_module_on_gpu_bg \
    "aef-baselines-shard-${worker_idx}" \
    "${gpu_id}" \
    "${PLAN_AEF_THREADS}" \
    "$(plan_child_log_path "006-aef-baselines-shard-${worker_idx}")" \
    cli.run_aef_baselines \
    --config "${shard_config}" \
    --views-dir "${STATS_DIR}/window_views" \
    --registry "${STATS_DIR}/dataset_registry.csv" \
    --results-dir "${shard_dir}" \
    --report-out "${shard_dir}/aef_summary.md" \
    --baseline-results "${RESULTS_DIR}/counterfactual_2x2.csv"
done

plan_wait_bg

plan_log "merge AEF baseline shard outputs"
PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
  "${PLAN_AEF_TMP_DIR}" \
  "${RESULTS_DIR_ABS}" \
  "${REPORTS_DIR_ABS}/aef_summary.md" \
  "${RESULTS_DIR_ABS}/counterfactual_2x2.csv" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cli.run_aef_baselines import build_summary_markdown
from data import write_markdown


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


weak_df = read_many("aef_results.csv")
control_df = read_many("aef_control_results.csv")
weak_errors_df = read_many("aef_window_errors.csv")
control_errors_df = read_many("aef_control_window_errors.csv")

sort_cols = ["dataset_name", "lookback", "horizon", "train_view_name", "eval_view_name", "model_name"]
if not weak_df.empty:
    weak_df = weak_df.sort_values(sort_cols).reset_index(drop=True)
if not control_df.empty:
    control_df = control_df.sort_values(sort_cols).reset_index(drop=True)

weak_df.to_csv(results_dir / "aef_results.csv", index=False)
control_df.to_csv(results_dir / "aef_control_results.csv", index=False)
weak_errors_df.to_csv(results_dir / "aef_window_errors.csv", index=False)
control_errors_df.to_csv(results_dir / "aef_control_window_errors.csv", index=False)

baseline_df = pd.read_csv(baseline_path, low_memory=False) if baseline_path.exists() else pd.DataFrame()
write_markdown(report_out, build_summary_markdown(weak_df, control_df, baseline_df))
PY

plan_log "done"

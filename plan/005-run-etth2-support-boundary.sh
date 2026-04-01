#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_arrays
load_plan_gpu_arrays

if ! has_dataset "ETTh2" "${PLAN_DATASETS_ARR[@]}"; then
  plan_log "ETTh2 not in the current plan dataset list; skip"
  exit 0
fi

if [[ "${#PLAN_GPU_IDS_ARR[@]}" -eq 0 ]]; then
  plan_log "no GPUs detected; ETTh2 auxiliary eval requires at least one GPU"
  exit 1
fi

PLAN_ETTH2_GPU_ID="${PLAN_GPU_IDS_ARR[0]}"
PLAN_ETTH2_THREADS="$(resolve_plan_worker_threads 1)"
PLAN_ETTH2_NUM_WORKERS="$(resolve_plan_dataloader_workers)"
PLAN_ETTH2_TMP_DIR="$(plan_make_temp_dir "005-etth2")"
trap 'cleanup_plan_temp_dir "${PLAN_ETTH2_TMP_DIR}"' EXIT

COUNTERFACTUAL_CONFIG_RESOLVED="${PLAN_ETTH2_TMP_DIR}/counterfactual_eval.yaml"
write_dataset_shard_config \
  "${COUNTERFACTUAL_CONFIG}" \
  "${COUNTERFACTUAL_CONFIG_RESOLVED}" \
  "ETTh2" \
  "${PLAN_ETTH2_NUM_WORKERS}"

plan_log "gpu=${PLAN_ETTH2_GPU_ID} threads=${PLAN_ETTH2_THREADS} dataloader_workers=${PLAN_ETTH2_NUM_WORKERS}"

run_module \
  cli.build_etth2_channel_support \
  --events "${STATS_DIR}/final_artifact_events.csv" \
  --views-dir "${STATS_DIR}/window_views" \
  --out "${RESULTS_DIR}/etth2_channel_support.csv" \
  --lookbacks "$(join_by_comma "${PLAN_LOOKBACKS_ARR[@]}")" \
  --horizons "$(join_by_comma "${PLAN_HORIZONS_ARR[@]}")"

run_module_on_gpu \
  "${PLAN_ETTH2_GPU_ID}" \
  "${PLAN_ETTH2_THREADS}" \
  cli.run_etth2_variable_stratified_eval \
  --config "${COUNTERFACTUAL_CONFIG_RESOLVED}" \
  --registry "${STATS_DIR}/dataset_registry.csv" \
  --events "${STATS_DIR}/final_artifact_events.csv" \
  --views-dir "${STATS_DIR}/window_views" \
  --support-csv "${RESULTS_DIR}/etth2_channel_support.csv" \
  --results-out "${RESULTS_DIR}/etth2_variable_stratified_eval.csv" \
  --report-out "${REPORTS_DIR}/etth2_variable_stratified_eval.md"

plan_log "done"

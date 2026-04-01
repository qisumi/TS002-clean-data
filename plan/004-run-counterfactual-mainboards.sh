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
  plan_log "no GPUs detected; counterfactual run requires at least one GPU"
  exit 1
fi

PLAN_COUNTERFACTUAL_THREADS="$(resolve_plan_worker_threads "${#PLAN_GPU_IDS_ARR[@]}")"
PLAN_COUNTERFACTUAL_NUM_WORKERS="$(resolve_plan_dataloader_workers)"
PLAN_COUNTERFACTUAL_TMP_DIR="$(plan_make_temp_dir "004-counterfactual")"
trap 'cleanup_plan_temp_dir "${PLAN_COUNTERFACTUAL_TMP_DIR}"' EXIT

PLAN_DATASETS_CSV="$(join_by_comma "${PLAN_DATASETS_ARR[@]}")"
COUNTERFACTUAL_CONFIG_RESOLVED="${PLAN_COUNTERFACTUAL_TMP_DIR}/counterfactual_eval.yaml"
COUNTERFACTUAL_MANIFEST_PATH="${PLAN_COUNTERFACTUAL_TMP_DIR}/counterfactual_manifest.jsonl"

plan_log "datasets=${PLAN_DATASETS_CSV}"
plan_log "gpus=$(join_by_comma "${PLAN_GPU_IDS_ARR[@]}")"
plan_log "threads_per_worker=${PLAN_COUNTERFACTUAL_THREADS}"
plan_log "dataloader_workers_per_proc=${PLAN_COUNTERFACTUAL_NUM_WORKERS}"

write_dataset_shard_config \
  "${COUNTERFACTUAL_CONFIG}" \
  "${COUNTERFACTUAL_CONFIG_RESOLVED}" \
  "${PLAN_DATASETS_CSV}" \
  "${PLAN_COUNTERFACTUAL_NUM_WORKERS}"

run_module \
  cli.run_counterfactual_eval \
  --config "${COUNTERFACTUAL_CONFIG_RESOLVED}" \
  --manifest-out "${COUNTERFACTUAL_MANIFEST_PATH}" \
  --build-manifest-only

plan_bg_reset
for shard_id in "${!PLAN_GPU_IDS_ARR[@]}"; do
  gpu_id="${PLAN_GPU_IDS_ARR[shard_id]}"
  run_module_on_gpu_bg \
    "counterfactual-shard-${shard_id}" \
    "${gpu_id}" \
    "${PLAN_COUNTERFACTUAL_THREADS}" \
    "$(plan_child_log_path "004-counterfactual-shard-${shard_id}")" \
    cli.run_counterfactual_eval \
    --config "${COUNTERFACTUAL_CONFIG_RESOLVED}" \
    --manifest "${COUNTERFACTUAL_MANIFEST_PATH}" \
    --views-dir "${STATS_DIR}/window_views" \
    --view-manifest "${STATS_DIR}/eval_view_manifest.csv" \
    --events "${STATS_DIR}/final_artifact_events.csv" \
    --registry "${STATS_DIR}/dataset_registry.csv" \
    --results-dir "${RESULTS_DIR}" \
    --report-out "${REPORTS_DIR}/counterfactual_eval_summary.md" \
    --setting-logs-dir "${LOGS_DIR}/counterfactual_eval_settings/shard_${shard_id}" \
    --shard-id "${shard_id}" \
    --num-shards "${#PLAN_GPU_IDS_ARR[@]}" \
    --skip-merge
done

plan_wait_bg

run_module \
  cli.run_counterfactual_eval \
  --config "${COUNTERFACTUAL_CONFIG_RESOLVED}" \
  --manifest "${COUNTERFACTUAL_MANIFEST_PATH}" \
  --views-dir "${STATS_DIR}/window_views" \
  --view-manifest "${STATS_DIR}/eval_view_manifest.csv" \
  --events "${STATS_DIR}/final_artifact_events.csv" \
  --registry "${STATS_DIR}/dataset_registry.csv" \
  --results-dir "${RESULTS_DIR}" \
  --report-out "${REPORTS_DIR}/counterfactual_eval_summary.md" \
  --setting-logs-dir "${LOGS_DIR}/counterfactual_eval_settings" \
  --merge-only

plan_log "done"

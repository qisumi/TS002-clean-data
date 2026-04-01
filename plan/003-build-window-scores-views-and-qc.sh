#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_arrays

plan_log "datasets=$(join_by_comma "${PLAN_DATASETS_ARR[@]}")"
plan_log "lookbacks=$(join_by_comma "${PLAN_LOOKBACKS_ARR[@]}")"
plan_log "horizons=$(join_by_comma "${PLAN_HORIZONS_ARR[@]}")"

plan_bg_reset
for lookback in "${PLAN_LOOKBACKS_ARR[@]}"; do
  run_module_bg \
    "window-scores-L${lookback}" \
    "$(plan_child_log_path "003-window-scores-L${lookback}")" \
    cli.build_window_scores \
    --registry "${STATS_DIR}/dataset_registry.csv" \
    --events "${STATS_DIR}/final_artifact_events.csv" \
    --datasets "${PLAN_DATASETS_ARR[@]}" \
    --lookback "${lookback}" \
    --horizons "${PLAN_HORIZONS_ARR[@]}" \
    --out-dir "${STATS_DIR}/window_scores" \
    --spec "${VIEW_SPEC_CONFIG}"
done
plan_wait_bg

run_module \
  cli.build_eval_views \
  --spec "${VIEW_SPEC_CONFIG}" \
  --scores-dir "${STATS_DIR}/window_scores" \
  --out-dir "${STATS_DIR}/window_views" \
  --events "${STATS_DIR}/final_artifact_events.csv" \
  --manifest-out "${STATS_DIR}/eval_view_manifest.csv" \
  --report-out "${REPORTS_DIR}/eval_view_design.md" \
  --datasets "${PLAN_DATASETS_ARR[@]}" \
  --lookbacks "${PLAN_LOOKBACKS_ARR[@]}" \
  --horizons "${PLAN_HORIZONS_ARR[@]}"

run_module \
  cli.build_clean_view_qc \
  --events "${STATS_DIR}/final_artifact_events.csv" \
  --views-dir "${STATS_DIR}/window_views" \
  --out-md "${REPORTS_DIR}/clean_view_qc_report.md" \
  --out-csv "${REPORTS_DIR}/clean_view_support_summary.csv" \
  --datasets "${PLAN_DATASETS_ARR[@]}" \
  --lookbacks "${PLAN_LOOKBACKS_ARR[@]}" \
  --horizons "${PLAN_HORIZONS_ARR[@]}"

plan_log "done"

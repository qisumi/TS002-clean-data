#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_arrays

run_module \
  cli.build_011_baseline_diagnosis \
  --views-dir "${STATS_DIR}/window_views" \
  --manifest "${STATS_DIR}/eval_view_manifest.csv" \
  --results-dir "${RESULTS_DIR}" \
  --out "${REPORTS_DIR}/011_step0_baseline_diagnosis.md" \
  --lookbacks "${PLAN_LOOKBACKS_ARR[@]}"

run_module \
  cli.build_unified_leaderboard_appendix \
  --results-dir "${RESULTS_DIR}" \
  --reports-dir "${REPORTS_DIR}" \
  --support-summary "${REPORTS_DIR}/clean_view_support_summary.csv"

run_module \
  cli.build_handoff_reports \
  --results-dir "${RESULTS_DIR}" \
  --reports-dir "${REPORTS_DIR}" \
  --stats-dir "${STATS_DIR}"

if [[ "${RUN_ORGANIZE_HANDOFF:-1}" == "1" ]]; then
  ORGANIZE_ARGS=(cli.organize_handoff_outputs --manifest-out "${CLEANUP_MANIFEST_OUT}")
  if [[ "${CLEAN_EXISTING_HANDOFF:-0}" == "1" ]]; then
    ORGANIZE_ARGS+=(--clean-existing)
  fi
  run_module "${ORGANIZE_ARGS[@]}"
else
  plan_log "RUN_ORGANIZE_HANDOFF=0; skip organize_handoff_outputs.py"
fi

plan_log "done"

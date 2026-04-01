#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_arrays

PLAN_DATASETS_CSV="$(join_by_comma "${PLAN_DATASETS_ARR[@]}")"
plan_log "datasets=${PLAN_DATASETS_CSV}"

run_module cli.detect_artifacts --datasets "${PLAN_DATASETS_CSV}"
run_module cli.review_artifacts --datasets "${PLAN_DATASETS_CSV}"
run_module \
  cli.merge_candidates_to_events \
  --stats-dir "${STATS_DIR}" \
  --datasets "${PLAN_DATASETS_ARR[@]}" \
  --out-csv "${STATS_DIR}/final_artifact_events.csv" \
  --out-md "${STATS_DIR}/final_artifact_events.md"
run_module \
  cli.render_event_merge_qa \
  --events "${STATS_DIR}/final_artifact_events.csv" \
  --out-md "${REPORTS_DIR}/event_merge_visual_qa.md" \
  --n-per-dataset "${EVENT_QA_PER_DATASET}"

plan_log "done"

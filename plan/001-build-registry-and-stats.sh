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

run_module cli.build_dataset_registry --datasets "${PLAN_DATASETS_CSV}"
run_module cli.generate_dataset_statistics --datasets "${PLAN_DATASETS_CSV}"

plan_log "done"

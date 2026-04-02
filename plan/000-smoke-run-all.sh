#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_gpu_arrays

SMOKE_DATASET="${SMOKE_DATASET:-ETTh2}"
SMOKE_HORIZON="${SMOKE_HORIZON:-96}"
SMOKE_LOOKBACK="${SMOKE_LOOKBACK:-96}"
SMOKE_SEED="${SMOKE_SEED:-0}"
SMOKE_BACKBONE="${SMOKE_BACKBONE:-DLinear}"
SMOKE_RESULTS_ROOT="${SMOKE_RESULTS_ROOT:-smoke_runs}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-full_pipeline_$(date '+%Y%m%d_%H%M%S')}"
SMOKE_RUN_ROOT="${ROOT_DIR}/${SMOKE_RESULTS_ROOT}/${SMOKE_RUN_NAME}"
SMOKE_CONFIG_DIR="${SMOKE_RUN_ROOT}/configs"
SMOKE_PLAN_LOG_DIR="${SMOKE_RUN_ROOT}/logs/plan"
SMOKE_STEP_LOG_DIR="${SMOKE_RUN_ROOT}/logs/runtime"
SMOKE_RESULTS_DIR="${SMOKE_RUN_ROOT}/results"
SMOKE_REPORTS_DIR="${SMOKE_RUN_ROOT}/reports"
SMOKE_STATS_DIR="${SMOKE_RUN_ROOT}/statistic_results"
SMOKE_TEMP_ROOT="${SMOKE_RUN_ROOT}/temp"
SMOKE_KEEP_TEMP="${SMOKE_KEEP_TEMP:-1}"
SMOKE_GPU_COUNT="${SMOKE_GPU_COUNT:-2}"

mkdir -p \
  "${SMOKE_CONFIG_DIR}" \
  "${SMOKE_PLAN_LOG_DIR}" \
  "${SMOKE_STEP_LOG_DIR}" \
  "${SMOKE_RESULTS_DIR}" \
  "${SMOKE_REPORTS_DIR}" \
  "${SMOKE_STATS_DIR}" \
  "${SMOKE_TEMP_ROOT}"

if [[ "${#PLAN_GPU_IDS_ARR[@]}" -eq 0 ]]; then
  plan_log "no GPU detected; smoke run requires at least one CUDA device"
  exit 1
fi

SMOKE_SELECTED_GPUS=()
for gpu_id in "${PLAN_GPU_IDS_ARR[@]}"; do
  SMOKE_SELECTED_GPUS+=("${gpu_id}")
  if (( ${#SMOKE_SELECTED_GPUS[@]} >= SMOKE_GPU_COUNT )); then
    break
  fi
done
if (( ${#SMOKE_SELECTED_GPUS[@]} == 0 )); then
  SMOKE_SELECTED_GPUS=("${PLAN_GPU_IDS_ARR[0]}")
fi

plan_log "smoke_root=${SMOKE_RUN_ROOT}"
plan_log "dataset=${SMOKE_DATASET} lookback=${SMOKE_LOOKBACK} horizon=${SMOKE_HORIZON} seed=${SMOKE_SEED}"
plan_log "backbone=${SMOKE_BACKBONE}"
plan_log "gpus=$(join_by_comma "${SMOKE_SELECTED_GPUS[@]}")"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
  "${ROOT_DIR}/configs/counterfactual_eval.yaml" \
  "${ROOT_DIR}/configs/aef.yaml" \
  "${ROOT_DIR}/configs/aef_plus.yaml" \
  "${ROOT_DIR}/configs/aif_plus.yaml" \
  "${SMOKE_CONFIG_DIR}" \
  "${SMOKE_DATASET}" \
  "${SMOKE_HORIZON}" \
  "${SMOKE_LOOKBACK}" \
  "${SMOKE_SEED}" \
  "${SMOKE_BACKBONE}" <<'PY'
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml


counterfactual_base = Path(sys.argv[1])
aef_base = Path(sys.argv[2])
aef_plus_base = Path(sys.argv[3])
aif_plus_base = Path(sys.argv[4])
out_dir = Path(sys.argv[5])
dataset_name = sys.argv[6]
horizon = int(sys.argv[7])
lookback = int(sys.argv[8])
seed = int(sys.argv[9])
backbone_name = sys.argv[10]

out_dir.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


cf = load_yaml(counterfactual_base)
cf_defaults = cf.setdefault("defaults", {})
cf_defaults["datasets"] = [dataset_name]
cf_defaults["horizons"] = [horizon]
cf_defaults["seeds"] = [seed]
cf_defaults["lookback"] = lookback
cf_defaults["backbones"] = [
    item for item in cf_defaults.get("backbones", []) if str(item.get("name", "")) == backbone_name
]
runtime = cf_defaults.setdefault("runtime", {})
runtime["epochs"] = 1
runtime["patience"] = 1
runtime["batch_size"] = min(int(runtime.get("batch_size", 64)), 16)
runtime["eval_batch_size"] = min(int(runtime.get("eval_batch_size", 128)), 32)
runtime["num_workers"] = 0
runtime["max_train_windows"] = 128
runtime["max_val_windows"] = 32
runtime["max_test_windows"] = 32
dump_yaml(out_dir / "counterfactual_eval.smoke.yaml", cf)


def patch_eval_views(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    defaults = cfg.setdefault("defaults", {})
    defaults["datasets"] = [dataset_name]
    defaults["horizons"] = [horizon]
    defaults["seeds"] = [seed]
    defaults["lookback"] = lookback
    eval_views = defaults.get("eval_views")
    if isinstance(eval_views, dict):
        defaults["eval_views"] = {dataset_name: eval_views.get(dataset_name, ["raw", "clean_like", "intervened"])}
    return cfg


aef = patch_eval_views(load_yaml(aef_base))
aef_runtime = aef.setdefault("defaults", {}).setdefault("runtime", {})
aef_runtime["epochs"] = 1
aef_runtime["patience"] = 1
aef_runtime["batch_size"] = min(int(aef_runtime.get("batch_size", 128)), 16)
aef_runtime["eval_batch_size"] = min(int(aef_runtime.get("eval_batch_size", 256)), 32)
aef_runtime["num_workers"] = 0
aef_runtime["max_train_windows"] = 128
aef_runtime["max_val_windows"] = 32
aef_runtime["max_test_windows"] = 32
dump_yaml(out_dir / "aef.smoke.yaml", aef)

aef_plus = patch_eval_views(load_yaml(aef_plus_base))
aef_plus.setdefault("defaults", {})["use_dataset_presets"] = False
aef_plus_runtime = aef_plus.setdefault("defaults", {}).setdefault("runtime", {})
aef_plus_runtime["epochs"] = 1
aef_plus_runtime["patience"] = 1
aef_plus_runtime["batch_size"] = min(int(aef_plus_runtime.get("batch_size", 64)), 8)
aef_plus_runtime["eval_batch_size"] = min(int(aef_plus_runtime.get("eval_batch_size", 128)), 16)
aef_plus_runtime["num_workers"] = 0
aef_plus_runtime["max_train_windows"] = 96
aef_plus_runtime["max_val_windows"] = 24
aef_plus_runtime["max_test_windows"] = 24
aef_plus_runtime["use_ema"] = False
dump_yaml(out_dir / "aef_plus.smoke.yaml", aef_plus)

aif_plus = load_yaml(aif_plus_base)
aif_defaults = aif_plus.setdefault("defaults", {})
aif_defaults["use_dataset_presets"] = False
aif_defaults["datasets"] = [dataset_name]
aif_defaults["horizons"] = [horizon]
aif_defaults["seeds"] = [seed]
aif_defaults["lookback"] = lookback
view_alias = aif_defaults.get("view_alias")
if isinstance(view_alias, dict):
    aif_defaults["view_alias"] = {dataset_name: view_alias.get(dataset_name, {"clean_like": "clean_like"})}
aif_runtime = aif_defaults.setdefault("runtime", {})
aif_runtime["patience"] = 1
aif_runtime["batch_size"] = min(int(aif_runtime.get("batch_size", 48)), 8)
aif_runtime["eval_batch_size"] = min(int(aif_runtime.get("eval_batch_size", 96)), 16)
aif_runtime["num_workers"] = 0
aif_runtime["max_train_windows"] = 96
aif_runtime["max_val_windows"] = 24
aif_runtime["max_test_windows"] = 24
aif_runtime["use_ema"] = False
stages = aif_defaults.setdefault("stages", {})
for stage_name in ["stage_a", "stage_b", "stage_c"]:
    stage_cfg = stages.setdefault(stage_name, {})
    stage_cfg["epochs"] = 1
    stage_cfg["patience"] = 1
dump_yaml(out_dir / "aif_plus.smoke.yaml", aif_plus)
PY

env \
  CONDA_ENV_NAME="${CONDA_ENV_NAME}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  PLAN_DATASETS="${SMOKE_DATASET}" \
  PLAN_HORIZONS="${SMOKE_HORIZON}" \
  PLAN_LOOKBACKS="${SMOKE_LOOKBACK}" \
  PLAN_GPU_IDS="$(join_by_comma "${SMOKE_SELECTED_GPUS[@]}")" \
  PLAN_LOG_DIR="${SMOKE_PLAN_LOG_DIR}" \
  PLAN_TEMP_ROOT="${SMOKE_TEMP_ROOT}" \
  PLAN_KEEP_TEMP="${SMOKE_KEEP_TEMP}" \
  RESULTS_DIR="${SMOKE_RESULTS_DIR}" \
  REPORTS_DIR="${SMOKE_REPORTS_DIR}" \
  STATS_DIR="${SMOKE_STATS_DIR}" \
  LOGS_DIR="${SMOKE_STEP_LOG_DIR}" \
  COUNTERFACTUAL_CONFIG="${SMOKE_CONFIG_DIR}/counterfactual_eval.smoke.yaml" \
  AEF_CONFIG="${SMOKE_CONFIG_DIR}/aef.smoke.yaml" \
  AEF_PLUS_CONFIG="${SMOKE_CONFIG_DIR}/aef_plus.smoke.yaml" \
  AIF_PLUS_CONFIG="${SMOKE_CONFIG_DIR}/aif_plus.smoke.yaml" \
  RUN_ORGANIZE_HANDOFF=0 \
  EVENT_QA_PER_DATASET=1 \
  bash "${PLAN_DIR}/010-run-all.sh"

REQUIRED_PATHS=(
  "${SMOKE_STATS_DIR}/dataset_registry.csv"
  "${SMOKE_STATS_DIR}/final_artifact_events.csv"
  "${SMOKE_STATS_DIR}/eval_view_manifest.csv"
  "${SMOKE_RESULTS_DIR}/counterfactual_2x2.csv"
  "${SMOKE_RESULTS_DIR}/etth2_channel_support.csv"
  "${SMOKE_RESULTS_DIR}/aef_results.csv"
  "${SMOKE_RESULTS_DIR}/aef_plus_results.csv"
  "${SMOKE_RESULTS_DIR}/aif_plus_results.csv"
  "${SMOKE_REPORTS_DIR}/counterfactual_eval_summary.md"
  "${SMOKE_REPORTS_DIR}/unified_leaderboard_appendix.md"
)

missing_count=0
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [[ -e "${required_path}" ]]; then
    plan_log "ok ${required_path}"
  else
    plan_log "missing ${required_path}"
    missing_count=$(( missing_count + 1 ))
  fi
done

if (( missing_count > 0 )); then
  plan_log "smoke failed: missing_count=${missing_count}"
  exit 1
fi

cat <<EOF

Smoke run completed.
root: ${SMOKE_RUN_ROOT}
plan logs: ${SMOKE_PLAN_LOG_DIR}
runtime logs: ${SMOKE_STEP_LOG_DIR}
results: ${SMOKE_RESULTS_DIR}
reports: ${SMOKE_REPORTS_DIR}
stats: ${SMOKE_STATS_DIR}

EOF

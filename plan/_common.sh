#!/usr/bin/env bash

if [[ -n "${PLAN_COMMON_SH_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
PLAN_COMMON_SH_LOADED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN_DIR="${ROOT_DIR}/plan"

PLAN_DATASETS_DEFAULT=(ETTh1 ETTh2 ETTm1 ETTm2 solar_AL weather exchange_rate electricity)
PLAN_HORIZONS_DEFAULT=(96 192 336 720)
PLAN_LOOKBACKS_DEFAULT=(96)

CONDA_ENV_NAME="${CONDA_ENV_NAME:-zzq}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-results}"
REPORTS_DIR="${REPORTS_DIR:-reports}"
STATS_DIR="${STATS_DIR:-statistic_results}"
LOGS_DIR="${LOGS_DIR:-logs}"
COUNTERFACTUAL_CONFIG="${COUNTERFACTUAL_CONFIG:-configs/counterfactual_eval.yaml}"
AEF_CONFIG="${AEF_CONFIG:-configs/aef.yaml}"
AEF_PLUS_CONFIG="${AEF_PLUS_CONFIG:-configs/aef_plus.yaml}"
AIF_PLUS_CONFIG="${AIF_PLUS_CONFIG:-configs/aif_plus.yaml}"
VIEW_SPEC_CONFIG="${VIEW_SPEC_CONFIG:-configs/view_specs.yaml}"
CLEANUP_MANIFEST_OUT="${CLEANUP_MANIFEST_OUT:-cleanup_manifest.md}"
EVENT_QA_PER_DATASET="${EVENT_QA_PER_DATASET:-3}"

plan_log() {
  printf '[%s] [%s] %s\n' "$(date '+%H:%M:%S')" "${PLAN_STEP_NAME:-$(basename "$0")}" "$*"
}

ensure_root_dir() {
  cd "${ROOT_DIR}"
}

find_conda_bin() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  local candidate
  for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/home/amax/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

ensure_conda_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME}" ]]; then
    return 0
  fi

  local conda_bin
  conda_bin="$(find_conda_bin || true)"
  if [[ -z "${conda_bin}" ]]; then
    plan_log "conda not found; cannot activate ${CONDA_ENV_NAME}"
    return 1
  fi

  local conda_base
  conda_base="$("${conda_bin}" info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"

  if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    plan_log "failed to activate conda env ${CONDA_ENV_NAME}"
    return 1
  fi

  plan_log "activated conda env ${CONDA_ENV_NAME}"
}

load_plan_arrays() {
  if [[ -n "${PLAN_DATASETS:-}" ]]; then
    split_csv_to_array "${PLAN_DATASETS}" PLAN_DATASETS_ARR
  else
    PLAN_DATASETS_ARR=("${PLAN_DATASETS_DEFAULT[@]}")
  fi
  if [[ -n "${PLAN_HORIZONS:-}" ]]; then
    split_csv_to_array "${PLAN_HORIZONS}" PLAN_HORIZONS_ARR
  else
    PLAN_HORIZONS_ARR=("${PLAN_HORIZONS_DEFAULT[@]}")
  fi
  if [[ -n "${PLAN_LOOKBACKS:-}" ]]; then
    split_csv_to_array "${PLAN_LOOKBACKS}" PLAN_LOOKBACKS_ARR
  else
    PLAN_LOOKBACKS_ARR=("${PLAN_LOOKBACKS_DEFAULT[@]}")
  fi
}

split_csv_to_array() {
  local raw_text="$1"
  local -n out_ref="$2"
  out_ref=()
  raw_text="${raw_text//,/ }"
  local token
  for token in ${raw_text}; do
    if [[ -n "${token}" ]]; then
      out_ref+=("${token}")
    fi
  done
}

load_plan_gpu_arrays() {
  if [[ -n "${PLAN_GPU_IDS:-}" ]]; then
    split_csv_to_array "${PLAN_GPU_IDS}" PLAN_GPU_IDS_ARR
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Respect scheduler-assigned GPU visibility instead of guessing host GPU indices.
    split_csv_to_array "${CUDA_VISIBLE_DEVICES}" PLAN_GPU_IDS_ARR
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t PLAN_GPU_IDS_ARR < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk 'NF {print $1}')
  else
    PLAN_GPU_IDS_ARR=()
  fi
}

resolve_plan_cpu_cores() {
  if [[ -n "${PLAN_CPU_CORES:-}" ]]; then
    printf '%s\n' "${PLAN_CPU_CORES}"
    return 0
  fi
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return 0
  fi
  printf '1\n'
}

resolve_plan_worker_threads() {
  local worker_count="${1:-1}"
  if (( worker_count <= 0 )); then
    worker_count=1
  fi
  if [[ -n "${PLAN_CPU_THREADS_PER_WORKER:-}" ]]; then
    printf '%s\n' "${PLAN_CPU_THREADS_PER_WORKER}"
    return 0
  fi
  local cpu_cores
  cpu_cores="$(resolve_plan_cpu_cores)"
  local threads=$(( cpu_cores / worker_count ))
  if (( threads < 1 )); then
    threads=1
  fi
  printf '%s\n' "${threads}"
}

resolve_plan_dataloader_workers() {
  if [[ -n "${PLAN_DATALOADER_WORKERS_PER_PROC:-}" ]]; then
    printf '%s\n' "${PLAN_DATALOADER_WORKERS_PER_PROC}"
    return 0
  fi
  printf '1\n'
}

plan_child_log_dir() {
  local base_dir="${PLAN_RUN_LOG_DIR:-${ROOT_DIR}/logs/newplan}"
  local child_dir="${base_dir}/children"
  mkdir -p "${child_dir}"
  printf '%s\n' "${child_dir}"
}

plan_child_log_path() {
  local child_name="$1"
  local child_dir
  child_dir="$(plan_child_log_dir)"
  printf '%s/%s.log\n' "${child_dir}" "${child_name}"
}

plan_make_temp_dir() {
  local prefix="$1"
  local temp_root="${PLAN_TEMP_ROOT:-${ROOT_DIR}/.plan_tmp}"
  mkdir -p "${temp_root}"
  mktemp -d "${temp_root}/${prefix}.XXXXXX"
}

cleanup_plan_temp_dir() {
  local temp_dir="$1"
  if [[ -z "${temp_dir}" || ! -d "${temp_dir}" ]]; then
    return 0
  fi
  if [[ "${PLAN_KEEP_TEMP:-0}" == "1" ]]; then
    plan_log "keep temp dir ${temp_dir}"
    return 0
  fi
  rm -rf "${temp_dir}"
}

partition_items_round_robin() {
  local -n items_ref="$1"
  local worker_count="$2"
  local -n groups_ref="$3"
  groups_ref=()
  if (( worker_count <= 0 )); then
    return 0
  fi
  local idx
  for ((idx = 0; idx < worker_count; idx++)); do
    groups_ref[idx]=""
  done
  for idx in "${!items_ref[@]}"; do
    local worker_idx=$(( idx % worker_count ))
    if [[ -n "${groups_ref[worker_idx]}" ]]; then
      groups_ref[worker_idx]+=","
    fi
    groups_ref[worker_idx]+="${items_ref[idx]}"
  done
}

write_dataset_shard_config() {
  local base_config="$1"
  local out_config="$2"
  local datasets_csv="$3"
  local dataloader_workers="$4"
  local plan_pythonpath="${ROOT_DIR}/src"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    plan_pythonpath="${plan_pythonpath}:${PYTHONPATH}"
  fi
  plan_log "write shard config ${out_config} datasets=${datasets_csv} num_workers=${dataloader_workers}"
  PYTHONPATH="${plan_pythonpath}" "${PYTHON_BIN}" - "${base_config}" "${out_config}" "${datasets_csv}" "${dataloader_workers}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml


base_config, out_config, datasets_csv, dataloader_workers = sys.argv[1:5]
config = yaml.safe_load(Path(base_config).read_text(encoding="utf-8")) or {}
defaults = config.setdefault("defaults", {})
defaults["datasets"] = [item.strip() for item in datasets_csv.split(",") if item.strip()]
runtime = defaults.setdefault("runtime", {})
runtime["num_workers"] = int(dataloader_workers)
out_path = Path(out_config)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

plan_bg_reset() {
  PLAN_BG_PIDS=()
  PLAN_BG_NAMES=()
  PLAN_BG_LOGS=()
}

run_module_on_gpu() {
  local gpu_id="$1"
  local cpu_threads="$2"
  local module_name="$3"
  shift 3
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  export OMP_NUM_THREADS="${cpu_threads}"
  export MKL_NUM_THREADS="${cpu_threads}"
  export OPENBLAS_NUM_THREADS="${cpu_threads}"
  export NUMEXPR_NUM_THREADS="${cpu_threads}"
  export VECLIB_MAXIMUM_THREADS="${cpu_threads}"
  export BLIS_NUM_THREADS="${cpu_threads}"
  run_module "${module_name}" "$@"
}

run_module_on_gpu_bg() {
  local job_name="$1"
  local gpu_id="$2"
  local cpu_threads="$3"
  local log_path="$4"
  local module_name="$5"
  shift 5

  mkdir -p "$(dirname "${log_path}")"
  plan_log "launch ${job_name} gpu=${gpu_id} threads=${cpu_threads} log=${log_path}"
  (
    PLAN_STEP_NAME="${PLAN_STEP_NAME}/${job_name}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export OMP_NUM_THREADS="${cpu_threads}"
    export MKL_NUM_THREADS="${cpu_threads}"
    export OPENBLAS_NUM_THREADS="${cpu_threads}"
    export NUMEXPR_NUM_THREADS="${cpu_threads}"
    export VECLIB_MAXIMUM_THREADS="${cpu_threads}"
    export BLIS_NUM_THREADS="${cpu_threads}"
    run_module "${module_name}" "$@"
  ) > >(tee "${log_path}") 2>&1 &

  PLAN_BG_PIDS+=("$!")
  PLAN_BG_NAMES+=("${job_name}")
  PLAN_BG_LOGS+=("${log_path}")
}

run_module_bg() {
  local job_name="$1"
  local log_path="$2"
  local module_name="$3"
  shift 3

  mkdir -p "$(dirname "${log_path}")"
  plan_log "launch ${job_name} log=${log_path}"
  (
    PLAN_STEP_NAME="${PLAN_STEP_NAME}/${job_name}"
    run_module "${module_name}" "$@"
  ) > >(tee "${log_path}") 2>&1 &

  PLAN_BG_PIDS+=("$!")
  PLAN_BG_NAMES+=("${job_name}")
  PLAN_BG_LOGS+=("${log_path}")
}

plan_wait_bg() {
  local failed=0
  local idx
  for idx in "${!PLAN_BG_PIDS[@]}"; do
    local pid="${PLAN_BG_PIDS[idx]}"
    local job_name="${PLAN_BG_NAMES[idx]}"
    local log_path="${PLAN_BG_LOGS[idx]}"
    if wait "${pid}"; then
      plan_log "completed ${job_name}"
    else
      plan_log "failed ${job_name}; see ${log_path}"
      failed=1
    fi
  done
  return "${failed}"
}

join_by_comma() {
  local IFS=","
  printf '%s' "$*"
}

has_dataset() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

run_py() {
  plan_log "run ${PYTHON_BIN} $*"
  "${PYTHON_BIN}" "$@"
}

run_module() {
  local module_name="$1"
  shift
  local plan_pythonpath="${ROOT_DIR}/src"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    plan_pythonpath="${plan_pythonpath}:${PYTHONPATH}"
  fi
  plan_log "run PYTHONPATH=${plan_pythonpath} ${PYTHON_BIN} -m ${module_name} $*"
  PYTHONPATH="${plan_pythonpath}" "${PYTHON_BIN}" -m "${module_name}" "$@"
}

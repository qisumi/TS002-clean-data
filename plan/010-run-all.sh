#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_gpu_arrays

PLAN_LOG_DIR="${PLAN_LOG_DIR:-logs/newplan}"
if [[ "${PLAN_LOG_DIR}" == /* ]]; then
  PLAN_LOG_DIR_ABS="${PLAN_LOG_DIR}"
else
  PLAN_LOG_DIR_ABS="${ROOT_DIR}/${PLAN_LOG_DIR}"
fi
mkdir -p "${PLAN_LOG_DIR_ABS}"

PLAN_FROM="${PLAN_FROM:-}"
PLAN_TO="${PLAN_TO:-}"
PLAN_ONLY="${PLAN_ONLY:-}"
PLAN_SKIP_DONE="${PLAN_SKIP_DONE:-0}"
export PLAN_RUN_LOG_DIR="${PLAN_LOG_DIR_ABS}"

STEP_SCRIPTS=(
  "001-build-registry-and-stats.sh"
  "002-detect-review-and-merge-events.sh"
  "003-build-window-scores-views-and-qc.sh"
  "004-run-counterfactual-mainboards.sh"
  "005-run-etth2-support-boundary.sh"
  "006-run-aef-baselines.sh"
  "007-run-aef-plus.sh"
  "008-run-aif-plus.sh"
  "009-build-final-reports.sh"
)

print_usage() {
  cat <<'EOF'
Usage:
  bash plan/010-run-all.sh [--from STEP] [--to STEP] [--only STEPS] [--skip-done]
  bash plan/010-run-all.sh [STEP]

Options:
  --from STEP      Start from the given step, e.g. 008 or 008-run-aif-plus.sh
  --to STEP        Stop at the given step
  --only STEPS     Run only the listed steps, comma-separated or space-separated
  --skip-done      Skip steps whose .ok marker already exists
  --list           Print available steps and exit
  -h, --help       Show this help message

Examples:
  bash plan/010-run-all.sh --from 008
  bash plan/010-run-all.sh 008
  bash plan/010-run-all.sh --only 008,009
  bash plan/010-run-all.sh --from 008 --skip-done

Environment variables PLAN_FROM / PLAN_TO / PLAN_ONLY / PLAN_SKIP_DONE are still supported.
EOF
}

list_steps() {
  local step_script
  for step_script in "${STEP_SCRIPTS[@]}"; do
    printf '%s\n' "${step_script}"
  done
}

normalize_step_list() {
  local raw_text="$1"
  raw_text="${raw_text//,/ }"
  printf '%s\n' "${raw_text}"
}

step_rank() {
  local token="$1"
  token="${token##*/}"
  token="${token%.sh}"
  token="${token%%-*}"
  [[ "${token}" =~ ^[0-9]+$ ]] || return 1
  printf '%d\n' "$((10#${token}))"
}

validate_step_token() {
  local label="$1"
  local token="$2"
  if ! step_rank "${token}" >/dev/null; then
    plan_log "invalid ${label}: ${token}"
    return 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --from)
        shift
        [[ $# -gt 0 ]] || { plan_log "--from requires a step"; return 1; }
        PLAN_FROM="$1"
        ;;
      --to)
        shift
        [[ $# -gt 0 ]] || { plan_log "--to requires a step"; return 1; }
        PLAN_TO="$1"
        ;;
      --only)
        shift
        [[ $# -gt 0 ]] || { plan_log "--only requires step list"; return 1; }
        PLAN_ONLY="$1"
        ;;
      --skip-done)
        PLAN_SKIP_DONE=1
        ;;
      --list)
        list_steps
        exit 0
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        plan_log "unknown argument: $1"
        print_usage >&2
        return 1
        ;;
      *)
        if [[ -n "${PLAN_FROM}" ]]; then
          plan_log "unexpected positional argument: $1"
          print_usage >&2
          return 1
        fi
        PLAN_FROM="$1"
        ;;
    esac
    shift
  done

  if [[ -n "${PLAN_FROM}" ]]; then
    validate_step_token "PLAN_FROM" "${PLAN_FROM}" || return 1
    PLAN_FROM="$(step_rank "${PLAN_FROM}")"
  fi
  if [[ -n "${PLAN_TO}" ]]; then
    validate_step_token "PLAN_TO" "${PLAN_TO}" || return 1
    PLAN_TO="$(step_rank "${PLAN_TO}")"
  fi
  if [[ -n "${PLAN_ONLY}" ]]; then
    local token
    for token in $(normalize_step_list "${PLAN_ONLY}"); do
      validate_step_token "PLAN_ONLY token" "${token}" || return 1
    done
  fi

  if [[ -n "${PLAN_FROM}" && -n "${PLAN_TO}" ]] && (( 10#${PLAN_FROM} > 10#${PLAN_TO} )); then
    plan_log "PLAN_FROM must be <= PLAN_TO"
    return 1
  fi
}

step_id_matches() {
  local step_id="$1"
  local token="$2"
  local step_rank_value
  local token_rank_value
  step_rank_value="$(step_rank "${step_id}")" || return 1
  token_rank_value="$(step_rank "${token}")" || return 1
  (( step_rank_value == token_rank_value ))
}

step_selected() {
  local step_id="$1"
  if [[ -n "${PLAN_ONLY}" ]]; then
    local token
    for token in $(normalize_step_list "${PLAN_ONLY}"); do
      if step_id_matches "${step_id}" "${token}"; then
        return 0
      fi
    done
    return 1
  fi

  if [[ -n "${PLAN_FROM}" ]] && (( 10#${step_id} < 10#${PLAN_FROM} )); then
    return 1
  fi
  if [[ -n "${PLAN_TO}" ]] && (( 10#${step_id} > 10#${PLAN_TO} )); then
    return 1
  fi
  return 0
}

parse_args "$@"

plan_log "logs=${PLAN_LOG_DIR_ABS}"
plan_log "conda_env=${CONDA_DEFAULT_ENV:-<unset>}"
plan_log "python=$(command -v "${PYTHON_BIN}")"
plan_log "gpus=$(join_by_comma "${PLAN_GPU_IDS_ARR[@]}")"
plan_log "cpu_cores=$(resolve_plan_cpu_cores)"
plan_log "cpu_threads_per_worker=${PLAN_CPU_THREADS_PER_WORKER:-auto}"
plan_log "dataloader_workers_per_proc=$(resolve_plan_dataloader_workers)"
plan_log "selectors from=${PLAN_FROM:-<unset>} to=${PLAN_TO:-<unset>} only=${PLAN_ONLY:-<unset>} skip_done=${PLAN_SKIP_DONE}"

for step_script in "${STEP_SCRIPTS[@]}"; do
  step_id="${step_script%%-*}"
  if ! step_selected "${step_id}"; then
    plan_log "skip ${step_script} due to PLAN_FROM/PLAN_TO/PLAN_ONLY"
    continue
  fi

  script_path="${PLAN_DIR}/${step_script}"
  log_path="${PLAN_LOG_DIR_ABS}/${step_script%.sh}.log"
  ok_path="${PLAN_LOG_DIR_ABS}/${step_script%.sh}.ok"
  fail_path="${PLAN_LOG_DIR_ABS}/${step_script%.sh}.failed"

  if [[ "${PLAN_SKIP_DONE}" == "1" ]] && [[ -f "${ok_path}" ]]; then
    plan_log "skip ${step_script} because ${ok_path##*/} exists"
    continue
  fi

  rm -f "${ok_path}" "${fail_path}"
  plan_log "start ${step_script}"
  plan_log "log -> ${log_path}"

  if PLAN_RUN_LOG_DIR="${PLAN_LOG_DIR_ABS}" bash "${script_path}" 2>&1 | tee "${log_path}"; then
    printf 'ok %s\n' "$(date '+%F %T')" > "${ok_path}"
    plan_log "done ${step_script}"
  else
    printf 'failed %s\n' "$(date '+%F %T')" > "${fail_path}"
    plan_log "failed ${step_script}; see ${log_path}"
    exit 1
  fi
done

plan_log "all selected steps completed"

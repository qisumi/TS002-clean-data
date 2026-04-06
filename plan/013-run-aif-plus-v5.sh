#!/usr/bin/env bash
set -euo pipefail

PLAN_STEP_NAME="$(basename "$0")"
# shellcheck source=plan/_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ensure_root_dir
ensure_conda_env
load_plan_arrays
load_plan_gpu_arrays

AIF_PLUS_V5_CONFIG="${AIF_PLUS_V5_CONFIG:-configs/aif_plus_v5.yaml}"
if [[ "${AIF_PLUS_V5_CONFIG}" == /* ]]; then
  AIF_PLUS_V5_CONFIG_ABS="${AIF_PLUS_V5_CONFIG}"
else
  AIF_PLUS_V5_CONFIG_ABS="${ROOT_DIR}/${AIF_PLUS_V5_CONFIG}"
fi

if [[ "${#PLAN_GPU_IDS_ARR[@]}" -eq 0 ]]; then
  plan_log "no GPUs detected; AIF-Plus-V5 requires at least one GPU"
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

SHARDS_DIR_ABS="${RESULTS_DIR_ABS}/shards"
ARTIFACTS_DIR_ABS="${ROOT_DIR}/artifacts"
mkdir -p "${SHARDS_DIR_ABS}" "${ARTIFACTS_DIR_ABS}" "${REPORTS_DIR_ABS}"

declare -a PLAN_AIF_V5_SHARDS_RAW=(
  "ETTh1,ETTh2"
  "ETTm1,ETTm2"
  "solar_AL,weather"
  "exchange_rate,electricity"
)

filter_shard_csv() {
  local shard_csv="$1"
  local out=()
  local items=()
  split_csv_to_array "${shard_csv}" items
  local item
  for item in "${items[@]}"; do
    if has_dataset "${item}" "${PLAN_DATASETS_ARR[@]}"; then
      out+=("${item}")
    fi
  done
  join_by_comma "${out[@]}"
}

PLAN_AIF_V5_SHARDS=()
for shard_csv in "${PLAN_AIF_V5_SHARDS_RAW[@]}"; do
  filtered_csv="$(filter_shard_csv "${shard_csv}")"
  if [[ -n "${filtered_csv}" ]]; then
    PLAN_AIF_V5_SHARDS+=("${filtered_csv}")
  fi
done

if [[ "${#PLAN_AIF_V5_SHARDS[@]}" -eq 0 ]]; then
  plan_log "no V5 shards remain after PLAN_DATASETS filtering"
  exit 1
fi

PLAN_AIF_V5_WORKER_COUNT="${#PLAN_GPU_IDS_ARR[@]}"
if (( PLAN_AIF_V5_WORKER_COUNT > ${#PLAN_AIF_V5_SHARDS[@]} )); then
  PLAN_AIF_V5_WORKER_COUNT="${#PLAN_AIF_V5_SHARDS[@]}"
fi
PLAN_AIF_V5_THREADS="$(resolve_plan_worker_threads "${PLAN_AIF_V5_WORKER_COUNT}")"

plan_log "config=${AIF_PLUS_V5_CONFIG}"
plan_log "visible_gpus=$(join_by_comma "${PLAN_GPU_IDS_ARR[@]}")"
plan_log "threads_per_worker=${PLAN_AIF_V5_THREADS}"
plan_log "active_shards=${#PLAN_AIF_V5_SHARDS[@]}"

HORIZONS_CSV="$(join_by_comma "${PLAN_HORIZONS_ARR[@]}")"
SEEDS_CSV="${AIF_PLUS_V5_SEEDS:-0}"
RESULT_SHARD_PATHS=()
WINDOW_SHARD_PATHS=()
CHECKPOINT_SHARD_PATHS=()

for shard_idx in "${!PLAN_AIF_V5_SHARDS[@]}"; do
  rm -f \
    "${SHARDS_DIR_ABS}/aif_plus_v5.shard${shard_idx}_results_online.csv" \
    "${SHARDS_DIR_ABS}/aif_plus_v5.shard${shard_idx}_window_errors.csv" \
    "${SHARDS_DIR_ABS}/aif_plus_v5.shard${shard_idx}_checkpoint_comparison.csv"
done

for ((batch_start = 0; batch_start < ${#PLAN_AIF_V5_SHARDS[@]}; batch_start += PLAN_AIF_V5_WORKER_COUNT)); do
  plan_bg_reset
  for ((slot = 0; slot < PLAN_AIF_V5_WORKER_COUNT; slot++)); do
    shard_idx=$(( batch_start + slot ))
    if (( shard_idx >= ${#PLAN_AIF_V5_SHARDS[@]} )); then
      break
    fi

    gpu_id="${PLAN_GPU_IDS_ARR[slot]}"
    dataset_group="${PLAN_AIF_V5_SHARDS[shard_idx]}"
    shard_results="${SHARDS_DIR_ABS}/aif_plus_v5.shard${shard_idx}_results_online.csv"
    shard_windows="${SHARDS_DIR_ABS}/aif_plus_v5.shard${shard_idx}_window_errors.csv"
    shard_checkpoint="${SHARDS_DIR_ABS}/aif_plus_v5.shard${shard_idx}_checkpoint_comparison.csv"
    RESULT_SHARD_PATHS+=("${shard_results}")
    WINDOW_SHARD_PATHS+=("${shard_windows}")
    CHECKPOINT_SHARD_PATHS+=("${shard_checkpoint}")

    run_module_on_gpu_bg \
      "aif-plus-v5-shard-${shard_idx}" \
      "${gpu_id}" \
      "${PLAN_AIF_V5_THREADS}" \
      "$(plan_child_log_path "013-aif-plus-v5-shard-${shard_idx}")" \
      cli.run_aif_plus \
      --config "${AIF_PLUS_V5_CONFIG}" \
      --views-dir "${STATS_DIR}/window_views" \
      --registry "${STATS_DIR}/dataset_registry.csv" \
      --events "${STATS_DIR}/final_artifact_events.csv" \
      --support-summary "${REPORTS_DIR}/clean_view_support_summary.csv" \
      --baseline-results "${RESULTS_DIR}/counterfactual_2x2.csv" \
      --datasets "${dataset_group}" \
      --horizons "${HORIZONS_CSV}" \
      --seeds "${SEEDS_CSV}" \
      --run-tag "shard${shard_idx}" \
      --write-window-errors true \
      --window-error-rich-fields false \
      --debug-write-val-window-errors false \
      --results-out "${shard_results}" \
      --window-errors-out "${shard_windows}" \
      --checkpoint-comparison-out "${shard_checkpoint}" \
      --arg-out "" \
      --wgr-out "" \
      --ri-out "" \
      --report-out ""
  done

  if ! plan_wait_bg; then
    plan_log "one or more AIF-Plus-V5 shard jobs failed"
    exit 1
  fi
done

MERGED_RESULTS_ONLINE="${RESULTS_DIR_ABS}/aif_plus_v5_results_online.csv"
MERGED_WINDOW_ERRORS="${RESULTS_DIR_ABS}/aif_plus_v5_window_errors.csv"
MERGED_CHECKPOINT_COMPARISON="${RESULTS_DIR_ABS}/aif_plus_v5_checkpoint_comparison.csv"

plan_log "merge checkpoint comparison shards"
PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
  "${MERGED_CHECKPOINT_COMPARISON}" \
  "${CHECKPOINT_SHARD_PATHS[@]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

out_path = Path(sys.argv[1])
paths = [Path(item) for item in sys.argv[2:]]
out_path.parent.mkdir(parents=True, exist_ok=True)
frames = [pd.read_csv(path, low_memory=False) for path in paths if path.exists()]
if frames:
    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [
        "dataset_name",
        "backbone",
        "lookback",
        "horizon",
        "train_view_name",
        "eval_view_name",
        "seed",
        "checkpoint_variant",
    ]
    existing = [column for column in sort_cols if column in merged.columns]
    if existing:
        merged = merged.sort_values(existing).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
else:
    pd.DataFrame().to_csv(out_path, index=False)
PY

plan_log "recompute finite-window tables from shard window_errors"
PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
  "${ROOT_DIR}/scripts/recompute_aif_plus_tables_from_window_errors.py" \
  --results-online "${RESULT_SHARD_PATHS[@]}" \
  --window-errors "${WINDOW_SHARD_PATHS[@]}" \
  --merged-results-out "${MERGED_RESULTS_ONLINE}" \
  --merged-window-errors-out "${MERGED_WINDOW_ERRORS}" \
  --results-out "${RESULTS_DIR_ABS}/aif_plus_v5_results.csv" \
  --diagnostics-out "${RESULTS_DIR_ABS}/aif_plus_v5_validity_diagnostics.csv" \
  --arg-out "${RESULTS_DIR_ABS}/aif_plus_v5_artifact_reliance_gap.csv" \
  --wgr-out "${RESULTS_DIR_ABS}/aif_plus_v5_worst_group_risk.csv" \
  --ri-out "${RESULTS_DIR_ABS}/aif_plus_v5_ranking_instability.csv" \
  --mainboard-out "${RESULTS_DIR_ABS}/aif_plus_v5_mainboard.csv" \
  --boundary-board-out "${RESULTS_DIR_ABS}/aif_plus_v5_boundary_board.csv" \
  --appendix-board-out "${RESULTS_DIR_ABS}/aif_plus_v5_appendix_board.csv" \
  --summary-out "${REPORTS_DIR_ABS}/aif_plus_v5_summary.md"

plan_log "package V5 artifacts"
"${PYTHON_BIN}" - \
  "${ROOT_DIR}" \
  "${ARTIFACTS_DIR_ABS}/013-aif-plus-v5-artifacts.zip" \
  "${MERGED_WINDOW_ERRORS}" \
  "${REPORTS_DIR_ABS}/aif_plus_v5_summary.md" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path
import zipfile

root_dir = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
merged_window_errors = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
archive_path.parent.mkdir(parents=True, exist_ok=True)

window_limit_mb = int(os.environ.get("AIF_PLUS_V5_MAX_WINDOW_ERRORS_ZIP_MB", "512"))
include_window_errors = merged_window_errors.exists() and merged_window_errors.stat().st_size <= window_limit_mb * 1024 * 1024

artifact_paths = [
    root_dir / "src/cli/run_aif_plus.py",
    root_dir / "baseline/AIFPlus/AIFPlus.py",
    root_dir / "configs/aif_plus_v5.yaml",
    root_dir / "plan/013-run-aif-plus-v5.sh",
    root_dir / "scripts/recompute_aif_plus_tables_from_window_errors.py",
    root_dir / "results/aif_plus_v5_results_online.csv",
    root_dir / "results/aif_plus_v5_results.csv",
    root_dir / "results/aif_plus_v5_validity_diagnostics.csv",
    root_dir / "results/aif_plus_v5_mainboard.csv",
    root_dir / "results/aif_plus_v5_boundary_board.csv",
    root_dir / "results/aif_plus_v5_appendix_board.csv",
    root_dir / "results/aif_plus_v5_artifact_reliance_gap.csv",
    root_dir / "results/aif_plus_v5_worst_group_risk.csv",
    root_dir / "results/aif_plus_v5_ranking_instability.csv",
    root_dir / "results/aif_plus_v5_checkpoint_comparison.csv",
    root_dir / "reports/aif_plus_v5_summary.md",
]
if include_window_errors:
    artifact_paths.append(root_dir / "results/aif_plus_v5_window_errors.csv")

log_dir = root_dir / "logs"
if log_dir.exists():
    artifact_paths.extend(sorted(log_dir.glob("*.log")))
    artifact_paths.extend(sorted(log_dir.glob("newplan/**/*.log")))

missing = [path for path in artifact_paths if not path.exists()]
if missing:
    missing_text = ", ".join(str(path.relative_to(root_dir)) for path in missing)
    raise FileNotFoundError(f"Missing V5 artifacts before packaging: {missing_text}")

summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
window_note = (
    f"\n- merged window_errors zip status: {'included' if include_window_errors else 'excluded'}"
    f"\n- merged window_errors local path: `{merged_window_errors}`\n"
)
if "merged window_errors zip status" not in summary_text:
    summary_path.write_text(summary_text.rstrip() + "\n" + window_note, encoding="utf-8")

if archive_path.exists():
    archive_path.unlink()

with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for path in artifact_paths:
        zf.write(path, arcname=str(path.relative_to(root_dir)))

print(f"archive={archive_path}")
PY

plan_log "done"

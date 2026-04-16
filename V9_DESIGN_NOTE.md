# AIF-Plus-V9 design note

## Goal

Target **ETTh1** and **exchange_rate** on the clean mainboard while keeping the rest of the mainboard as close to V8 as possible.

## Why V9 is dataset-targeted instead of globally changed

- V8 is already the strongest AIF variant on the clean mainboard overall.
- The remaining persistent gaps are concentrated on **ETTh1** and **exchange_rate**.
- Non-target datasets (especially `solar_AL` and `ETTm1`) should not be destabilized by another global architecture rewrite.

## Diagnosis carried over from V8

### ETTh1
- Small / clean dataset.
- V8 improved over V7/V4 but still trails the best baseline on all four horizons.
- The V8 direct linear head is likely too rigid for longer horizons.
- The strongest baseline family on ETTh1 includes strong local temporal inductive bias (e.g. ModernTCN-like behavior).

### exchange_rate
- Very small training sets, especially at long horizon.
- V8 already helped by delaying over-early convergence, but the linear head is still too inflexible.
- A tiny amount of cross-channel interaction is still useful here, but full channel-dependent modeling is unnecessary.

## V9 architectural changes

### 1. Decomposition-aware direct head (new)
- Add `direct_head_type`:
  - `shared_linear`
  - `decomp_linear`
- `decomp_linear` performs a simple moving-average decomposition into seasonal + trend components and predicts them with separate linear maps.
- Used for `ETTh1` only.

### 2. Adaptive basis direct-head correction (new)
- Add optional `AdaptiveBasisLinearHead`.
- It implements a **low-rank adaptive linear correction** on top of the direct head.
- This is a lightweight "predictor generation" style mechanism intended to fix the rigidity of a pure linear head.
- Used for:
  - `ETTh1`
  - `exchange_rate`

### 3. Local temporal refiner (new)
- Add optional `LocalTemporalRefiner`.
- It is a tiny depthwise temporal conv residual head operating on the clean normalized series.
- It is intended to restore a small amount of ModernTCN-like local inductive bias without reverting to a heavy backbone.
- Used for `ETTh1` only.

### 4. Preserve V8 defaults elsewhere
- All new modules are **off by default**.
- Non-target datasets remain as close as possible to V8.

## Dataset-specific plan

### ETTh1
- Move from plain shared linear head to `decomp_linear`.
- Turn on adaptive basis correction.
- Turn on local temporal refiner.
- Keep channel context off.
- Keep uncertainty residual branch off.
- Slightly enlarge the deep residual corrector budget and use a longer large patch.

### exchange_rate
- Keep the direct head simple (`shared_linear`) but add adaptive basis correction.
- Keep tiny state-level channel mixer on.
- Keep the model small.
- Reduce batch size and warmup; increase epochs and patience to avoid under-training due to too few optimizer steps per epoch.

## Files changed
- `baseline/AIFPlus/AIFPlus.py`
- `src/cli/run_aif_plus.py`
- `configs/aif_plus_v9.yaml`

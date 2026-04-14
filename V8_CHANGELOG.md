# AIF-Plus-V8 small-clean revision

## Why V8

V7 already became the best AIF variant on the clean mainboard overall, but ETTh1 and exchange_rate still lagged the strongest baselines.
Log inspection shows a strong early-fit pattern on the small/clean datasets:

- exchange_rate: best epoch often at 1-2.
- ETTm1: best epoch often at 4-6.
- ETTh1: best epoch usually still early/mid training.

This suggests the deep trunk is still too flexible relative to the available clean training signal.

## Main V8 changes

1. Add `use_linear_head`:
   - a shared NLinear-style direct forecasting head on normalized masked inputs.
   - intended for small clean datasets.

2. Turn the deep token model into a residual refiner:
   - `pred_clean = pred_linear + lambda_deep * pred_deep`
   - `lambda_deep` is bounded by `deep_residual_max` and initialized small.

3. Add `use_periodic_branch` / `use_frequency_branch`:
   - ETTh1 / ETTm1 / exchange_rate can disable these branches to reduce capacity.

4. Add `use_state_channel_mixer`:
   - a tiny channel-state mixer over per-channel latent states.
   - enabled for exchange_rate only.

5. Restore clean-val priority for exchange_rate via config:
   - `min_clean_val_windows: 1`
   - avoids raw-val fallback on very small clean validation splits.

## Dataset-specific intent

- ETTh1: smaller model + direct linear head + no periodic/frequency branches.
- ETTm1: keep V7 backbone mostly intact, but add direct linear head and drop periodic/frequency branches.
- exchange_rate: much smaller model, direct linear head, tiny channel-state mixer, smaller batch size, clean-val priority.

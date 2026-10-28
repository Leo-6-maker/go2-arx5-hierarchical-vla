# PI0251 G0 — K/X source audit and stored-screen mechanism breakdown

**Authority:** `PI0251_XYZ_EXCESS_RISK_TO_TRAIN_SIM_GO`, protocol commit `e2b86ef2c2d3ee2fb8cab05d24de24a43cd0f0a6`.

**Status:** G0 read-only analysis complete. It adds no model queries and leaves the original thresholds, checkpoints, sample table, and downstream gates unchanged.

## Pair-identity evidence classification

The PI0249 X closeout is `PASS_POSTHOC_CPU_REPLAY`, using `EXACT_COMPARE_OF_CAPTURED_K_X_TRAINING_IDENTITIES`. The 2,000-row pair file contains the preserved raw first 1,000 rows and a 1,000-row recovery suffix reconstructed from the saved K/X rollout, step-RNG, and sample-consumption records. Reconciliation made zero model queries. This is **post-hoc comparison of observed training captures**, not a newly regenerated expected-randomness comparison. It still does **not** prove that X checked parity online before backward or optimizer update.

The closeout records reconciliation-source SHA-256 `f10b27ac…9b7c9`, which matches the preserved local source copy. The current Corelab copy hashes to `0f56826c…0cf`; the closeout does not record the executed path. Both identities are retained in the source index as a source-copy variance. The underlying pair rows and input capture hashes are separately indexed.

## Final-2000 screen verdict

Each screen uses the same frozen 64-source set and B0/B1 noise identities, the original S4 path, H40 and `dt=0.05`. CPU replay independently passed for all 128 cells in each arm. The reported quality rule is unchanged: zero `REJECT`, median XYZ RMSE ≤ 0.020 m, p95 XYZ RMSE ≤ 0.035 m, and median SO(3) RMSE ≤ 0.050 rad.

| Arm | Cohort | Noise | Median XYZ (m) | p95 XYZ (m) | Median SO(3) (rad) | Original group gate |
|---|---:|---:|---:|---:|---:|---|
| K2000 | all64 | B0 | 0.020316 | 0.043039 | 0.016014 | FAIL |
| K2000 | all64 | B1 | 0.020317 | 0.042327 | 0.017308 | FAIL |
| K2000 | old16 | B0 | 0.016364 | 0.030553 | 0.014834 | PASS |
| K2000 | old16 | B1 | 0.016761 | 0.033201 | 0.017294 | PASS |
| X2000 | all64 | B0 | 0.018867 | 0.040995 | 0.018293 | FAIL |
| X2000 | all64 | B1 | 0.018385 | 0.039688 | 0.019154 | FAIL |
| X2000 | old16 | B0 | 0.014882 | 0.028264 | 0.015690 | PASS |
| X2000 | old16 | B1 | 0.016824 | 0.033709 | 0.019154 | PASS |

There are zero `REJECT` cells in these groups. K all64 misses both the median and p95 limits in both noises; X all64 misses p95 in both noises. The old16 four groups all pass. This preserves the PI0249 result and does not convert an old16 PASS into a 64-source qualification.

## What the stored NPZs show

All 256 existing NPZs were hash-checked against their records. Recalculation from each saved physical `decoded` and `expert` array matched the stored XYZ RMSE, tail-35:39 SSE fraction, and maximum residual increment for all 256 cells (768 comparisons). The metric is `sqrt(mean_t(sum_xyz(residual[t]^2)))`; XYZ was not divided by three.

- **Hard windows repeat across arms.** The largest B0 error in both arms is `pi0227_persistent_train_a_j03_yp015_s04_ym050_r03_r03_227079:obs-0073` (new48): 0.06565 m for K and 0.06076 m for X. Its expert path is 0.626 m with a 1.672 m/s maximum speed. Other recurring hard windows are `pi0227_persistent_train_a_j00_nominal_s00_c0_r02_r02_227002:obs-0066` (K 0.04486 m, X 0.03845 m), `pi0227_persistent_train_b_j01_xp015_s00_c0_r01_r01_227201:obs-0000` (K 0.04486 m, X 0.04153 m), and `pi0227_persistent_train_d_j02_xm015_s03_yp050_r02_r02_227594:obs-0006` (K 0.04236 m, X 0.04218 m). The complete ranked lists and per-cell hashes are in the JSON breakdown.
- **Window-specific bias is material but not a single global offset.** Median per-window bias magnitude is 6–12 mm across the old16/all64 groups. The median fraction of per-window XYZ squared error explained by its mean residual vector is 0.33–0.41. Yet the mean bias vector averaged across windows is close to zero on each axis, so directions cancel across sources. Within-window residual variation accounts for the rest.
- **Errors grow late in the old16 windows while remaining within its registered limits.** For old16, median H30–39 to H00–09 RMSE ratios range from 1.63 to 2.05; the H35–39 segment contributes about 20–21% of per-window position SSE. Across all64 the corresponding median ratios are near 0.89–1.10, with H30–39 median error around 19–20 mm. This points to heterogeneous, time-varying errors rather than one uniform tail-only failure.
- **No active-XYZ boundary saturation appears in the saved raw outputs.** Across 7,680 raw XYZ scalars per all64/noise group, none is ≥0.99 or outside ±1. The maximum absolute value is 0.883 for K and 0.898 for X. Thus decoder-boundary clipping is not supported as an explanation for the all64 failures.

These are descriptive measurements of existing outputs; they do not establish causality. They do not justify a new loss, threshold, sample, or optimizer search. The authorized C/E contrast remains the next test.

## Evidence files

- `PI0251_G0_ACTUAL_NPZ_BREAKDOWN_20260928.json` — SHA-256 `77538e8d6b14cb82b8c2ab76a8fdb2cd1e9f3d15e854589ee63b008c812eb3c7`.
- `PI0251_G0_SOURCE_INDEX_20260928.json` — SHA-256 `c282d9452fdfa3dafc7182e1de49beb6b637135507c10275e4dcb222f602aa06`; exact screen/training code, reports, receipts, model revisions, source hashes, and the explicit reconciliation-source variance.
- `scripts/pi0251_g0_npz_breakdown.py` — read-only analysis code. It ran against Corelab’s existing screen NPZs; 256/256 records and 768/768 stored-metric comparisons closed.

`selected_checkpoint` remains `NONE`. No FULL, selector, WBC, or simulator admission is inferred from this G0 report.

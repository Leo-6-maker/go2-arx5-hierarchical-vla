# PI0254 P1 fixed quality comparison

Status: P1_VALID_BASELINES_COMPLETE. 320/320 valid new queries;1280 inference head forwards. Both points independently CPU replayed.

| Model / cohort / noise | Cells | Median XYZ (m) | p95 XYZ (m) | Median SO3 (rad) | Model REJECT | Expert REJECT | Model-only REJECT |
|---|---:|---:|---:|---:|---:|---:|---:|
| G / P64 / B0 | 64/64 | 0.1037272902 | 0.3249429038 | 0.0740325794 | 0 | 0 | 0 |
| FULL10000 / P64 / B0 | 64/64 | 0.0681637566 | 0.2383896399 | 0.0290103675 | 0 | 0 | 0 |
| G / P64 / B1 | 64/64 | 0.1029578087 | 0.3247023825 | 0.0744153176 | 0 | 0 | 0 |
| G / E64 / B0 | 64/64 | 0.0920797366 | 0.3697688307 | 0.0769903202 | 0 | 2 | 0 |
| G / E64 / B1 | 64/64 | 0.0946892662 | 0.3696786982 | 0.0763442181 | 0 | 2 | 0 |

## Interpretation

FULL improves XYZ RMSE in 46/64 paired P64 windows (71.87500%). This supports transfer/interference as a compatible account alongside the preserved O64 regression; it does not identify a unique optimizer cause, prove data insufficiency, or prove GR00T cannot generalize.

P64 was excluded from this new stage only; FULL previously trained on it and C_T is inherited. E64 is a trained-on fitting audit. One paired training family and two noise draws are not independent seed replications. All REJECT trajectories remain in denominators. No world target distance or physical task-success claim.

## Fixed paired differences (FULL minus G)

| Metric | Median paired difference | p95 paired difference | Fraction FULL improved |
|---|---:|---:|---:|
| centered_shape_rmse_m | -0.0088812235 | 0.0200530576 | 0.734375 |
| first_so3_error_rad | -0.0462698426 | 0.0072271625 | 0.937500 |
| first_xyz_error_m | -0.0012853161 | 0.0138086403 | 0.593750 |
| last_so3_error_rad | -0.0460510500 | 0.0076638726 | 0.937500 |
| last_xyz_error_m | -0.0364607269 | 0.0765030843 | 0.750000 |
| mean_offset_m | -0.0292974122 | 0.0505313978 | 0.703125 |
| so3_rmse_rad | -0.0461699696 | 0.0075256867 | 0.937500 |
| xyz_rmse_m | -0.0307303485 | 0.0519217699 | 0.718750 |

A–D/phase/source-rollout aggregates, all per-window first/last/offset/shape errors, gate issues and exact identities are preserved in P1_comparison.json and CPU reports. FULL/B1 was not queried.

Decision: CONTINUE_FROZEN_U576_THEN_R576; no sample/loss/seed/initialization changes. Deployment selected_checkpoint=NONE.

P1 comparison SHA256: bf8e72f59d3d9e1fb4e3e5be47e045f09ee70c53135bac8f88e1664d8878d055

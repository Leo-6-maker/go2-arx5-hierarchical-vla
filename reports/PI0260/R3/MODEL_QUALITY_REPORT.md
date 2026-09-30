# PI0260-R3 matched Motion2 pilot — fixed offline quality

**Scientific verdict: both final models FAIL the original full offline acceptance.** Both Base and G learned substantially lower trajectory error on the synthetic KIN-VAL cases and passed all four numeric task/noise gates at step 6000. Base failed M1 instruction and paraphrase checks; G failed all M1 and M3 instruction and paraphrase checks. No selected or deployment checkpoint is granted.

## Scope and provenance

- One sealed COMPACT_30_10 kinematically rendered reference dataset, 30 TRAIN and 10 KIN-VAL geometry families, unchanged labels, TRAIN-only processor/statistics, identical 6000-visit schedule and paired noise for both arms. Base/G each completed 6000 main optimizer updates, with full Trainer and RNG/optimizer recovery checks. The prior R2 H1 blocker remains as historical evidence; R3 admitted the measured same-command hidden-time alias without claiming full observability.
- G6000 complete Trainer: HF PUBLIC `leoyuer/pi0246-gate-pass-campaign@fae3f84169beff33bf42bd47c85d8d2c082ba202/arms/PI0260/R3/G/checkpoint-6000`, tree `48bc0cefca21bb52fa791492ffa0a0fbc0d34a9dd10db7b0063dd142542e89cb`, 16/16 fixed-revision full-file readback PASS. Base6000: revision `52abc8d151b67bb667e43fcf874dfe2cfa253364`, tree `24ed9d1b5a0e42e0aff6d62c54d8438d2f202d765b006d90c62d8623604e4ce5`, same full readback PASS.
- Frozen evaluation card: 4 model points × 2 actual noise packets × (156 Motion2 + 64 O64 cases) = **1760 logical cells**. All original row labels, timestamps, and gates were kept. 1724 actual query attempts and 6896 S4 head forwards, under caps 1824/7296. Base_0 and G_0 each had 18 redundant identical-condition forwards; they are counted, their first-valid results retained, and the final points reused identical forwards while scoring every row separately. Four independent saved-array CPU replays each checked 440 cells; all PASS. Same-condition/same-noise raw repeats had maximum absolute difference 0.
- Two noise packets cannot establish the model's full conditional distribution. The KIN-VAL split was part of development admission and is not blind TEST. The H40 t0 semantic checks assess entry/target discrimination, not completion of the full M3 contour or any physics.

## Primary fixed KIN-VAL table

Thresholds per task and noise: median XYZ ≤ 0.020 m, p95 XYZ ≤ 0.035 m, median SO3 ≤ 0.050 rad, REJECT = 0. Instruction and equivalent-paraphrase checks each require at least 10/12 correct, independently.

| Point | Task/noise | Median XYZ m | p95 XYZ m | Median SO3 rad | REJECT | Instruction | Paraphrase |
|---|---|---:|---:|---:|---:|---:|---:|
| Base0 | M1/B0 | .0494 | .0585 | .0671 | 16 | 6/12 | 6/12 |
| Base0 | M1/B1 | .0594 | .0642 | .0809 | 16 | 6/12 | 6/12 |
| Base0 | M3/B0 | .0500 | .0583 | .0623 | 16 | 6/12 | 6/12 |
| Base0 | M3/B1 | .0585 | .0609 | .0747 | 16 | 6/12 | 6/12 |
| Base6000 | M1/B0 | .0053 | .0119 | .0072 | 0 | 7/12 | 6/12 |
| Base6000 | M1/B1 | .0079 | .0136 | .0095 | 0 | 8/12 | 7/12 |
| Base6000 | M3/B0 | .0086 | .0151 | .0029 | 0 | 12/12 | 12/12 |
| Base6000 | M3/B1 | .0070 | .0134 | .0047 | 0 | 11/12 | 11/12 |
| G0 | M1/B0 | .0230 | .0414 | .0536 | 0 | 6/12 | 6/12 |
| G0 | M1/B1 | .0230 | .0414 | .0536 | 0 | 6/12 | 6/12 |
| G0 | M3/B0 | .0247 | .0343 | .0265 | 0 | 6/12 | 6/12 |
| G0 | M3/B1 | .0247 | .0342 | .0266 | 0 | 6/12 | 6/12 |
| G6000 | M1/B0 | .0073 | .0141 | .0129 | 0 | 8/12 | 6/12 |
| G6000 | M1/B1 | .0073 | .0154 | .0144 | 0 | 8/12 | 7/12 |
| G6000 | M3/B0 | .0089 | .0205 | .0022 | 0 | 6/12 | 6/12 |
| G6000 | M3/B1 | .0083 | .0201 | .0017 | 0 | 7/12 | 6/12 |

All four final numeric groups pass for **each** arm. The full conjunction fails for both arms. The G model's lower O64 median XYZ error (.178/.178 m versus G0 .184/.184 m for B0/B1) remains a large historical-task regression under the new processor; Base6000 is .184/.187 m. O64 does not grant old G qualification.

## Command and ambiguity diagnostics

The 12-case instruction and paraphrase groups contain six cases per command. Base6000 M1 command 0 is 6/6 and 6/6 on B0, while command 1 is 1/6 and 0/6; Base6000 M3 retains both commands (B0 6/6 per command). G6000 M3 command 0 is 6/6 in both B0 instruction/paraphrase groups, while command 1 is 0/6 in both; B1 command 1 is 1/6 instruction and 0/6 paraphrase. This is a measured command-direction bias despite low trajectory error. These are fixed reference-margin outcomes, not a full physical contour evaluation.

The actual processed-input audit has no mixed-command alias group or processor merge. TRAIN has 7425/13003 windows in repeated exact-condition groups, with empirical deterministic squared-position floor RMSE .00147 m; KIN-VAL has 2662/4374 and floor .00143 m. Under the frozen 6000-visit exposure weighting the TRAIN floor is .00134 m. Maximum observed same-command H40 trajectory separation is .01577 m RMSE, with .02591 m maximum pointwise separation. These are label-distribution properties, not model scores or a corrected gate.

The fixed 16-case KIN-VAL subset per task/noise contains no two distinct rows from the same exact-condition group, so its *within-subset* empirical floor is zero even though 8–9 selected rows per task/noise belong to alias groups in the full KIN-VAL corpus. Accordingly, the fixed subset's model MSE cannot be fairly reduced by, or directly compared against, the full-corpus .00143 m floor. The separate saved-array breakdown reports each alias/singleton case, phase quartile, family, command margin and valid within-subset MSE/floor. Final family-median XYZ spans across the eight KIN-VAL families are .0032–.0120 m for Base and .0037–.0158 m for G depending on task/noise; phase-quartile medians span .0032–.0135 m and .0043–.0158 m respectively. These small per-family groups are descriptive, not independent confidence intervals.

The observed M1 and G M3 cross-command failures are not explained solely by the accepted *within-command* hidden-time alias. The experiment does not isolate whether visual features, language conditioning, shared-task interference or the fixed objective caused the bias. No seed, loss, target, phase feature or model was changed in response.

## Media and stopping boundary

`PI0260_R3_Base_G_before_after.mp4` is a 15 s, 300-frame fixed-case comparison using saved actual 96×96 body/wrist RGB and saved H40 predictions at common XY/XZ physical scales. Four chapters use the first sealed KIN-VAL paired semantic family for M1 and M3 at B0; a fifth shows the documented same-input M1 t0/t0.8 reference alias. It makes zero model or renderer calls and is marked **KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC**. A video frame or numerical pass is not a deployment result.

The scientific pilot has no passing full-acceptance checkpoint. `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`; physical/WBC/TEST/DEV-F/real hardware remain at zero. The protocol stops at physical-transfer admission rather than opening another training or physics branch.

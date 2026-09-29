# PI0254 U final4608 — independent valid endpoint, R blocked

Status: **PARTIAL_R_INPUT_PAIRING_BLOCKED; U_NOT_PROMISING**.

U completed 4608 valid optimizer updates. All384 final cells are first-valid and passed independent CPU replay of the actual B0/B1 noise, original S4 traces, decoder, metrics and gate. The original unrounded qualification function passes E64/P64 under each noise, but fails all four O64/old16 retention groups. This is a valid U quality failure. R has no valid training endpoint and consumed both allowed submissions; its input-pairing guard failure is an integrity block, not a model quality FAIL. No equal-budget rehearsal contrast or research winner exists.

## Original endpoint table

All-row metrics include rejected outputs. XYZ is in millimetres; SO3 is in radians. Rounded display does not determine qualification.

| Group | Noise | Valid | Median XYZ | p95 XYZ | Median SO3 | Model REJECT | Expert REJECT | Original group result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| O64 | B0 | 64/64 | 34.591 | 80.784 | .032485 | 1 | 0 | FAIL |
| O64 | B1 | 64/64 | 34.435 | 79.566 | .031836 | 1 | 0 | FAIL |
| old16 | B0 | 16/16 | 30.699 | 74.638 | .037863 | 0 | 0 | FAIL |
| old16 | B1 | 16/16 | 30.978 | 75.504 | .038958 | 0 | 0 | FAIL |
| E64 | B0 | 64/64 | 49.091 | 213.492 | .037679 | 0 | 2 | PASS relative research rule |
| E64 | B1 | 64/64 | 50.098 | 212.916 | .037948 | 0 | 2 | PASS relative research rule |
| P64 | B0 | 64/64 | 85.164 | 235.133 | .046034 | 0 | 0 | PASS relative research rule |
| P64 | B1 | 64/64 | 84.501 | 233.595 | .044528 | 0 | 0 | PASS relative research rule |

E64 median XYZ ratios to G are .53313153/.52907582 (46.7/47.1 percent reductions). P64 ratios are .82104237/.82073441 (17.9/17.9 percent reductions). Both also satisfy their original p95/SO3 no-worse and zero model-only-REJECT checks. E64 is trained support; P64 is not replayed in this stage but is previously exposed TRAIN/development data. Neither establishes blind generalization or physical success.

O64/old16 require zero REJECT, median XYZ <=20mm, p95 <=35mm and median SO3 <=.050rad independently. Both XYZ statistics fail in all four retention groups. The two O64 model-only REJECTs are the same source `pi0227_persistent_train_a_j03_yp015_s04_ym050_r03_r03_227079:obs-0073`, with peak linear speeds2.30307102/2.16987293m/s versus the unchanged2.0 threshold. This is outside old16. No rejected output was repaired or resampled.

The endpoint improves new-support fit relative to G while losing old-support qualification. It does not identify a unique optimizer/objective/observability cause, and cannot measure the effect of rehearsal without R. Task, phase, source-rollout, expert-nonREJECT, first/last, mean-offset and centered-shape summaries remain in `partial_closeout.json`. Centering is diagnostic decomposition only.

## Actual counters and durable model

- Successful optimizer updates: U4608; R0; raw total4608.
- U valid-run heads: native4608, S4 rollout18432, backward recomputation18432 =41472.
- Initial failed engineering attempt637485 additionally consumed native1 + rollout4, backward0. **Raw training heads41477**, not41472. `raw_training_counters_v2.json` binds both source ledgers. Earlier endpoint-only counters remain unchanged and are explicitly separated.
- New inference queries1088; inference head forwards4352; invalid queries0. Includes baseline320, U1152 384, U4608 384. R milestone allocations remain unspent and cannot fund another condition.
- U final complete9-file Trainer revision `a90b3d3c751a51dff5ecfda1e9254c9f6d786664`, tree `c3e6cdd7c47606ddeeb1fed6438545ccac0c5adce2c3e74af6eab36b77b66b12`; fixed full-content readback already PASS.
- Historical `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`. No new simulation, CALIB/DEV/F read, merge, mail, monitor or training branch.

Final raw arrays and training audit are archived separately under a unique PI0254 evidence prefix. The HF receipt supplies the fixed revision and full-content readback; this report alone does not claim that upload has completed. Quota headroom remains UNKNOWN.

Completed archive binding: `leoyuer/pi0246-gate-pass-campaign@9aec61d5e4f5b9858059bcb3b45ac8956c448f71`, prefix `evidence/PI0254/U-final4608-partial-closeout-20260929`; two HF files74,643,637 bytes, tree `7a4cb3ec8258a5c3b1b1839963afa99ba95fdece63a82d5eac9afaeda5734f20`. The archive contains1169 members89,282,899 logical bytes, including the67-member training-audit tar whose entries were independently checked on Corelab. Full HF file-content SHA/size readback PASS used authentication; it is not labelled anonymous. The repository is public. See `U_final_partial_HF_receipt.json` and `MEMBER_MANIFEST.json`.

The original first-valid `partial_closeout.json` is preserved. `partial_closeout_with_raw_counters.json` adds the failed-engineering head accounting without changing cells, quality decisions or original counters. Archive-only code and receipts do not imply new model work.

# PI0252 R2 pre-FULL TRAIN sim: valid physical FAIL

Authorization: `PI0252_R2_G_QUALIFIED_SIM_FULL_CONTINUATION_GO`, commit `a5ccfba37a54e53b1ad1818a095edf67319aba9c`, Issue #11 canonical 5866382788. P0 correction canonical 5868341369 remains valid. A training arm valid FAIL and G offline four-group PASS remain unchanged.

**Decision: `PI0252_R1_PRE_FULL_TRAIN_SIM_FAIL`.** G task TRAIN-A/C0 first-valid one-chunk missed the original 35 mm actual world-TCP criterion. The dependency chain stops here: task B–D model queries, receding, FULL, CALIB and post-FULL sim were not launched. `development_checkpoint=G2000`; `formal_selected_checkpoint=NONE`.

## Actual quality results

| Episode | Original gate | Reference minimum mm | Achieved minimum mm | Achieved final mm | First hit tick | Full duration / clean | Result |
|---|---|---:|---:|---:|---:|---|---|
| Expert A | ACCEPT | 0 | 8.712 | 19.258 | 41 | 97 ticks / yes | PASS |
| Expert B | ACCEPT | 0 | 2.692 | 90.449 | 44 | 97 ticks / yes | PASS |
| Expert C | ACCEPT | 0 | 4.514 | 71.122 | 41 | 97 ticks / yes | PASS |
| Expert D | ACCEPT | 0 | 9.828 | 69.432 | 37 | 97 ticks / yes | PASS |
| G TRAIN-A/C0 | ACCEPT | 104.978 | 156.498 | 201.346 | none | 97 ticks / yes | **FAIL** |

The original success criterion uses an actual target hit during a complete, clean episode. Endpoint distance is retained separately; expert B–D did not remain within 35 mm at the final tick. No new endpoint gate is introduced.

G reference endpoint distance was 176.476 mm. Tracking error max/mean/final was 166.815/137.424/140.697 mm. No native physical termination, fall or reset occurred. A frozen WBC control can reach the target in this runtime, but G's generated reference itself misses by 104.978 mm. This result cannot be explained solely by insufficient WBC tracking. It does not invalidate the prior fixed-cohort offline quality PASS or establish generalization performance.

## Bound runtime and budget

- G fixed HF revision `efecea485ae77000599d19171b02491f8d514f42`; complete Trainer tree `cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226`. Four current inference-view model files were fully rehashed before use.
- Original PI0226 effective reset/query seed 22301; native dt 0.02 s, H40 sample dt 0.05 s, one persistent reference duration 1.95 s. Same scene, fresh body/wrist RGB, StateV1, original dense-relative decoder, original gate and frozen WBC.
- Fresh WBC archive/live comparison: 742 files identical, V2 tree `9d653c39c73d5755a29d46f4dff91b69120244f660f41a977db471d58ba8b193`. Correct dog weight SHA is `51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670`; an older shortened field remains preserved and explicitly invalid.
- Current effective C_T processor stats are inherited. The base statistics hash `33aa1449...` differs from T-parent `c33e6aee...`, as already documented in G's frozen card; custom embodiment state/action stats are overwritten with frozen TRAIN stats `c24d465b...` and relative stats `10b52e09...` by the original helper.
- **New project model queries: 1; actual head calls: 4; optimizer updates: 0.** Actual initial noise was `[1,40,132]` BF16 on cuda:0, SHA256 `585570a70dbb0cfb597bfd49c6ffce1e77b598c744f2a89eb3e890455c9a0852`. No expert prefix, RTC, padding intervention, reseeding retry or action correction.
- Initial model reference binding preserves root/dof/episode/history identity. No reference updates follow initial binding. CPU replay verifies target distances, world-frame tracking, raw BF16 noise bytes, original gate decision and initial binding.
- Corelab GPU0 evaluation coexists with unrelated projects, as explicitly authorized. No Corelab training and no unrelated process termination.

## Preserved engineering and reporting deviations

1. Raw expert A attempt1 retained: unbound archived C0 reference used an older anchor and never traversed the current target. It is an invalid expert target-binding attempt, not a G quality result. The existing PI0227 expert target/anchor binding function was then reused before canonical controls. Raw accounting: five control attempts, four canonical valid controls; extra invalid attempt 96 WBC ticks; no G query consumed by that attempt. No WBC, scene pose or gate change.
2. The original Stage21 summary increments `reset_events` on the wrapper's normal `REFERENCE_COMPLETE` done signal. It reports one aggregate event here, while all 97 actual `reset_buf` values and native termination counts are zero. Independent replay documents this distinction; the original summary is preserved.
3. Inherited canary oracle side fields use the old unbound expert anchor and unmatched sampling. `oracle_world_target_stats` and `oracle_vs_model_position_rmse_m` are invalid comparison diagnostics and are not used in qualification. The valid negative result comes directly from G reference/actual achieved TCP versus the fixed world target.
4. First loading waited on shared storage; the existing healthy process was kept. One query returned in 6.341 s after readiness; synchronous paused simulation is not a real-time claim.
5. Report-only video encoding uses existing native FFmpeg, and the CPU scorer received that path/encoder repair after the generation binding was frozen. Generation, gate, decoder and WBC source bytes were unchanged. This caused no replay or extra model query.

## Evidence

Generation binding SHA256: `74b8cce6872af2026f3b87f47246b5a814a852cf1207d9f6b0328039cd26fcb7`.

First-valid G manifest SHA256: `f136ec0213bda0d8598f9fd64509b15c3c0f4bd4f6172a15e0b206e44854f739`.

Actual TCP rows SHA256: `f84e85461cc3c1ddf846c67347a939e446165f8a9272dc019063a34d99d71256`.

Actual S4 receipt SHA256: `c91f135edbda93db0089a5ee3894119fb0ada0764d7ebffba4cf4a18d41f45e4`.

Corelab evidence root: `/home/liuyu-nfs/Projects/go2-arx5/PI0252/pre_full_sim/`. Full raw camera arrays, encoded videos, noise, request/response packets, actual model action, control episodes, source identities and independent CPU replay are retained. HF public fixed-revision readback receipt accompanies this report.

No new seed, candidate, loss or sampling branch is admitted by this failure. Emails and monitors remain off; DEV/F, DAgger, hardware and merge remain locked.

## Fixed public archive and CPU receipt

[HF public immutable evidence](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/d0cdfac5297c10fb8dd4ccc71f3e60cf578dbaa7/evidence/PI0252/r2-pre-full-train-sim-20260928): **PASS_FULL_FILE_READBACK**, all 61 files, 11621636 bytes, fixed revision `d0cdfac5297c10fb8dd4ccc71f3e60cf578dbaa7`. Every file was streamed and fully hashed at that revision; no second model weights download or training was performed. Evidence tree SHA256: `b0b15ccba077a015ce2b289adcda775dc573017fb8091f724c723f3c5af11497`.

Independent CPU replay SHA256: `fae083942291b9598a1a8729ee96d3e9a284b90a6b0379d681588aa6136dc87b`. The full HF readback receipt is `reports/evidence/PI0252_R2_PRE_FULL_SIM/hf_readback.json`.
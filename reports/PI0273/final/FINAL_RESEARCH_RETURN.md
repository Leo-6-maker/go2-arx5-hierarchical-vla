# PI0273 — full-support replay and reliable feedback research return

Date: 2026-10-05 SGT. Authority: `PI0273_FULL_SUPPORT_S4_REPLAY_AND_RELIABILITY_GO_V1`; release `b6e4e436dced30445a014721e2c0df268989244f`; Issue #11 canonical `5982104756`.

## Decision

The fixed experiment, both 5120-update arms and all registered endpoint panels, has been executed. Both arms FAIL original/reliable virtual qualification. No reliable virtual research candidate is nominated. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`. Old scientific FAILs, PI0272's negative DIRECT-superiority finding and prior WBC limitations remain unchanged.

BROAD improves M3 first completion and early fitting. TARGET improves terminal correction more than BROAD but has fewer original/reliable successes and seven virtual collisions. The registered targeted-replay superiority criterion is not satisfied. No additional replay, seed, mixture, loss, model or endpoint search follows from this result.

Terminal scientific decision: **PI0273_FULL_SUPPORT_REPLAY_RELIABILITY_COMPLETE — STOP_FOR_INDEPENDENT_TRANSFER_OR_SINGLE_TARGETED_REPAIR_DECISION**. Final publication and archive receipts, separately published with this report, prove administrative closeout; intermediate preparation fields such as `*_still_required` in historical receipts remain time-stamped history, not overwritten PASSs.

## Source, intervention and actual learning

Both arms load weights from PI0270 AGG4096, HF `leoyuer/pi0246-gate-pass-campaign`, revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`, prefix `arms/PI0270/AGG/checkpoint-4096`, tree `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`. They use fresh AdamW/cosine5120, LR1e-5, warmup10, accumulation2, weight decay1e-5 and clipping1.

Each consumes all original 8192 AGG input/label occurrences plus 2048 supplemental occurrences, in four base units followed by one supplemental unit. B has 6716 distinct processed input/label identities. T cycles all 296 registered conditions: 272 receive seven visits and 24 six. C cycles the original occurrence pools matched on family/task/command/exact language. Supplemental T has 296 distinct identities, C 1624; 104/2048 slots coincide in both original occurrence and processed input/label. These coincidences were retained. Full B/T/C membership, exact languages, schedules and exposure counts are in the frozen `design/FITTING_CARD.json`, `evidence/A0_SOURCE_AND_PLAN.json`, `evidence/BASE_ENCODING8192.json` and actual sample journals, retained in the summary archive.

CPU admission proved both complete 10240-micro schedules and actual encoding identity. Actual training audit proves every consumed input, native/random packet matching between arms, original loss/scopes, four finite nonzero generated-node gradients and mode restoration. The inherited objective is `Lnative + .4 Lposition + .1 Lincrement + .1 Lrotation + .1 Lpeak_speed`; original temporal weighting, S4, physical H40/dt, decoder and gate remain bound. There are 471 trainable and 559 frozen tensors. Training uses native BF16 epsilon, FP32 VL/query/Euler states and physical loss; model-only evaluation uses BF16 S4 queries with FP32 decoding. These phase contracts are preserved, not required to be cross-phase bit identical.

Both actual continuous8 versus fresh-process4+save+restore+4 canaries PASS: all 1030 model tensors and 1413 optimizer tensors have max absolute difference zero; saved RNG and scheduler are exact. Formal training is 5120 per arm, with complete 2560/5120 Trainers. Canary computation totals 32 completed updates, not sixteen. Raw total is 10272 updates and 184944 training heads; technical completed updates zero. Hopper job646518 exited zero. No arm was restarted to improve quality.

Prescott library identity `9254d0854dd7615e11de28d771ae408878ca8123a7ac204f21e4cc7a376cc2e5`, actual per-phase encoder receipts and CPU input equality are recorded. Actual loss, gradient, temporal-component, sample, RNG and S4 journals are retained in `evidence/training-proofs`; their twelve arm-journal hashes agree with `MAIN_TRAINING_AUDIT.json`. The audit SHA256 is `aa2a7750c40d7e37a3bda66f891ec5f0333ce4c39398077757c15025fe45772d`.

## Fixed endpoint qualification

| Point | M1 E0 | M1 E1 | M3 E0 | M3 E1 | Original successes | Reliable successes | Entire point safe | Original ROT legacy |
|---|---:|---:|---:|---:|---:|---:|---|---|
| AGG4096, reused |16/16|16/16|1/16|1/16|34/64|6/64|Yes|PASS|
| BROAD5120 |16/16|16/16|11/16|5/16|48/64|9/64|Yes|PASS|
| TARGET5120 |16/16|16/16|4/16|3/16|39/64|3/64|No, seven collisions|PASS|

Original qualification requires each task/backend at least13/16 and each command6/8, plus full-point safety. Reliable qualification also requires original completion by35s, full safe40s and all251 ticks in the last5s within35mm/.10rad. These are AND gates; aggregate counts do not qualify a point. TARGET's M1 raw success remains16/16 per backend, while its point-level safety failure also makes the point's M1 qualification false.

M3 command counts, c0/c1 out of eight each: BROAD E0=4/7, E1=1/4; TARGET E0=0/4, E1=0/3. Reliable M1 totals BROAD5/32, TARGET0/32; reliable M3 BROAD4/32, TARGET3/32. Neither arm clears any complete reliable point gate.

Matched cases: BROAD gains14 original successes versus AGG and loses none. TARGET gains7 and loses2. TARGET loses9 BROAD successes and gains none. For reliability BROAD gains9 AGG cases but loses all6 former reliable cases; its net +3 is not persistence of the old reliable set. TARGET gains3 and loses6 versus AGG. TARGET gains1 reliable BROAD case, loses7 and retains2. This is developed geometry and one training seed; correlated ticks/noise are not independent population trials.

Eight-family summaries (each family has eight fixed cases), in order028–035:

| Point | Original successes by family | Reliable successes by family |
|---|---|---|
| AGG |4,4,4,6,4,4,4,4|0,2,0,3,1,0,0,0|
| BROAD |5,6,6,8,5,7,6,5|1,1,3,2,1,1,0,0|
| TARGET |4,4,6,6,4,6,5,4|0,0,1,1,0,1,0,0|

Full per-command/backend/family and V2 side results remain in `VIRTUAL192.csv`. All primary scoring remains original V3, with first-completion semantics. Later exits and reversals remain reported rather than retroactively rewriting original success.

## Early/late support and intervention result

The early192 inputs are TRAIN-supported, not blind holdout. Both new arms consumed all of them through B; supplemental exposure differs. Historical FIT224 retains its actual per-arm membership: only32 exact conditions consumed,192 not consumed. Neither the whole FIT panel nor NEW_FIT can be renamed unseen.

Early per-output prefix medians, XYZ mm / SO3 rad, 96 outputs per task:

| Point | M1 | M3 |
|---|---|---|
| AGG |2.701 / .007830|2.120 / .006722|
| old S4FIT, descriptive only |4.264 / .009564|3.525 / .005254|
| BROAD |1.278 / .004833|1.115 / .003752|
| TARGET |1.324 / .005766|1.390 / .005415|

Versus AGG, BROAD improves prefix XYZ on88/96 M1 and89/96 M3; TARGET on91/96 and83/96. Prefix SO3 improves87/96 and87/96 for BROAD,81/96 and80/96 for TARGET. TARGET versus BROAD improves XYZ only37/96 M1 and23/96 M3, worsens59/96 and73/96; SO3 improves23/96 and22/96, worsens73/96 and74/96. There are no ties in these comparisons. Thus both recover early precision compared with the narrow S4FIT history, but TARGET is worse than the matched BROAD control. Early M3 retains one negative-direction output per point; zero is not asserted. Complete prefix/full deltas and command/family/j0-j1 strata are retained in paired tables, `SUPPORT_STRATA.csv`, `SUPPORT_FAMILIES.csv` and `DESCRIPTIVE_ANALYSIS.json`.

M3 terminal panel is 24 conditions from two TRAIN families with two frozen draws, not48 scenes. Prefix XYZ median over48 outputs: AGG8.954mm, BROAD5.218mm, TARGET2.819mm, old S4FIT2.668mm. Wrong directions are35/48 for AGG and0/48 for BROAD/TARGET/S4FIT. Over24 per-condition j0/j1 mean MSE values, TARGET-minus-BROAD median delta is **−1.7676252806703948e−5 m²**, favoring TARGET on20/24 conditions and BROAD on4/24. Corresponding paired prefix SO3 favors TARGET19/24, median delta−.00068239609rad. Full XYZ favors TARGET19/24; suffix XYZ19/24 and suffix SO3 21/24. However first-point XYZ favors TARGET only1/24 (worse23/24), and first-point SO3 only6/24. All first/prefix/endpoint/suffix/full metrics remain in the original per-output and per-condition tables; marginal medians do not replace the registered mean-MSE comparison.

Continuous terminal correction matters: teacher displacement median13.077mm; predicted displacement medians AGG4.713mm, BROAD5.176mm, TARGET9.132mm, S4FIT10.323mm. Signed progress along the teacher direction is respectively−1.839,+3.900,+8.576,+9.218mm. Raw teacher/predicted vectors and magnitudes for every draw remain in `LATE_CONTEXT2368.csv`, including tiny-displacement cases and the fixed direction guard. No positive direction is interpreted as complete path or pose hold.

The targeted hypothesis has a favorable terminal fitting result and unfavorable early/control/task results. TARGET does not achieve higher reliable task count or qualification and loses safety; the joint hypothesis is **NOT SUPPORTED** under this fixed design. BROAD's superior original completion is evidence against attributing improvement uniquely to targeted supplemental state selection. This comparison does not isolate data quantity, one axis, architecture limitation or a unique neural mechanism. Historical S4FIT has different total optimization and no new main panel; its reliable performance is UNOBSERVED here.

## Reference, installation, tracking and late failure

Saved transition analysis covers12368 new main query transitions and128 episodes. The reference prediction, installed reference, actual achieved pose and target error are separate columns with the same physical time index. E0 is the ideal mathematical EEF executor: accepted endpoint reference and achieved pose coincide, so same-tick tracking error is zero. It still fails reliable tasks; those failures cannot all be assigned to WBC tracking. E1 is the original .15s response executor, not a second robot.

E1 same-tick tracking medians/p95, XYZ mm and SO3 rad:

| Point/task | XYZ median / p95 | SO3 median / p95 |
|---|---|---|
| BROAD M1 |.539 /13.585|.001788 /.008099|
| BROAD M3 |2.349 /3.368|.001149 /.002188|
| TARGET M1 |.578 /14.265|.002050 /.008808|
| TARGET M3 |2.081 /3.451|.001394 /.002777|

M3 achieved-to-terminal-target query-time XYZ medians are BROAD E0/E1=53.413/58.408mm and TARGET55.955/59.308mm. These are descriptive whole-episode aggregates, including approach/follow, not final-window qualification. First reference point biases are small (M3 medians .238/.240mm BROAD, .275/.266mm TARGET), while later task failures remain. Rejected/collision-stopped transitions preserve missing actual execution rather than inventing achieved values. No difference between two independently minimized errors is called tracking.

M1: all32 episodes first-complete for each arm. BROAD exits after first success in27/32, exceeds final-window XYZ in13/32 and SO3 in26/32. TARGET exits in32/32, exceeds XYZ in16/32 and SO3 in32/32. Pose-hold failure is therefore not hidden by the first-completion success.

M3: BROAD has5/32 without observable entry and11/32 post-entry path/hold failures; TARGET13/32 without entry and12/32 post-entry failures. BROAD has4 late completions; TARGET seven safety stops and seven incomplete final windows. These flags overlap and must not be summed as a mutually exclusive partition. Independent entry/ordered coverage/reversal/shortcut geometry is in `TRANSITION_ANALYSIS.json`; full first-success exit and final XYZ/SO3 diagnosis is in `FAILURE_AND_POST_SUCCESS_DIAGNOSIS128.csv`. No failed suffix or late reversal was removed. Components retain all32 valid placements per point, diagnostic V3 success8/32 each; component success does not substitute for64 main cases.

## Legacy and CPU reconstruction

Original ROT2048 comparison is exact XYZ .015121491176463085m, SO3 .046877545849429575rad, zero REJECT. Legacy medians AGG=.013326206377524397m/.02406404308719176rad; BROAD=.01275840741308312m/.0243467906049097rad; TARGET=.012768678564995865m/.023939700207033714rad. All retain zero REJECT and PASS the original legacy gate. AGG is reported but never substituted as the official baseline.

`CPU_REPLAY.json` independently reconstructs all17152 new S4 raw/decode outputs, arithmetic and physical metrics. All128 new main and64 component trajectories reconstruct at20ms; maximum numerical trajectory difference is at most1e−12. Independent entry geometry is checked. Native V3 scorer, original gate and executor functions are shared and explicitly disclosed; this is not a claim of an entirely independent second implementation. CPU analysis uses bounded threads and no neural calls, rendering, WBC or physics.

Primary tables: offline2904, early576, main192, component96. Descriptive early context includes old S4FIT for768 rows; late four-point context2368 and per-condition means1184. Complete original/reliable/safety/legacy point decisions are in `PRIMARY_VERDICT.json`; full inventories and journal hashes in `QUALITY_OUTPUT_INVENTORY.json` and `FINAL_ARTIFACT_AUDIT.json`.

## Budget and preservation

| Actual new consumption | Count | Hard cap |
|---|---:|---:|
| Completed updates, incl. canaries |10272|10528|
| Training heads |184944|196608|
| Quality attempts |17152|18432|
| Quality S4 heads |68608|73728|
| Main/component starts |192|208|
| Observation frame pairs |13648|15360|
| Teacher / WBC / dynamic physics / hardware |0 /0 /0 /0|0 /0 /0 /0|

Zero technical quality retries. A pre-query model-view path materialization failure consumed no new neural query and was repaired without retraining. First valid outputs and actual early stops were retained. Prediction archive has80016 members, source bytes3058927142, tar3121684480 bytes. HF PUBLIC revision `eb52b2915bd865066964cad71700f5cfebac30ed`, prefix `evidence/PI0273/full-support-replay-20261005/predictions`, tree `62697e20d984e2851985da5c42f3884a42be09945838598fc48d0680c3a2a991`; anonymous whole-file and every-member streamed hashes PASS.

Four complete26-file Trainer archives, prefix `arms/PI0273/{arm}/checkpoint-{step}`, all full-file readbacks PASS:

| Trainer | Fixed revision | Tree SHA256 |
|---|---|---|
| BROAD2560 |b168fa13b9042daf8e7155f47bee4c5c1b71472d|bb3a68b7f23c7124556f67d34f93b5e809a3a9c6d3e78e02ff3ac26c8e0ba38d|
| BROAD5120 |5bed9a0d209ccaf23cfcd8bb28fb5a4c41a699c4|038a9b53de8d455a6a5a5c844a18538e14cd43f727ca3566929b694b880ac188|
| TARGET2560 |145a8b62810f107aede84983dee6914790cc30da|00f418aa65c876766201b00dac73d552e60e236b9ab3da253c07b4861c930843|
| TARGET5120 |dd5902dd8fb999f8891a8239069f8f3e6c6bb718|b05be5f98208edbbd80b7a78143ac1e3efcffc2b5fc637c93e6ecd3a931c7582|

Video files `PI0273_BROAD5120_VIRTUAL.mp4` and `PI0273_TARGET5120_VIRTUAL.mp4` use the first fixed main family028 and all eight chapters each, including failures and late behavior. BROAD has800 frames, TARGET736, fps5, explicit2x playback, actual96x96 RGB without re-render. TARGET's collision chapter retains36 observations through14.0s; its actual episode stop is14.4s, with no saved later observation fabricated. Watermarks state VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP, point/task/command/backend, actual timestamp, stop reason and NONE selection. The V3/reliable label is the final episode outcome displayed throughout a chapter, not a claim of success already at t=0. Full decode and all chapter first/middle/last frames were reviewed; contact sheets preserve that review. Video hashes and final summary archive/readback are in their receipts.

## Limits and next decision

This is single-seed, developed-scene virtual EEF evidence. It is not WBC, physical transfer, real-time deployment, a grasping demonstration or independent generalization. Formal selected/deployment remain NONE. The unchanged full-H40 gate and safety terminations were preserved.

The strongest next diagnostic hypothesis from existing arrays is persistence of correct pose/reference after first completion, with M3 entry/path issues retained as separate evidence. E0's zero tracking residual and both arms' M1 post-success SO3 drift argue against explaining everything by executor lag. Neither this report nor the local fitting benefit authorizes a new intervention. Stop for an independent transfer or one explicitly authorized targeted-repair decision; do not automatically add replay or tune a composite score.

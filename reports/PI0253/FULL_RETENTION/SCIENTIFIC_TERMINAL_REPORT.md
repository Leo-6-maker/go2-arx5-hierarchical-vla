# PI0253 scientific terminal: NO_ADMISSIBLE_CHECKPOINT

Authority: PI0253_LIVE_BRIDGE_TO_FULL_SUPPORT_GO, release82554098fb55b39ceca4b955252f8952c91c0dc0 / canonical5870278574. Original G M64 four-group PASS and TRAIN-A/C0 valid physical FAIL remain unchanged.

## Final original CALIB table

|Profile|Role|VALID|ACCEPT|WARN|REJECT|Original admissibility|
|---|---|---:|---:|---:|---:|---|
|Adapted1000|Original comparison only|1088|0|0|1088|Not a formal candidate|
|FULL2500|Formal candidate|1088|43|1034|11|FAIL|
|FULL5000|Formal candidate|1088|50|1029|9|FAIL|
|FULL7500|Formal candidate|1088|48|1029|11|FAIL|
|FULL10000|Formal candidate|1088|44|1034|10|FAIL|

All formal-candidate REJECT cells include INITIAL_ROTATION_DISCONTINUITY_REJECT. No full candidate satisfies the original complete1088/zeroREJECT admissibility rule. WARN would have been admissible, but positive REJECT counts are not. The baseline remains comparison only, not FULL1000 and not a fifth candidate.

The original selector froze NO_ADMISSIBLE_CHECKPOINT, selected_checkpoint=NONE, selection_frozen_before_wbc=true. Selection SHA256 e2d982b5bb0486a66d688293c8fcb26e5c2da4a4209f554598275a8c8c8063ae; provenance lock SHA256 f7b03be67084d970fb34bfff5aef1f35742634dff3ac3ab716fa85e352f7cd2c. Independent read-only diagnosis invoked the unchanged original select function (SHA25680af848239a5505836d7eeecb75d889403af4ac82c640ccf14e90d0b0703b607), reproduced its exact no-admissible exception, verified all five profile hashes and complete unique source sets, and did not create another selection.

Winner retention and new TRAIN one-chunk/receding are NOT_ADMITTED_NO_WINNER. New physical episodes=0, new achieved TCP=UNOBSERVED_DEPENDENCY_GATE_CLOSED. No G substitution, runner-up, alternative noise, extra training or additional query follows. This is a valid scientific stop under section7.2, not an engineering failure or a sim-success claim.

## Full training and retention context

PBS637051.hopper-m-02 completed the unchanged unique FULL_G10000 stage, exit0, exact suffix resume from2500. Four complete Trainer checkpoints2500/5000/7500/10000 retain their original fixed HF revisions. The 1000 observation is model-only, not a candidate or resume. Original exposure9792+208, source split/loss/optimizer/LR/S4/full gradients/free-generation/frozen range and RNG namespace remain unchanged. Raw10000-row recheck confirms actual sample/key/noise/loss/mode evidence, 40000 rollout forwards and40000 backward recomputes; recorded first-update all-four-state gradients are positive finite. Original update/frozen-parameter integrity remains separate evidence.

All640 logical retention cells closed (64G reuse +576 new), with each of the four formal candidates'128 first-valid cells independently replayed on CPU. All four candidates FAIL their own all64/old16×B0/B1 joint quality gates; zero REJECT alone does not mean quality PASS.

|Formal step|All64B0 median XYZ mm|All64B0 p95 XYZ mm|All64B0 median SO3 rad|Own four groups|
|---|---:|---:|---:|---|
|2500|54.544|194.305|0.050837|FAIL|
|5000|72.613|190.767|0.040674|FAIL|
|7500|54.862|190.460|0.033288|FAIL|
|10000|54.376|181.114|0.036091|FAIL|

The fixed early barrier did not trigger: FULL1000 and2500 B0 both had0/64 LINEAR_SPEED_REJECT; intermediate ordinary FAIL therefore continued to the preregistered10000 endpoint. No candidate was removed or model retrained based on quality.

Four original live-input/noise Q side predictions and separate CPU replay completed (4queries/16heads), with original gate ACCEPT. Their reference minimum target distances are0.0573847897/0.0647291220/0.0493921632/0.0564006719m at2500/5000/7500/10000. These are offline reference diagnostics, not achieved TCP, and did not enter training or selector ranking. Original live G actual minimumTCP0.1564976776m and reference minimum0.1049779477m remain the old physical FAIL; corrected same-tick mean tracking0.1374236374m is not a subtraction of minima.

## Accounting and evidence

D0=0queries; D1=8logical (2exact reuse+6new); retention=576new; full live=4new; CALIB=5440new; new sim=0. Offline new queries=6026≤6028, native offline heads=24104. FULLcanonical optimizer=10000, raw also10000 in this stage, plus the preserved637034 zero-update engineering attempt. No new parity query or expert-control episode; original identical-runtime controls reused. Older campaign raw updates and valid FAIL evidence remain historical, not reset.

Full10000 CALIB raw archive: repo leoyuer/pi0246-gate-pass-campaign, revision538b0cb87c3d9ad53bb0c93af223a083347bd37f, prefixevidence/PI0253/calib-full10000-20260929, nine files/136648814bytes, treee09044fac13dff0acf713845e75559c864c28e0c6c85fe2e206271a8b4c6c01b. All3267 source profile files in the lossless member-hash-verified tar; full fixed-revision content readback PASS. The other four profile archives retain their separate immutable readback receipts.

Selection/runtime/provenance/read-only-audit closure archive: revision68280a7ca4607b00d3bed278e66d105a4eaf833b, prefixevidence/PI0253/full-selection-20260929,28files/110407bytes, tree000ddc405eb2a4555b18d76087b7c9e70ef81e47b5aa475c7b134b554ddf99e9, full-content readback PASS.

Science queries are now closed. Remaining strict Trainer content-readback storage companions and final requirement audit are still being completed; this report does not claim overall thread completion yet. All formal HF assets remain protected. No DEV/F, DAgger, real hardware, WBC/gate/selector change, monitor/mail or merge.

# PI0279 — Learn release, clearance and stable post-placement feedback

Date: 2026-10-07, Asia/Singapore.
Authority: **PI0279_SINGLE_TASK_RELEASE_CLEAR_FEEDBACK_FINETUNING_GO_V1**.
Reviewed source/evidence SHA: `97ccfe986a3c34a777e0fead22dd4e23ac7c108f`.
Predecessor terminal: Issue #11 **6035812443**, PI0278 release `83af9099f3bab9a4ac08eaa85380c2cfd6351f34`.
The new Issue #11 canonical binds this document and `reports/PI0279/design/REVIEW_AND_EXECUTION_LIMITS.json` at one immutable release. Research charter #5994405991 remains binding.

## 1. Decision: finish one physical task, not another scene-expansion campaign

**GO for a bounded same-task physical counterfactual/data pilot, followed automatically on data admission by one CLEAR4096 GR00T fine-tuning lineage and teacher-free physical qualification.** All learning/evaluation concerns the existing fixed-language source-Panda StackCube task. There is no new scene, object, camera, task, target-WBC or hardware campaign. The immediate deliverable is a locally loadable policy that itself generates EEF/gripper references completing the original task and leaving the objects stable.

The intervention is new post-placement reference demonstrations from learner-visited states: open, move the tool clear, and retain a fixed safe reference. It is NOT online teacher takeover, a stop token, output smoothing, a success-triggered runtime freeze or a relaxed scorer. It changes the TRAIN data and terminal teacher behavior explicitly, while retaining the final effective loss, processor, model scope, inference and physical execution contracts. A matched physical continuation probe is included; a matched additional training control is deferred for the user's policy-first priority. Do not claim isolated causal gains from data versus extra optimization.

PI0278 and PI0277 remain sealed failures with all historical results/counters intact. The extra budget in section 10 is new explicit authorization, not unused old capacity or a reset. Their old clock cutoffs are not inherited or silently bypassed: the new driver must bind PI0279. No new clock-time success promise is made.

One existing main owner on `pi0267/executor-decoupled-recovery`. Read actual current jobs/locks/resources at adoption; do not start a competing autonomous goal, duplicate qsub, archive monitor, or new long-lived branch. Preserve legitimately newer work. Protocol publication is not a run receipt. Commit implementation and source bindings before execution, without editing files used by sealed or active predecessor workers.

## 2. What the final evidence actually supports

Latest terminal ledger read: 2026-10-07 18:14:49 SGT, #6035812443. The training job651626 actually ended11:12:09 SGT. The reviewer has no SSH/live scheduler access.

COVER93 persistent TRAIN4:512=2/4;1024 unmeasured/backlog;2048=0/4;4096=2/4. All12 episodes were full32s/640-control executions. At4096 all4 lifted, reached on-top and had first native success. All4 had zero IK holds and no gripper-envelope termination. The final qualification-window subpredicates are:

| seed | on-top /41 | static /41 | not-grasped /41 | result |
|---|---:|---:|---:|---|
|10000|41|12|25|FAIL|
|10001|41|25|41|FAIL|
|10002|41|41|41|PASS|
|10003|41|41|41|PASS|

The final failures are not failures to reach the goal geometry, nor explained by final IK/gate rejection. Static/not-grasped are actual predicate failures, not proof by themselves of a unique contact mechanism. Aggregate contact counts do not identify which link disturbed which cube. Raw-command-to-installed XYZ RMS is0.0495–0.0614mm; installed-to-achieved full-episode XYZ RMS is3.62–4.88mm. These RMS values are not the roughly1mm control-end median quoted separately in the terminal report. Do not conflate means/medians/RMS or add them algebraically.

All93x4 source temporal strata were actually consumed as prescribed; all8192 microbatch proofs and final559 frozen payloads passed. Source group3 full-H40 XYZ fitting median improved4.65 to3.62mm, but persistent task admission did not. Expert20–30s states are mostly stationary duplicates. Thus broader chronological replay is not demonstrated learner-state recovery coverage. It is incorrect to keep saying no terminal data existed, or that another full-bank repeat must solve the task.

Actual source teacher code opens the gripper and repeats its final commanded pose to640 controls. It does not independently guarantee clearance from the stacked object under a deviated learner grasp/pose. A plausible mechanism is continued near-object commands, renewed contact/grasp, or unstable placement under closed-loop noise; each must be checked against actual arrays. A physics/measurement floor or unstable object support remains an alternative until the paired continuation pilot distinguishes them. No universal submillimetre requirement or proven base-model limitation follows.

## 3. A0: short existing-array audit and explicit binding

Reuse the sealed final four COVER934096 episodes, commands, installed targets, query outputs, observations and contacts. Do not rerun all historical policies, replay all archives or wait for HF. CPU reconstruction must identify, over8–32s and particularly28–30s:
- each static/not-grasped transition, cube velocities and supported on-top state;
- actual tool/cube clearance and named link/cube contacts when available, commanded versus measured opening, regrasp intervals;
- requested/installed/achieved motion, first/last point of each executed prefix and successive replan changes;
- whether contact or reference changes precede object disturbance, versus spontaneous instability after verified separation.

Use object/phase truth only offline or inside the TRAIN-data teacher, never inside the learned inference path. Lack of a saved force or exact shape fact must be labelled unobserved. No inference from contact-count correlation alone.

Bind local COVER934096 weights plus effective processor/statistics/mapping/decode and source versions using the predecessor final handoff/inventories. Preserve the latest full parent on Hopper. The Core model-only view is not an optimizer-resume source. Compare actual __file__/SHA and reconciled source manifests, not a similarly named helper. Record current personal quota and atomic-save headroom; do not infer GPU availability from a finished job or user quota from shared df.

## 4. A1: fixed paired physical continuation pilot and actual recovery supervision

### Fixed scope
Use ONLY the four original TRAIN initial states10000..10003 and the final COVER934096 saved predictions. Use fixed fork times **8,16,24s** (control indices160,320,480), chosen now for all successes and failures alike. This creates12 paired slots, not12 independent tasks. No extra seed search or score-dependent fork-time replacement.

For every slot run two physical continuations, each from the same native reset and through the original32s horizon:
1. **RECORDED:** replay the original installed command sequence through the original controller, including the recorded learned suffix. This is a physical replay/control, not a fresh closed-loop policy query.
2. **OPEN_CLEAR:** replay the same recorded prefix, then apply the one teacher below for the suffix.

Total scheduled starts24, all in existing CPU physics/CUDA rendering stack. Use source-native pd_ee_pose, base/world/TCP conversion, row-rot6d and gripper transport unchanged. Replay original full recorded chunks/gate decisions when replaying learned commands; do not concatenate unrelated replans into a fictitious H40 and claim identical admission. No new neural or planner call is needed to reconstruct prefixes.

Replay is the default rather than manually setting q/cube state. At each fork compare paired prefix robot q/qdot, TCP, both cube poses/velocities, simulation clock and native predicates against the saved trajectory. Position/joint coordinate residuals must be <=1e-5 in their native SI units, rotation geodesic residuals<=1e-4rad, velocity residuals<=1e-4 in native units, and task predicates/clock must agree; report dimensions separately. These are prospective reconstruction-admission tolerances, not relaxed task tolerances. Check throughout the final0.4s before the fork as well as the fork itself. If replay is inconsistent, isolate an engineering/source mismatch; do not hide it by resetting contact state, replaying different actions, widening tolerances or selecting another seed. Native simulator snapshots may optimize prefix reuse only after equivalent controller/RNG/contact-history continuation is actually checked and counted. No teleportation or welding of objects is allowed.

At a teacher fork require the original on-top geometry true over the preceding0.20s and finite physical state. This eligibility is for privileged training-data construction only. An ineligible fixed slot is retained as ineligible and is not substituted. It does not change the denominator of any learned-model evaluation.

### Prospective OPEN_CLEAR teacher, fixed before native trials
Let T0 be the ACTUAL world TCP at the fork, not an old desired pose. Keep T0 orientation and horizontal position fixed. Command gripper fraction1 throughout the teacher suffix (original actuator mapping only).
- Controls j=0..7: hold T0 while opening (0.4s).
- Controls j=8..31: raise TCP by0.08m along world +z over1.2s, using s=(j-7)/24 and h(s)=10s^3-15s^4+6s^5.
- Controls j>=32: hold the SAME world target Tclear,0.08m above T0, until the original32s end. Do not repeatedly reset the target to measured TCP, which would encode following drift rather than restoring behavior.

Generate the full future reference using that fixed function; check original complete-H40 gate and command envelope at each replan. The planned hold can be evaluated beyond32s for gate lookahead, but only commands actually recorded within the episode are eligible training labels. Never invent padded future labels. No IK/gain/stiffness/friction/mass change, dynamic target attraction, box pose change, new runtime gripper clipping, or oracle success stop. Native success does not alter the teacher clock/path after the fork. A failed gate/IK/task outcome remains a failure, not a prompt to tune the0.08m/0.4s/1.2s values.

This teacher encodes release and geometric clearance, not force-optimal manipulation or universal recovery. Its sufficiency is an experiment. In particular, opening/retracting from an unsupported placement may fail; do not label such trajectories as positive demonstrations.

### Pilot/data continuation gate
A positive OPEN_CLEAR suffix requires full native execution and the unchanged persistent object success at28–30s, plus measured command/state evidence of release and actual clearance: during the same window actual opening>=0.075m and TCP z>=fork TCP z+0.06m. These extra checks qualify the teacher data only; they do not weaken or replace model task scoring. Require no teacher-side gate rejection or IK hold. Record contacts/saturation and all failures.

To proceed automatically to training, require **>=10/12 eligible, valid positive OPEN_CLEAR suffixes, at least2/3 in every initial state**, with all12 RECORDED controls completed or truthfully resolved. At least one of the two original failed initial states must show a paired conversion from failed RECORDED continuation to successful OPEN_CLEAR. Confirm existing persistent failures in the replay rather than assuming reproduction. If the control fails to reproduce or withdrawal does not help, report that evidence and stop this particular training hypothesis; do not waste4096 updates on an unvalidated teacher. No new scientific GO is needed when this registered gate passes.

The paired test identifies the effect of a scripted suffix on these reconstructed states, not the effect of learned recovery or teacher-free success. Stop/hold/open teacher success is never counted as model success.

## 5. Dataset: learner-visited recovery, not shifted old labels

Record fresh two-view RGB, actual StateV1/TCP/opening, raw teacher commands, installed targets and achieved physics on each OPEN_CLEAR branch. Create H40 labels only for anchors at/after its fork with40 genuine future commands inside640 controls. Do not use the predecessor RGB for changed suffix states. Prefix learner mistakes and RECORDED-control actions are diagnostic evidence, not positive behavior-cloning labels. If all12 teacher suffixes pass, at most3372 such H40 windows exist; they are correlated windows, not3372 demonstrations.

Keep all failures in a separate source inventory. Positive training selection uses independent source-task teacher validity, never target Go2-WBC success. No Go2 teleoperation or WBC call is needed.

Retain the16 original FOCUS TRAIN episodes10000..10015 as task-acquisition replay; no other77-episode coverage campaign this round. Freeze8192 microbatches before training:
-50% original replay; each old phase0/1/2/3 exactly1024 visits, balanced by episode then full temporal strata, with genuine unchanged labels.
-50% admitted OPEN_CLEAR suffixes. Balance the four initial states first; within each state balance its admitted fixed fork slots and the three suffix anchor categories OPEN(j<8), CLEAR(8<=j<32), HOLD(j>=32). Across4096 recovery visits, each state receives1024; category and slot counts differ by at most1 within the feasible round robin. Choose eligible anchors using fixed SHA256 ordering `PI0279|27903|seed|fork|category|uid`, cycling only after exhaustion. Preserve transition anchors and representative late anchors; do not let long constant holds dominate by duration.

Audit exact condition/target duplicates and original-versus-new terminal multimodality. Different valid terminal trajectories are not automatically a codec bug; neither are duplicate late frames new state diversity. A genuine coordinate/time/label contradiction blocks. Do not silently relabel old data or refit statistics to hide conflicts.

Freeze new input/label/mask proofs through the ACTUAL inherited processor. Verify CPU roundtrip, full-H40 gate, measured-command distinction, non-unit rotation and unchanged masks. A finite reference outside an empirical TRAIN quantile is not by itself an excuse to clamp it; original codec/gate semantics govern. If physical/normalization semantics cannot represent the admitted teacher, stop for that specific contract decision. Statistics, camera processing and action representation remain unchanged. New UIDs/cache keys must include source and processor identity to avoid stale cache reuse.

Metadata such as fork time, episode seed, teacher phase, privileged on-top/contact and teacher identity is forbidden in model inputs. No new history or phase token. The fixed task instruction stays unchanged.

## 6. One actual accelerated training lineage

Start **CLEAR4096** from the final COVER934096 effective weights (not from the prettier historical checkpoint). Use one fresh optimizer and4096-step cosine schedule; afterward preserve optimizer/RNG/scheduler/cursor on continuation. Exact parent paths/tree/files are read from the final PI0278 handoff and written into PI0279 admission before launching.

Keep471 trainable/559 frozen tensors, batch1/accum2,lr1e-5,warmup10,weight_decay1e-5,grad_clip1.0,4-step S4 and the actual inherited R1 precision. Wire EEF commands still pass through the existing processor/relative transforms/decode; do not assume world-frame wire values mean the neural representation has no relative transform.

Loss from the first step remains:
`L_native +0.4 L_position +0.1 L_increment +0.1 L_rotation +0.1 L_speed +0.1 L_generated_gripper +0.1 L_box`.
Preserve original reductions/units: generated gripper80% first8/20% suffix with0.10 scale; box uniform fullH40 with0.05 scale. Reuse actual source-bound helpers and gradient path. No extra head, sigmoid/binary policy, additional S4 generation, loss search, backbone unfreezing, precision/batch change, compile, DDP or optimizer-kernel experiment.

Use qualified FAST_CACHE2/ordered CPU workers/pinned memory and stream new CPU entries as needed. Do not benchmark more workers: predecessor median data wait0.000142s is already negligible. One real continuous8 versus4-save-fresh-process-restore4 check is authorized:16 completed updates, same new data/actual noise and inherited exact criteria. No repeated performance tournament. Once data and this gate pass, formal training is already authorized; do not stop at DATA_READY.

Snapshots512/1024/2048/4096, with full local recovery at least every1024 and at safe pauses. Training and immutable-snapshot evaluation may overlap on separate legal resources; otherwise serialize. At most one pending snapshot. Unrun superseded checkpoints may be explicitly skipped for transfer backlog; valid failures and active episodes may not be discarded. The final4096 screen is required unless a prior candidate is frozen or a concrete blocker ends the run.

Continue exact replacement-tensor transport over a verified local parent when compatible; enumerate changed buffers and reconstruct complete source hashes. Reuse source-proven unchanged bytes. Fall back to ordinary direct model-only transfer if needed; never route through HF or transfer full optimizer state for each screen. No new serving-stack project. Check quota for atomic saving; preserve parent, latest/previous recovery and referenced candidate. Retire only proven redundant/disposable payloads with inventory, never unique data or unresolved failures.

## 7. Teacher-free qualification, not training with an online finish controller

Freeze a48-row diagnostic panel before new learner queries:24 original TRAIN windows (four initial states x first anchors of four phases plus20s/29s terminal anchors, deterministic unique fallback) and24 recovery windows balanced by state/category/fork. Verify sufficient unique eligible rows, do not substitute based on prediction error. Query parent once per non-reusable row and each evaluated snapshot once with fixed stored noise. Fitting is a diagnostic, not success.

Each of up to4 snapshots runs original TRAIN4 from native initial states for32s, with original H40/20Hz/execute8/replan0.4s behavior and stored matched episode noise. **At no point may the learned policy call OPEN_CLEAR, read the on-top/static/grasp truth, use a scripted8s switch, freeze after success or substitute a teacher action.** It must itself generate both the grasp/place path and subsequent safe EEF/gripper references from allowed RGB/actual state/language. Log raw, installed, achieved and object results separately.

Keep original success: native on-top AND static AND not-grasped at all41 control samples28–30s, plus full32s interface-valid execution. Track loss after first success and30–32s flicker separately without retrospectively changing the old verdict. Old failures remain failures.

The existing qualification ladder remains TRAIN>=3/4 -> DEV>=6/8 -> one untouched CONFIRM16>=12/16. Existing DEV8=30000..30007 checks reproducibility within the same task, not an expanded scene campaign. At most two DEV candidates: first TRAIN-qualified checkpoint>=1024, and then4096 if needed and not already evaluated. Do not repeat the same candidate. Freeze the first qualified candidate, verify its local identity/input/decode/native receipts, then open the original still-unused30032..30047 once. Verify absence of previous access before creating confirmation noises.30048..30063 and any new scene generation remain locked.

If the user-facing intermediate milestone is TRAIN-only success, label it LOCAL_TRAIN_TASK_PROGRESS, not usable/generalized policy. Confirmation feedback cannot be used to select another checkpoint or train again. Full matched CONTROL, broad generalization, new tasks and WBC composition remain deferred.

A confirmation pass may set only a scoped `sprint_candidate` with immutable identity and `SOURCE_PICKPLACE_PILOT_CONFIRMED`; formal selected/deployment stay NONE. This is non-real-time source-Panda evidence, not Go2 or hardware qualification.

## 8. Interpretation and finite stopping

Do not automatically repeat another4096 source-only run. This goal tests a particular demonstrated terminal behavior on reached states.
- If withdrawing the tool does not stabilize the physical paired branches, investigate placement/support/physics rather than forcing a learner to copy an invalid remedy; stop the training lane with actual pilot results.
- If a valid teacher and its input/label conversion exist but learner-window fitting remains poor, report the measured fit/phase conflict and stop at the training cap.
- If teacher-window fitting is good but teacher-free post-placement behavior still fails, report reached-state/observability/noise/feedback evidence without declaring a unique cause.
- If acquisition regresses, report retention failures; do not silently replace the new-data ratio or parent.
- If a candidate is qualified, stop useful training at a recoverable optimizer boundary, complete one confirmation and handoff before any expanded-scene work.

No third branch, new teacher geometry, extra loss, seed replacement, reset budget, source-controller retuning or hidden success stop is authorized. Ordinary same-semantics code fixes may proceed with versioned receipts; changing these scientific contracts needs another explicit decision.

## 9. Local-first delivery and audit boundaries

HF bulk/public archive, old matched CONTROL, noncandidate exhaustive video/replay and long retrospective reports remain DEFERRED_USER_PRIORITY. Native scores, input/policy identity and durability are not optional. Use fast saved-native decision receipts while nonselected full S4 replay runs later; do not label deferred checks PASS. For the selected candidate, finish its own saved-input/S4/decode/native verification and local reload before confirmation/usable classification.

Deliver paired-branch comparison with denominators/eligibility; actual recovery data and teacher/codec proofs; original/new data-consumption counts; recoverable model and real inference command; all learned TRAIN/DEV/confirmation attempts; event/predicate/contact table; compact success and failure recordings; phase timings and current quota. Clearly separate replay examples from fresh inference, scripted suffix success from learner success, and source exposure from new feedback states.

One owner adopts actual resources. H200 terminal651626 is not an active allocation. Use one legal H200 training submission plus existing Corelab source evaluation; no login-node neural work, duplicate jobs, new cloud spend or background monitor. Publishing this GO does not itself start any agent or remote job.

## 10. Explicit additional budget

Training:4096 formal +16 canary=4112 planned completed updates. Hard caps4096 healthy formal,4176 all completed,76032 actual training heads (planned74016 at18/update; extra parity/recompute counted). No new generative teacher model is used.

Source pilot:12 forks x2 continuations=24 starts,76800 physics steps,15360 controls. OPEN_CLEAR teacher controls max4*((640-160)+(640-320)+(640-480))=3840; recorded learned-prefix/suffix replay is separately counted. All source failures stay in the denominator. Positive recovery H40 windows at most4*((640-160-39)+(640-320-39)+(640-480-39))=3372.

Learner evaluation max4*4 TRAIN +2*8 DEV +16CONFIRM=48 episodes,153600 physics steps,30720 controls. Diagnostics max5*48=240 queries (parent+4 snapshots), learned48*80=3840, total4080 planned quality requests. Exact byte-identical output reuse subtracts new calls, not logical coverage. New teacher/pilot uses zero neural queries.

Overall nominal:72 starts,230400 physics steps,46080 controls,<=46152 RGB pairs. Hard caps **76 starts,250000 physics steps,50000 controls,51000 RGB pairs,6400 teacher controls,4608 quality requests/18432 quality heads,65536 CPU input attempts**. At most4 same-condition technical replacements within ALL caps; none for valid quality failure or ineligible forks. Failed/partial attempts, reset physics, prefetch and extra parity calls remain counted. Source and learner ledgers are separate and jointly indexed.

Existing PI0277+PI0278 totals remain12288 formal/12344 all completed/222309 training heads. New authorized formal allowance raises the combined limit to16384; it does not relabel old work. No transfer of unused prior caps. WBC/hardware calls0. No broad task/scene generation. No clock deadline overrides the requirement for real qualification.

## 11. Implementation and review scope

Implement under `scripts/pi0279_execution/` and receipts under `reports/PI0279/`. Reuse sealed native/processor/physics helpers through explicit imports and configuration, not unreviewed global string replacement. State machine: adopt_actual -> saved_terminal_audit -> fixed_paired_suffix_pilot -> data_and_codec_admission -> short_resume_canary -> CLEAR_train_and_fixed_screens -> freeze_verify_confirm_once -> local_handoff_and_issue_readback. Data admission automatically authorizes the next listed stage.

Actually reviewed at the audit SHA: PI0278 final report, final completion/handoff indexes, final physical CSV, current training entry and original source-worker step/finish/teacher hold/gate logic; latest Issue #11 terminal and branch. Prior saved reports support the stationary late-state observation. Reviewer did not SSH, replay remote raw arrays, inspect all videos, profile GPU or execute a new teacher/model/physics test. Proposed release-clear behavior, timings and pilot thresholds are prospective engineering choices, not already measured fixes. The final false static/not-grasped counts are facts; their causal mechanism and effectiveness of this remedy remain testable.

Startup: read the canonical release, preserve sealed failures and local-first priority; run the12 fixed paired post-placement continuation slots, validate genuine OPEN_CLEAR supervision, then automatically fine-tune one CLEAR4096 from COVER934096 with existing S4 loss/runtime and task replay. Evaluate learned policy only, with no teacher/clock/success takeover; retain original task/gates and new finite counters. Deliver one confirmed source-task candidate or an explicit bounded failure before considering scene expansion.

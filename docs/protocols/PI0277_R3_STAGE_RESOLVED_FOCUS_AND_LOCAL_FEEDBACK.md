# PI0277 R3 — Stage-resolved single-task repair and fast local physical feedback

Date: 2026-10-07, Asia/Singapore.
Authority: **PI0277_R3_SINGLE_TASK_FOCUS_AND_FAST_PHYSICAL_FEEDBACK_GO**.
Reviewed code/evidence SHA: `255b53cc6d9559c1c9fb0e1d15d6b0e9d8cf4544`.
Parent: PI0277 R2 release `1a20200ff066ef5a003650110cd20106f9a4bf64`, Issue #11 #6015575969; underlying V1 math remains release `63151a4548243ab0fb4ba172eea6f7d92e02c576`.
The new canonical binds this file and `reports/PI0277/design/R3_REVIEW_AND_EXECUTION_LIMITS.json`. Research charter #5994405991 remains binding. R3 is an explicit continuation/amendment of the SAME PI0277 budget, not a new independent training campaign.

## 1. Decision and immediate goal

**GO: complete/adopt the actual GRIP4096 physical screen; use the already-authorized sole FOCUS lineage when its actual failure trigger is met; shorten the local result-to-decision path and diagnose grasp/lift/installation from saved arrays.** Do not add another loss, controller, dataset or task in this amendment. The aim remains a teacher-free, locally loadable StackCube pick/place prototype with actual object success.

There is no basis to label GRIP4096 a task FAIL before its physical result exists. Do not re-train GRIP, re-run its valid earlier conditions, or choose GRIP2048 merely because its video was inspected. If a newer qualifying candidate already exists, adopt it and advance to the fixed development/confirmation route instead of following a stale failure branch.

Preserve R2's 2026-10-07 08:00 SGT handoff target and 07:00 no-new-training boundary. At07:00 finish the current optimizer update and perform an atomic recoverable save; do not let a previously started long segment silently bypass the boundary. This is a planning cutoff, not a guarantee of learning success. If adopted after the cutoff, inventory actual work and report; no automatic fresh overnight budget or deadline extension.

HF bulk transfer/readback, CONTROL, broad offline panels, long-form reporting and multi-scene production remain DEFERRED_USER_PRIORITY. Local recovery, input/action semantics and genuine physical task scoring remain mandatory. WBC/hardware calls remain zero.

## 2. Actual evidence and corrected interpretation

Latest ledger read: #6021094740, 2026-10-07 00:49:31 SGT. Latest reviewed branch includes resource evidence observed00:46:01 SGT.

- GRIP4096 training completed4096 updates/8192 consumed microbatches. Job650860.hopper-m-02 was terminal F/Exit0. Complete local3072/4096 Trainer states and an immutable4096 inference snapshot exist; direct transfer to Corelab was underway. A finished job is not a currently allocated idle H200.
- Journal audit verifies all consumed input/label proofs, observed noise identities,471 trainable/559 frozen scope, four S4 steps and total objective `L_old + 0.1 L_gripper_generated`. Raw-noise regeneration and final frozen-tensor byte comparison were explicitly unobserved in that audit. Do not silently promote these to PASS.
- GRIP512 TRAIN4:1 sustained grasp,0 sustained2cm lifts,0 on-top,0 first/persistent stack success.
- GRIP2048 TRAIN4:4 sustained grasps,2 sustained2cm lifts,0 on-top,0 first/persistent stack success; all4 full physical episodes completed. This is real limited behavioral progress, not a usable policy or an isolated causal loss result.
- For GRIP2048 seeds10000/10001, IK holds are0 and command-to-actual position RMS is about6.67/6.89mm, but no sustained lift. For10002/10003, lift occurs and IK holds are359/446 out of640 commands each. Command-to-installed-FK RMS is29.34/26.79mm while installed-FK-to-actual RMS is4.82/4.46mm. All4 together have805/2560 IK-hold commands. The two error measures are not algebraically additive.
- GRIP4096 physical task quality remains unobserved in the last completed record. FOCUS was prepared but not triggered. Its frozen draft uses16 original TRAIN episodes,8192 visits/2048 per group,3648 distinct used windows; these are not new demonstrations or all9616 eligible windows.
- Production measured median update1.7490845575s and median loader wait0.0001728146s, peak GPU allocated34,377,103,872bytes. The actual qualified route is FAST_CACHE2. No paired full-GPU speedup claim follows. More loader workers are not justified by the tiny observed wait.

The working diagnosis must distinguish: (A) grasp/no lift with no IK holds; (B) lift followed by reference-installation problems; (C) failure of task progress or replan consistency. Neither gripper error, IK nor unseen-scene coverage is established as a unique cause. Expert EEF success proves the existence of an executable task reference, not that every learned reference is feasible.

## 3. Adoption: preserve useful work and avoid waiting for another coordinator

One existing main owner, same branch and directories. Read actual jobs, training/quality journals, snapshot transfers and quotas ONCE at adoption; update when a state transition matters. No second autonomous goal, archive watcher, PR77 prerequisite, duplicate qsub or helper-signature gate. Existing successful checks and actual partial work are reused.

Preserve live GRIP4096 evaluation and any legitimately started FOCUS. A newer useful run is never reset to follow this document. Stage changed dispatch/decision code in an explicit R3 execution seal; do not overwrite a source file used by a running frozen evaluator. The R2 literal release in a saved checkpoint or launcher is its provenance, not permission to ignore this amendment. An adoption manifest must distinguish original runtime release, new governing authority and any actually changed entrypoints.

Resolve cumulative counters from the shared top-level journals, not whichever lineage-local summary is smaller. The reviewed resource receipt reports4120 R2 completed updates while earlier project-wide records include additional qualification work. Keep both scopes and all failed attempts; no erasure/reset or guessing remaining balance. Formal GRIP4096 is established; remaining healthy formal capacity is at most4096, reduced by any newer legitimate work.

If GRIP4096 is still transferring, continue the one existing direct transfer and verify its manifest. Do not start another copy, re-download donor weights, or upload to HF. If it is already present, proceed immediately to the fixed FIT16/TRAIN4 runtime. Do not wait for a retrospective training report, video montage or public archive.

## 4. Fast physical decision receipts, separate from full delivery

Implement a **FAST_TASK_PANEL** decision path by reusing the existing `r2/panel.py:scores` native reconstruction. Do not invent a PASS status for code that was not run.

For each completed fixed episode, verify its immutable policy binding, expected seed/initial-state identity, original input whitelist, actual query/head accounting, valid source result and raw-array checksums. Recompute the native object predicates from saved100Hz arrays, compare saved predicates, retain all failures, and check full640-control execution for each counted success. Compare policy inputs with saved same-tick RGB/state and check original noise identity for the installed queries; cached proof reuse is permitted only for identical immutable bytes. No new model/physics calls are required for these checks.

The resulting receipt must name point/lane/cases, first and persistent successes, grasp/lift/on-top events, interface validity, array/policy hashes and exact checks performed. Use `PASS_NATIVE_TASK_RECONSTRUCTION_FOR_DECISION` with `full_S4_replay=PENDING` when full saved Euler/decode replay has not yet run. This is sufficient to trigger FOCUS or a development screen. **Do not require a completed video or full321-output S4 replay before starting useful follow-up training.** The old full `PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES` receipt can still be reused if already available.

Adapt `prepare_focus.py`/DEV decision guards explicitly to accept this truthful R3 receipt after verifying its required fields; do not merely replace the expected status string or bypass physics checks. Unknown/inconsistent semantics still block. A known semantic contradiction must be isolated and amended, never silently repaired as part of FOCUS.

For the ONE selected candidate, complete its saved-output decode/input verification, per-case native scoring and local reload identity BEFORE confirmation/usable-candidate classification. It remains independent of HF and presentation-video completion. Nonselected full trace replays and videos are a clearly marked backlog, not discarded evidence. Ongoing healthy full replay need not be killed when it is not contending with useful work.

## 5. Complete the primary decision and immediately take the correct branch

Finish/reuse GRIP4096 FIT16 and TRAIN4. Never re-run valid GRIP512/2048 episodes. If TRAIN persistent success is>=3/4, run the registered DEV8 as the first GRIP candidate; if DEV>=6/8, freeze it and perform R2 confirmation. Do not insist on FOCUS when a candidate already meets the goal.

If TRAIN<3/4, or its first DEV8<6/8, this is the existing R2 trigger for FOCUS. Prepare/adopt FOCUS from actual GRIP4096 weights immediately, without waiting for a new scientific GO, matched CONTROL, HF readback or all-frame video encoding. If FOCUS already started legitimately from a different R2-allowed trigger, keep its recorded parent and state; do not restart to force a new parent.

FOCUS is the **sole** extra lineage, not a third search after another FOCUS. Use the already frozen16-episode membership and deterministic8192-visit phase-balanced stream. Fresh optimizer is used once at the lineage start; evaluate without resetting it. Hyperparameters remain lr1e-5/cosine4096/warmup10/batch1/accum2,471 trainable559 frozen, unchanged S4=4/R1 precision and task processor. Objective remains exactly the GRIP objective. No new labels, hard-example score selection, phase input, teacher takeover, new head or camera change.

Use the qualified FAST_CACHE2 logic and verified cache entries. The authorized FOCUS sampler change requires its short continuous8 vs4-save-freshresume4 check once (16 completed updates); reuse any actual completed equivalent FOCUS check. Do not rerun the REF/cache speed tournament or GRIP qualification. Respect R2's cumulative new-qualification ceiling64 and total PI0277 caps.

Publish FOCUS inference snapshots at512/1024/2048/4096, within the existing maximum8 snapshot screens across GRIP and FOCUS. If an older not-yet-started milestone is superseded by a newer snapshot under an actual transfer/evaluation backlog, it may be skipped with `NOT_EVALUATED_BACKLOG`; do not skip a valid failure or stop an in-flight episode to hide it. Keep at most one pending transfer/evaluation snapshot. An evaluated checkpoint remains immutable. Give the first completed physical condition and incremental result table to the owner immediately.

A running FOCUS may be safely stopped at an optimizer boundary once a candidate has met the registered development screen, prioritizing frozen confirmation and handoff. No unseen confirmation feedback may be used to alter weights or choose another candidate.

## 6. Minimal saved-array diagnosis: localize the next task failure without turning it into another project

Perform this CPU-only diagnosis on the already saved TRAIN4 primary and any evaluated FOCUS outputs while useful compute proceeds. Limit inputs to already produced task panels and existing TRAIN teacher records; no new teacher, simulator, model or TEST access. Do not repeat the completed48-case A0 as a prerequisite.

Write one per-query/per-stage table containing:
1. raw world EEF targets at indices0/7/8/15/39; commanded opening over the executed prefix; actual TCP/opening before and after execution; maximum/achieved red-block lift, continuous grasp interval and any drop;
2. raw-command to installed-FK and installed-FK to achieved errors, source IK-hold spans, requested position/orientation increments, contact and joint-limit diagnostic facts actually saved;
3. whether reference requests lift/transfer in the executable prefix or only in the later horizon, and whether fresh replanning repeatedly defers the same future progress; report the actual sequences, not an invented binary mechanism label;
4. timing of first lift, first IK hold and first loss of grasp/on-top relative to each other; an IK count alone does not prove cause or physical impossibility;
5. same-input FIT16 prefix position/rotation and OPEN/CLOSE command errors at each evaluated candidate. Distinguish truly matched noise from a valid different stored noise. GRIP2048's10000-a131 output used n1 in its receipt; retain it, and do not call an n0/n1 comparison paired. No re-query is granted just to obtain a prettier comparison.

Use green/red object truth only in OFFLINE evidence reconstruction, never as policy input or controller feedback. Source teacher phase is not the true phase of a diverged learner. A small distance to the green block does not equal correct stacking or release. Non-grasping contact, off-centre grasp and local IK failure are hypotheses unless the actual recorded geometry/contact supports them. Do not infer stiffness/contact forces from impulse summaries alone.

Interpretation branches: no lift plus no commanded upward progress suggests task progress/feedback learning; requested upward progress without secure object motion suggests grasp geometry/timing; growing raw-to-installed errors with small installed-to-actual errors suggests reference feasibility/IK installation; good teacher-window fit but poor reached-state actions suggests a feedback-state gap. These are evidence-guided alternatives, not pre-authorized changes to the controller, horizon, labels or model architecture.

## 7. Keep the GPU/evaluation loop useful and locally recoverable

FAST_CACHE2 is already qualified and its measured data wait is tiny. Keep it; no4/8-worker tuning, new batch size, TF32/FP8, compile, fused optimizer, DDP, neural-feature cache or fewer S4 steps. Report actual time to snapshot, transfer, model load, physics panel and decision separately from update throughput. Do not convert a CPU cache benchmark into a claim of overall H200 speedup.

The main GRIP job ended. Verify whether FOCUS has a legal allocation or queue slot; use one owner and at most one training submission. Do not claim the H200 is active/available from a terminal receipt. Reuse an available existing legal allocation if present; otherwise one finite FOCUS PBS submission is authorized by the registered trigger. No repeated qsub or model work on login nodes. Existing separate Corelab inference/physics resources may evaluate immutable snapshots while H200 trains. If unavailable, serialize safely.

Continue direct local model-only snapshot delivery and reuse fixed base/config/processor files. A complete Trainer is for recovery, not the transfer unit for each screen. Do not build a new compression/serving stack. Keep per-file hashes, atomic READY publication and load verification. Never read mutable training weights from the evaluation process.

Use actual user quota, not shared filesystem `df`, for storage admission. Last receipt says928.9GB of1.0TB with a separately calculated conservative available byte bound; refresh it before FOCUS checkpoint writes. Share already verified read-only donor/cache bytes rather than duplicating them. Retain GRIP4096 parent and the latest/previous FOCUS recovery plus any referenced best candidate. Only explicitly disposable qualification payloads and byte-identical redundant copies may be retired after durable inventory/proof; never delete a unique source dataset, sole recovery, unresolved failure record or selected snapshot. If atomic-save headroom is inadequate, report a concrete storage stop; do not fill quota then label the lost state recoverable.

No bulky HF readback, repeated archival scan, periodic full directory hashing or video rendering belongs in the optimizer's per-step path. Durability and finite-value/consumption checks remain, with update/checkpoint/error boundary flushes. Unused mandatory reviews or new permission requests must not leave an otherwise ready allocated H200 idle.

## 8. Qualification and claim boundaries unchanged

TRAIN4=10000..10003; DEV8=30000..30007; FIT16 remains the original registered teacher-window panel. TRAIN>=3/4 and DEV>=6/8 define a local development candidate. Max2 expanded DEV candidates across the entire R2/R3 sprint, first qualifying checkpoint per lineage.

Freeze ONE candidate, complete its local proof, and test untouched CONFIRM16=30032..30047 once. Persistent>=12/16 yields `SOURCE_PICKPLACE_PILOT_CONFIRMED`, not formal32-case qualification or target-WBC deployment. If these conditions were already consumed, adopt the real status rather than rename another set fresh. A failed confirmation cannot be used to retune and retest.30048..30063 remain locked.

Every counted success requires original native on-top/static/not-grasped predicate in all41 control samples28..30s and full32s interface-valid execution. Keep first success and diagnostic grasp/lift separate. Continue the policy after first success; no oracle stop, binary gripper hack, time rescaling, target attraction, gain/IK change, extra planner calls or altered gate. The goal remains stacking, not substituted easier table placement.

FOCUS success on its training episodes is narrow fitting evidence; held-out confirmation tests only new instances of the same source task. Fixed language does not establish language-conditioned task selection. A source-Panda result is not Go2+ARX5, real-time behavior or real-robot safety. `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`; only a separately named sprint_candidate with actual tested scope is permitted.

If FOCUS4096 still fails, preserve the latest locally recoverable model, actual event funnel and unresolved mechanism, then STOP_FOR_NEXT_TARGETED_TASK_DECISION. Do not autonomously add a third lineage, more loss coefficients, new recovery data, a history/phase input or longer updates. These may be proposed from the observed next failure, not silently executed.

## 9. Shared budgets, accounting and morning handoff

No budget increase/reset. PI0277 cumulative caps remain: healthy formal8192; completed updates9048; training heads172032; quality requests14336/heads57344; physical starts234; physics750000; controls150000; RGB pairs155000; teacher controls48000; CPU input attempts65536; WBC/hardware0.

At the reviewed state at most4096 healthy formal updates remain for the sole FOCUS. New FOCUS qualification16 is counted, not concealed as setup. All earlier canary failures, data fetch/prefetch attempts, queries, valid failures and technical retries remain in the shared ledger.

R3's nominal remaining panels, IF GRIP4096 has not yet been evaluated: up to5 snapshot screens (one primary plus4 FOCUS)*4 TRAIN episodes + at most2*8 DEV +16 confirmation =52 learned episodes;52*80+5*16=4240 quality requests;166400 nominal physics steps and33280 controls. These are remaining capacity estimates, not additive entitlements. Reuse already executed work, subtract all prior usage and stop at the first existing cap. No new scientific native fixtures/source episodes are required by R3.

Required handoff: actual current policy/snapshot and recovery paths, frozen contract, inference command and input example; all attempted panel rows and selection trail; stage/installation diagnostic table; actual queue/update/transfer/evaluation timing; local storage inventory; archived-vs-deferred status. Encode at least the first successful and first failed saved-frame examples of the selected panel when present, without selecting only visually pleasing attempts. If no candidate qualifies, label `NOT_USABLE`/`NOT_CONFIRMED` with the achieved partial behavior, never success because the deadline arrived.

Stage driver: adopt_actual -> finish_GRIP4096_screen -> [candidate_dev_confirm OR authorized_FOCUS_canary_train_screens] -> freeze_verify_confirm_once -> local_handoff. Saved-array mechanism analysis runs alongside this bounded route, not as a new long blocking research stage. At07:00 no more training, by08:00 hand off actual results; no automatic later restart of deferred HF/CONTROL/large panels.

## 10. Review scope and publication

Reviewer read live ledger, fixed training audit, GRIP2048 native/installation/delivery records, FIT output index, FOCUS preparation and actual r2 panel/prepare_focus/focus_run/replay_panel code, plus user-quota evidence. Historical supplied notes are context for TRAIN-versus-generalization diagnosis, policy identity, physical object outcomes and target-WBC-independent task data; their old statuses and authorizations do not replace current results.

No reviewer SSH, GPU profiling, new policy/teacher/physics calls or full raw-array/video replay occurred. Source hashes and exact remote counters must be bound by the executing owner. R3 writes authorization documents only; it does not claim the new dispatcher or FOCUS ran. Concurrent valid owner commits must be preserved.

Startup: read this canonical/release, adopt actual final GRIP4096 evaluation and reuse FAST_CACHE2; when the actual registered task-fail trigger holds, execute the sole prepared FOCUS on H200 with local snapshots and native-result-first physical screens. Diagnose grasp/lift/replan/IK installation using existing arrays, not guessed causes. Preserve deadlines, original task/gates, local recovery and shared counters; no HF blocking, no GRIP rerun, no new task/WBC/oracle/budget reset. Freeze and confirm one candidate or honestly hand off the unresolved small-task failure.
# PI0278 — Single-task full-support coverage fine-tuning

Date: 2026-10-07, Asia/Singapore.
Authority: **PI0278_SINGLE_TASK_FULL_SUPPORT_COVERAGE_FINETUNING_GO_V1**.
Reviewed source/evidence SHA: `c944b286c0964b22b3740f7cda91df89baffd241`.
Predecessor terminal: Issue #11 comment **6025593090**, PI0277 R3, release `a5a7e569684d52170f02b0819ae006a7587800dd`.
The Issue #11 canonical publishing this document binds its immutable release and `reports/PI0278/design/REVIEW_AND_EXECUTION_LIMITS.json`. Research charter #5994405991 remains binding.

## 1. Decision, scope and new authorization

**GO for one new COVER93 fine-tuning lineage, up to4096 formal optimizer updates, actual accelerated H200 training, fixed source-physics screens and one conditional untouched confirmation.** This is not another request for a preparation-only report. Reuse the existing StackCube task, original93 successful TRAIN EEF episodes, existing processor and FOCUS4096's final objective. The principal intervention is an explicitly changed training visit schedule: preserve focused16 support, restore the other77 existing task episodes, and sample across each phase's full eligible time range rather than its earliest anchor ranks.

No new task, source scene generation, camera domain randomization, teacher, recovery-data collection, target WBC or hardware execution is authorized. This remains fixed-language source-Panda StackCube. It must not be relabelled easier table placement or general mobile manipulation. The goal is an actually loadable teacher-free EEF+gripper policy completing the original object task, not only lower loss.

PI0277 is closed with NOT_USABLE. Preserve its8192 formal/8232 all completed updates, results and deferred items. PI0278 has a separately explicit additional budget below; no old counter is reset and no old unused allowance is silently revived. The old Oct7 07:00/08:00 sprint cutoff belongs to the sealed predecessor, not this newly authorized goal. Bind PI0278 authority explicitly in the new driver rather than bypassing a stale date guard in the old run. No new clock-time success promise is made. Stop at qualification, the finite new budget, or a concrete validity/resource blocker.

One existing Codex main owner; same repository branch. Do not create a competing autonomous goal, archive monitor, long-lived branch, PR77 dependency, or helper-signature gate. Read actual current jobs and local paths at adoption. A terminal H200 job is not an available active allocation. Use one legal training allocation/submission; retain any legitimate newer work and do not duplicate it. Actual execution must be committed and hash-bound before running; publishing this protocol is not evidence the new runtime or training ran.

## 2. Verified predecessor result and limits of its interpretation

Latest terminal ledger read: 2026-10-07 05:15:33 SGT, #6025593090; final completion audit observed05:13:01 SGT. Later comments were empty at review. Final FOCUS job651442 ended04:09:09 SGT, F/exit0. Reviewer has no SSH/live scheduler observation.

- FOCUS4096 TRAIN4: all4 lifted and reached on-top; all4 had a first native success;3/4 met persistent success. TRAIN10002 first success5.20s, on-top4.73–17.74s, first IK hold27.60s. Thus late IK does not explain the initial loss of on-top by temporal precedence alone.
- DEV8: sustained grasp8/8, sustained lift5/8, on-top5/8, first native success4/8, persistent success3/8. All8 completed32s/640 controls, no final gripper-envelope termination. Successes30001/30004/30005; the latter two first succeeded27.75/25.80s. Do not relabel these slow successes as failures, but record their limited timing margin.
- DEV failure groups:30003/30007 no sustained lift with455/536 IK holds;30006 no sustained lift with0 IK holds (a first2cm crossing is not a sustained lift);30000 on-top observed but no native success;30002 first success4.70s but not persistent. Grasp, release/static, progress and installation remain separate questions.
- The box term was applied only at FOCUS updates1803–4096. The final panels' absence of envelope terminations is real, but training continuation, focused sampling and box introduction were not a matched causal study. Keep the final effective objective; do not run another coefficient search.
- Final teacher-window FIT16 has initial-prefix XYZ distances about29.2–41.9mm, versus about2.1–5.7mm on the twelve close/transfer/release anchors. These are mean per-point distances to the teacher, not physical controller errors or an empirical required task tolerance. Successful alternative trajectories can differ from the teacher. No arbitrary new millimetre gate is introduced.
- Three-of-four versus three-of-eight is a small, adaptively developed comparison. It supports investigating within-task robustness, not a quantified generalization theorem or sole-root-cause verdict.
- CPU replay976 final outputs and12 saved-RGB videos are sealed. Reuse, do not recompute all history as an admission ritual. Actual FAST_CACHE2 production median1.733s/update, loader wait0.000224s; no evidence supports more workers as the next useful optimization. Input-only speedup is not whole-GPU speedup.

## 3. New source finding: phase balance did not guarantee temporal support

Reviewed actual code:
- `scripts/pi0277_execution/r2/focus_plan.py` sorts eligible anchors per(seed,group), then selects `anchors[(cycle//16)%len(anchors)]` for8192 microbatches.
- Each of16 episodes receives128 visits per group. Therefore any group longer than128 anchors only exposes anchor ranks0..127 in that FOCUS schedule. Short groups repeat. A long release/terminal group is not evenly sampled to the end of the32s episode.
- The preceding PI0276 sampler uses the analogous chronological rank rule over93 episodes. Its4096 visits/group give roughly44/45 anchor selections per episode/group; the first8192-microbatch GRIP plan is narrower still.
- `source_worker.py:source_episode` actually records the post-solver hold to640 controls, and `dataset.py` exposes genuine H40 windows through anchor600 for fully eligible episodes. Data present on disk, eligible windows, consumed anchors, and future labels inside a consumed H40 are four different quantities.

This is a concrete coverage limitation, NOT proof that missing late anchors caused every failure. Some late views/states may be almost duplicates of earlier hold observations, and earlier training did contain terminal samples. Never say the model had no terminal data. Exact per-episode temporal gaps, effective distinct observations and actual consumed counts must be joined from immutable plans/journals at A0. The research intervention is to use existing support better before inventing more losses, tasks or a new teacher.

## 4. A0: short saved-data audit and deterministic plan; no blocking research detour

Reuse the final handoff and source manifests. CPU-only:
1. Join actual predecessor consumed UIDs to the original DATASET_CARD and command phases. Output per episode/group eligible count, visit count, unique anchors, first/last anchor, full-H40 label span, eight temporal-stratum occupancy, and observed20–28s/28–30s support. Count exact observation/condition and label hashes; flag near-constant terminal motion using saved RGB differences/TCP/opening, without embeddings or new model calls. Do not equate different floating-point hashes with substantively different states.
2. Extend saved native chronology to DEV8: determine which original on-top/static/not-grasped subpredicate failed and when. Join raw/installed/actual targets, gripper command/measured opening, lift/drop/on-top loss and first IK hold. Test chronological precedence; do not infer causal contact mechanics or stiffness from counts alone. No new learner rollout, teacher or source manipulation is required.
3. Verify existing93 TRAIN admission,16 focus membership10000..10015,77 remaining TRAIN identities, source command/observation hashes and unchanged input whitelist/codec. Source VAL/DEV/TEST and confirmation never enter optimizer inputs or statistics. Current runtime seals take precedence over a stale historical helper inventory; preserve the recorded reconciliation rather than rewriting old history as all-files PASS.
4. Freeze the new schedule below, all labels/masks/proofs, new noise namespace and code fileset. Unit-test rank coverage, cursor resume, boundaries and no overlap with evaluation-only conditions.

A0 is not a prerequisite to re-run every source expert, video or complete historical S4 replay. CPU findings run alongside staging on separate legal resources. A semantic contradiction in labels/frames/timing is a genuine stop; a missing cosmetic report is not. If the actual source schedule differs from the reviewed code, report that before claiming the specific limitation; do not silently correct the record. COVER93's prospective full-support schedule remains the authorized experiment if all input semantics are valid.

## 5. COVER93 data and sampling contract

Reuse immutable original PI0276 TASK data and effective statistics; no relabelling, new demonstration, target-conditioned data filtering or normalization fit.

Pools: F=the16 original FOCUS episodes; B=the other77 admitted TRAIN episodes. Both are original same-task data. The model previously saw some B data; B is not an independent test set.

Freeze8192 microbatch visits /4096 updates before training:
- group=`micro_index % 4` (four existing teacher phase groups, exactly2048 visits each);
- pool=F for even `floor(micro_index/4)`, B for odd; exactly1024 visits per(pool,group),50/50 overall;
- within each(pool,group), cycle episodes in ascending fixed seed order. Each F episode receives64 visits/group; each B episode receives13 or14 visits/group. This is a retention-weighted mixture, not uniform93-episode sampling or new broad scene production;
- per episode/group, sort ALL eligible anchors. Visit0 uses the first; visit1 uses the last. For later visits split the sorted list into `min(8,L)` contiguous, nonempty, near-equal-count temporal strata (`numpy.array_split` semantics), visit strata round-robin, and select within each stratum by a fixed SHA256 ordering of `PI0278|27803|seed|group|uid`, cycling only after that stratum's entries are exhausted. All choices depend only on source membership, not model loss or observed success;
- freeze the exact UID list; record unavoidable repeats and short-group fallbacks. Do not replace failed training/evaluation examples with easier conditions.

At full budget every episode/group must have the first and last eligible anchor and all its nonempty temporal strata represented. This exposes both phase transitions and late support without making the long terminal interval dominate by duration. Model input remains exactly permitted RGB, actual EEF/gripper state and fixed language. Teacher phase, anchor/time, sample group and source seed are sampler/diagnostic metadata only, never policy features. Labels remain actual future commands, not achieved poses or repeated invented setpoints.

Report at512/1024/2048/4096 what has REALLY been consumed, not prefetch reservations. At early termination do not claim full4096 coverage. Where late current observations are duplicates, report that limitation; this plan changes exposure, not the underlying distribution of learner states.

## 6. Fixed policy, objective and fine-tuning

Start from the actual FOCUS4096 effective weights, including the final bound phase. Preferred complete local donor:
`/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2_FOCUS/FOCUS/main/training/checkpoint-4096`
Full Trainer tree: `93e500e8ecb6a719d4a53be09d23b35bdeeb83fcc61e75416247a2c0380f4604`.
Core model-only view: `/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/R2/models/FOCUS4096`; not an optimizer-resume source.

New lineage initialization uses donor WEIGHTS and unchanged processor/statistics/mapping/decode with one fresh optimizer and a new4096-step schedule. Do not reuse a fully decayed old scheduler as if it were continuation. Once COVER93 starts, all segment resumes preserve its optimizer/scheduler/RNG/consumption cursor. These are explicit new fine-tuning steps, not relabelled old steps.

One arm **COVER93_4096**, seed27803,4-step S4, original471 trainable/559 frozen tensor scope, batch1/accumulation2, lr1e-5, cosine4096,warmup10, weight decay1e-5, clip1.0. Preserve actual R1 precision (training generated query/feature and Euler FP32; native epsilon and evaluation query paths as previously bound). No backbone unfreeze, history, phase token, new head, new latent policy, precision change, compile, DDP or fused-optimizer experiment.

Effective objective from the FIRST COVER93 step:
`L_total = L_native +0.4 L_position +0.1 L_increment +0.1 L_rotation +0.1 L_speed +0.1 L_generated_gripper +0.1 L_box`.
Reuse the actual qualified definitions/reductions and source-bound helper functions, not namesake historical code. The generated-gripper term keeps80% mass on indices0..7 and20% on8..39 and scale0.10. `L_box` uses uniform valid H40 weights and `(relu(-g)/.05)^2+(relu(g-1)/.05)^2` for the same differentiably decoded command fraction. No extra S4 traversal, detached supervision or output repair. Original full-H40[-.1,1.1] admission and original actuator clip[0,1] remain unchanged and logged. No new threshold search.

Thus COVER93 changes the training data-consumption recipe relative to the effective parent, not loss formulas, physics or runtime action semantics. Improvement versus the parent includes additional optimization and changed sampling; without a matched control this goal does not identify their separate causal contributions.

## 7. Accelerated, local-first execution

Use qualified FAST_CACHE2,CPU-only ordered workers,pin_memory and existing raw/preprocessed input caches. No new0/2/4/8-worker competition: measured steady-state data wait is negligible. Stream/fill previously unseen cache entries while useful training proceeds; distinguish cold misses from steady-state wait. Reuse deterministic processor outputs only, not neural backbone features or diffusion noise. Never let prefetch advance the consumed cursor. No data-processing or neural work on HPC login nodes.

One short true COVER93 continuous8 versus4-save-fresh-process-resume4 check (16 completed updates total) is authorized. Verify same UID/input/label/mask/actual noise bytes, loss/objective, trainable scope, optimizer/scheduler/RNG and original exact comparison criteria. Reuse qualified loader/S4 components rather than repeating the old384-step performance tournament. First formal steps must show finite gradients and actual updates; preserve the frozen payload comparison at completion. No requirement for an extra percentage speedup before useful training.

Local snapshots at512/1024/2048/4096; complete local recovery at least1024/2048/3072/4096 and safe pause/stop. Atomic immutable READY, correct ownership and checksums are required. Retain parent FOCUS4096 and new latest/previous complete recovery plus any selected snapshot; do not duplicate or delete unique original data. Use user quota, not shared `df`, and reserve atomic-save headroom. CPU caches may be bounded/evicted; do not require materializing all55893 windows before training. No GPU-idle full-bank hashing or repeated public archive scans.

Evaluation on existing separate Corelab resources may overlap H200 training using immutable snapshots. If resources are shared, serialize at safe update boundaries. Maintain at most one pending snapshot delivery; a not-yet-run superseded milestone may be explicitly skipped for backlog, never a valid failure or in-flight episode. The final4096 screen cannot be silently omitted if it is needed to determine the outcome.

HF bulk/readback and the old CONTROL remain DEFERRED_USER_PRIORITY. Do not re-download existing donor weights. Direct model-only transfer is the fallback; no optimizer transfer for every screen. A lossless replacement-tensor overlay over the already-local FOCUS4096 base is explicitly allowed as a transport-only optimization: enumerate every changed state-dict key (including buffers), preserve exact dtype/bytes, hash-reconstruct the FULL effective state against the source snapshot before admission. This is replacement tensors, NOT quantized deltas or floating-point difference/addition. Never omit changed nontrainable buffers. Verify this CPU path once; if not readily correct, use ordinary direct snapshot transfer rather than building a new serving system. Optimizer recovery remains a full local Trainer.

Record update, data wait, cache misses, snapshot save, transfer/readback, model load, physics screen and decision latency separately. No fabricated whole-GPU speedup claim. Actual neural quality uses fresh predictions, not the delivered cached-output example.

## 8. Fixed diagnostics, screens, confirmation and stop

Baseline physical results for FOCUS4096 on TRAIN4/DEV8 already exist and are reused with hashes; no rerun just to create a new table. The oldDEV8 is development, not blind TEST.

Before any COVER93 query freeze a48-row source TRAIN diagnostic panel: seeds10000..10003 plus the first four sorted B seeds; six unique anchors per episode selected without model results: first eligible group0, first eligible group1, first eligible group2, first eligible group3, closest eligible group3 anchor to400 (20s), and closest to580 (29s), ties earlier. If a requested anchor collides or does not exist, use the nearest unused eligible anchor in that group, with explicit provenance; never drop a difficult episode. The source card's >=2 per group and full task horizon must suffice; otherwise state a concrete panel-spec contradiction before neural calls. Freeze one actual-noise packet per UID, reused by parent and new snapshots; use existing identical saved outputs when available, otherwise allow one parent query per row. Report per-phase position/rotation/gripper errors, full-H40 range violations and actual source exposure. This panel is training diagnostics, not independent test data.

Each selected checkpoint milestone512/1024/2048/4096 runs that48-row panel plus original TRAIN4 seeds10000..10003, full32s/640 controls, original initial state and original matched episode noise. Maximum4 screens. No teacher/object truth/phase/success signal enters policy or stopping logic. Refusal/IK-hold/technical-incomplete distinctions are kept; valid failures cannot be retried.

DEV8 seeds30000..30007: at most TWO candidates. The first candidate is the first TRAIN-qualified milestone at or after1024 (>=3/4 persistent); if its DEV is below6/8, the second and final candidate is4096, provided its TRAIN is qualified and it has not already been evaluated onDEV. Do not repeatedly inspect DEV at every step. A candidate reaching TRAIN>=3/4 andDEV>=6/8 is frozen immediately; training may stop safely at its next optimizer boundary, retaining any later in-progress weights as nonselected. No further tuning is allowed after confirmation opens.

For the ONE frozen development candidate, finish its own input/noise/S4/decode/native checks and local reload identity, then execute untouched CONFIRM16 seeds30032..30047 once with frozen noise. Their untouched status must be verified against actual predecessor/current ledgers before creating them. If previously accessed concurrently, do not silently relabel another set fresh; report the conflict.30048..30063 remain locked. No new source expert run is required, and no failed condition leaves the denominator. A native-reset engineering impossibility is a concrete validity stop, not permission for favourable seed substitution.

Success is the unchanged native on-top/static/not-grasped predicate at all41 control samples28..30s plus full32s interface-valid execution. Report first success, sustained grasp/lift, on-top, release, subsequent loss, IK and raw/installed/achieved errors separately. Do not stop at first success or loosen the task to pass. Count original actuator saturation honestly.

Confirmation>=12/16 yields `SOURCE_PICKPLACE_PILOT_CONFIRMED` with this limited source-task scope; it is not32-case qualification, real-time control, language generalization or WBC/hardware deployment. `selected_checkpoint=NONE` and `deployment=NONE`; a separately named locally confirmed sprint_candidate is permitted. If the gate is not met, freeze the result as NOT_USABLE/NOT_CONFIRMED. A confirmation failure cannot trigger another candidate or more training on those conditions.

At4096, if no development-qualified candidate exists, STOP with the actual phase-specific result. Do not autonomously add loss coefficients, another4096 steps, a third optimizer branch, state-history inputs or recovery demonstrations. Propose a narrow next decision from the failure: source-window fitting still poor versus learner-state gap versus post-release disturbance versus installation. In particular, do not mistake source clock reweighting for a demonstrated learner-state recovery solution.

## 9. New finite budget and ledger

PI0278 planned:4096 formal +16 qualification=4112 completed updates. Hard caps:4096 healthy formal;4176 all completed (64 technical-recovery allowance, not extra healthy training);76032 training heads. All failed/partial optimizer attempts and actually consumed microbatches are recorded separately. At fixed math18 training head evaluations/update, planned4112 gives74016; actual head accounting governs if any extra calls occur.

Planned maximum learned physics:4 milestones*4 TRAIN +2 candidates*8DEV +16CONFIRM=48 episodes. Queries:48*80 +5*48 diagnostic queries (parent+4 new)=4080, less exact reused outputs/skipped unrun milestones. Hard quality requests4608/heads18432. Any new engineering neural query counts within this cap; saved-array CPU replay does not count as new inference.

Physics planned153600 steps;30720 controls;<=30768 RGB pairs at641 observations/episode. Hard caps52 starts,170000 physics steps,34000 controls,34500 RGB pairs,65536 CPU processed-input attempts including prefetch. Up to4 identical-condition engineering episode replacements within these caps; valid failures never replaced. Native reset steps and partial attempts count. Teacher/model-training data collection0; WBC/hardware0. One H200 training allocation plus existing separate evaluation resource when available; no extra GPU purchases or cloud provisioning.

New and predecessor ledgers remain separate and jointly indexed. The new4096 authorization raises the combined PI0277+PI0278 healthy formal allowance to12288, not an erased8192. Dataset/source labels are reused, not recounted as newly collected tasks. No bulk HF uploads or readbacks are required to advance or finish this goal.

## 10. Required delivery and actual implementation order

`adopt_terminal_and_local_sources -> CPU_exposure_and_DEV_event_audit -> freeze_COVER93_schedule_and_runtime -> short_exact_resume -> formal_train_with_local_screens -> first_qualifying_DEV_or_final_DEV -> freeze_verify_confirm_once -> local_handoff_and_GitHub_return`.

Deliver: per-failure evidence table; old/new actual temporal/episode exposure and distinct-condition summary; exact initial/effective policy identity; resumed Trainer and model-only/overlay distinction; consumed plan/noise/counter records; all attempted TRAIN/DEV/confirmation rows; timing and quota facts; compact saved-frame successful/failed examples when present; real inference startup command and input example (clearly distinguish cached replay from new inference). Candidate-local integrity and numeric task proof gate claims; video montage/public archive/long retrospective do not gate useful training or a native decision. No silent deletion of failure evidence.

Before any new run, commit implementation under `scripts/pi0278_execution/` and its source bindings under `reports/PI0278/`, preserving old frozen files. Use configuration for goal/paths/update limits rather than broad text substitutions over historical scripts. Publication of this GO only changes authorization; actual code and jobs require owner receipts. Ordinary same-semantics engineering fixes can proceed inside this finite scope without repeated scientific permission requests.

## 11. Review source index and limitations

All below were actually read at the reviewed SHA:
- `reports/PI0277/execution/R3/FINAL_SCIENTIFIC_VERDICT_AND_HANDOFF.md`;
- `FINAL_COMPLETION_AUDIT.json`, `FINAL_HANDOFF_INDEX.json` (relevant recovery portion), `PANEL_FOCUS4096_TEST.json`, `FOCUS4096_NATIVE_CHRONOLOGY.json` (relevant event portion), `PREFIX_FOCUS4096_SUMMARY.json`;
- `scripts/pi0277_execution/r2/focus_plan.py`, `scripts/pi0277_execution/audit_exposure.py`, `scripts/pi0277_execution/r3/bound_train.py`;
- `scripts/pi0276_execution/dataset.py`, `scripts/pi0276_execution/source_worker.py` (including source/EEF bridge and controls).

Source and cumulative caps from the actual terminal ledger were checked; arithmetic of the12 final outcome rows and the128-visit sampler rule was independently recomputed. Reviewer did not SSH, run training/physics, access remote full raw arrays, or watch all12 videos. Exact consumed coverage and DEV causal mechanisms remain A0 execution checks; source-derived facts, mathematical implications and prospective design choices are not interchangeable.

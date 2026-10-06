# PI0277 R2 — Overnight local-first pick/place policy sprint

Date: 2026-10-06, Asia/Singapore.
Authority: **PI0277_R2_OVERNIGHT_LOCAL_FIRST_POLICY_SPRINT_GO**.
Reviewed HEAD: `710927f3aa3b60512ec076e8e280eedfdc092f70`.
Parent authority: PI0277 V1, release `63151a4548243ab0fb4ba172eea6f7d92e02c576`, Issue #11 comment #6014658161.
Research charter #5994405991 remains binding. This amendment takes precedence over conflicting V1 sequencing, selection, timing and archival requirements from its publication onward. It does not rewrite previously executed experiments.

## 1. User priority and deliverable

The user explicitly prioritizes a usable learned pick/place policy by tomorrow morning over HF transfer/readback and complete causal ablations. Execute a bounded engineering feasibility sprint, not another preparation-only task. The planning cutoff is **2026-10-07 08:00 SGT** (an operational interpretation of “tomorrow morning,” not a guarantee of success). Start no new training segment after 07:00; use the remaining window for local candidate verification and an honest handoff. Earlier resource or numerical stops still apply. Do not automatically begin a new run after the cutoff.

The existing task remains ManiSkill StackCube: grasp red cube, lift/carry, place on green cube, release, and preserve the native task predicate. This is the present concrete pick/place target. Do not silently replace stacking with touching a target, flat-table placement, a different environment, or a relaxed predicate. This goal seeks a **source-Panda physical-simulation prototype**, not real-time control, Go2-WBC qualification, or hardware deployment.

Morning deliverable: one locally loadable frozen candidate package, exact scope and per-case physical scores, successful and failed saved-RGB examples, a runnable inference command, and remaining failures. If no candidate passes, deliver the best development result explicitly as NOT_USABLE; no deadline-based success claim. Full paper-quality ablations/generalization and public archive closure are deferred, not claimed completed.

## 2. New evidence: reuse A0, do not restart it

Issue #11 #6015145955 (2026-10-06 19:20:08 SGT), and `reports/PI0277/execution/A0_RETURN.json`, establish an executed saved-array audit of all48 TASK8192 physical cases:13 sustained native grasps, **zero sustained2cm lifts and zero on-top occurrences**. REPAIR_DEV has24 no-grasp and8 grasp/no-lift; TRAIN has10 no-grasp,5 grasp/no-lift and1 original gate refusal. The2cm/0.3s lift measure is a diagnostic, not a substituted success predicate.

Actual command-to-gripper transport was checked; no semantic contradiction was reported. The original16,384 training visits and93 command timelines were audited. Native loss uses400 active full-H40 elements, not132-channel averaging. Generated-gripper error was detached logging, while native supervision exists. The new differentiable gripper loss passed CPU checks. Real H200 cache/timing/canary and formal PI0277 training were NOT_RUN in this last saved receipt; the owner must adopt any newer actual work instead of assuming it is still idle.

This updates the working diagnosis from generic late drift to failure at grasp-to-lift/early transport. It does not establish a unique cause. Reuse these completed audits and source-positive31/32 EEF results; do not repeat a full audit or regenerate the dataset before training.

## 3. What changes immediately

| V1 requirement | R2 overnight instruction |
|---|---|
| CONTROL4096 before GRIP4096 | **GRIP first. Unstarted CONTROL deferred.** |
| Two96-step timing trials per route, possible extra192 steps | Compact correctness/resume qualification; production timing is observational. Do not spend384–576 discarded updates proving a speedup. |
| Complete2048/4096 Trainer HF readback before final evaluation | Local recoverable state and immutable inference snapshots suffice. HF is NOT a dependency. |
| Evaluate only4096 | Evaluate up to four declared milestones per lineage; adaptive development selection is authorized and disclosed. |
| Full offline384/fit384 and16+32 episodes per arm before confirmation | Small fixed task probes throughout training, then one frozen-candidate confirmation. |
| Unique causal gripper-loss result | Deferred without matched CONTROL. Improvement is engineering progress, not a proven isolated loss effect. |
| Full source/prediction/summary tar readback in the current critical path | **Suspend/defer bulk HF work until after the sprint.** Preserve local originals and honest pending status. |

## 4. Ownership, active jobs and storage

Keep one existing Codex owner and the existing branch. Implement this amendment in a new R2 runtime seal/output namespace, reusing validated V1 math and completed work. No competing autonomous goal, repeated ownership approval, broad migration or new long-lived branch.

At adoption record actual GPU/PBS/process state once, cumulative counters, local donor/data locations, available disk and active HF children. If GRIP is already training correctly, continue from its actual optimizer/RNG/scheduler state; do not restart from zero for this amendment. If CONTROL or a long discarded benchmark has begun, the user now permits graceful suspension at the next recoverable optimizer boundary to prioritize GRIP; preserve its partial result as PRIORITY_DEFERRED, never a task FAIL. Do not kill a process in the middle of checkpoint serialization or overwrite its logs.

For ongoing PI0276/277 bulk HF upload/readback, the owner is explicitly allowed to stop scheduling new files, finish an already atomic file operation when inexpensive, or gracefully stop the transfer process after preserving progress and local sources. This includes the old source/prediction readbacks previously protected as mandatory continuation. Do not abandon an unfinished multipart result as PASS, delete remote assets, or delete the sole local source. Record `ARCHIVE_DEFERRED_USER_PRIORITY` and verified/unverified byte/file ranges. Do not launch a replacement archive watchdog. Existing completed readback receipts remain valid.

**Keep local recovery.** Save an atomic full Trainer on reliable existing scratch/local storage at least every1024 completed updates and before a branch change, job suspension or final stop. Retain latest and previous recoverable states plus the chosen candidate; do not delete parent/donor/data or a referenced best snapshot. Check capacity before writes; saving to a node-local directory is not durable across node loss unless copied to the existing persistent filesystem. No public HF save/readback is required overnight.

Use light immutable model-only snapshots at evaluation milestones. Evaluation needs weights plus the full policy contract, not optimizer state. Reuse the fixed frozen model components already present on the evaluation host. If a lossless trainable-tensor delta loader is already easy to support, it may replace repeated whole-model transfers only with an exact base-file identity, complete changed-tensor/buffer inventory, original stored dtypes, and reconstructed-state verification; otherwise use a normal weights-only snapshot. Do not build a new compression system tonight. Transfer only what the next evaluation actually needs, directly between existing authorized hosts/shared storage; no HF round-trip and no repeated download of the same donor. Hash transferred files once and load-check them. A weights-only snapshot must never be advertised as resumable Trainer state.

## 5. H200 use and compact fastpath qualification

Use the existing authorized H200 allocation for training (one GPU,12CPU,225GB host RAM when available). Use the existing separate Corelab GPU/CPU source runtime for physical evaluation so training can continue while an immutable snapshot is evaluated. Do not create resource contention on one GPU. If a second evaluation GPU is unavailable, pause at a completed segment boundary and run the compact test, then restore training without resetting optimizer/scheduler/RNG. Reuse persistent inference processes where possible. No extra model is allowed to read mutable training weights.

Prioritize verified deterministic CPU preprocessing cache, pin_memory, ordered0/2-worker loading, incremental logs and existing data locality. Materialize only needed cache entries first; a complete full-bank preprocessing pass is not a prerequisite for the first update. Every used cache entry must match original condition/label/mask hashes. Missing entries can be generated on demand. Workers use spawn, independent loader RNG and an actual-consumption cursor; prefetched work never advances a saved training cursor. Workers must not import the running model, initialize CUDA or install simulator hooks. Keep cache<=48GiB and process-tree RAM<=180GiB within allocation.

Run at most one short CPU comparison per REF_W0/FAST_CACHE0/FAST_CACHE2 using64 warmup+256 fixed TRAIN visits. Select a correct available path, not necessarily two workers. Existing qualified measurements may be reused. A 10% throughput threshold is no longer a launch gate. A0/CPU analysis or a missing performance chart must not occupy the entire evening. If acceleration-specific setup stalls for45 minutes without a working candidate, proceed with the already-valid reference loader and continue cache work only without interrupting useful training. Report SPEEDUP_NOT_ESTABLISHED when appropriate.

For a new fastpath, compact actual-model qualification uses the GRIP objective: REF continuous8, FAST continuous8, FAST4-save-fresh-process-resume4 (**24 completed updates**, discarded). Compare actual inputs/noise, generated output/loss/gradients, permitted weights, optimizer, scheduler and RNG under the existing exact standard. Reuse already equivalent tests instead of repeating them. If using an unchanged qualified loader, continuous8 versus split4+4 is sufficient (16 updates). New generated-gripper gradient reach through S4 is mandatory once. A selected optional FOCUS lineage needs its own short continuous/split check if sampler integration changes. Total newly initiated qualification updates under R2 <=64; all inherited/R2 calls still share cumulative limits.

If FAST fails input or numerical equivalence, use a validated reference path rather than training with known corruption. If even reference GRIP validation fails, stop with that concrete blocker. Do not loosen precision or correctness to pass. Record production completed-update timing, data wait, snapshot cost and evaluation walltime where measurable; no invented utilization or speed ratio. No costly extra profiling runs. Common detached metric batching/less frequent diagnostic flush is allowed when values and event order are preserved; keep finite checks, consumption/RNG identities and durable update/checkpoint/error boundaries. Detailed tensors may be sampled at declared checkpoints instead of synchronously dumped every forward; record instrumentation differences, not a false exact audit claim.

Batch1/accum2, stored weight dtypes, R1 training-FP32 generated query/evaluation-BF16 query, S4 four steps, AdamW, LR schedule, preprocessing and action decoding remain fixed. No TF32/FP8, torch.compile, DDP, new model head, backbone unfreeze or optimizer replacement tonight. The acceleration claim is shorter time to a measured physical result, not a required artificial worker-count win.

## 6. Main learner and bounded adaptation authority

### Primary lineage: GRIP

Start or continue the V1 GRIP lineage from TASK8192 weights, with the V1 fresh optimizer and4096-step cosine context unless already legitimately running. Preserve the original first8192-microbatch schedule,93 TRAIN episodes, fixed task statistics, mixed parameter scope and actual noise identities. Objective stays exactly `L_old + 0.1*L_gripper_generated` from V1: final generated command fraction,0.10 reference scale,80% mass on executed indices0..7 and20% on8..39, no detach/clipping/rounding; original pose losses unchanged.

Export/evaluate at cumulative primary updates **512,1024,2048,4096**, skipping milestones already passed at adoption with a recorded reason rather than rewinding. An intermediate checkpoint is now eligible for development selection. Keep training while a separate host evaluates; allow at most one pending evaluation snapshot to prevent an unbounded transfer backlog. Skipped-not-run milestones remain explicitly unmeasured.

After the first candidate passes the development screen in section7, prioritize freezing/confirmation over further optimization. Do not insist on reaching4096 to call it useful. Save a complete local recovery at the stop boundary. Do not continue modifying the selected model during confirmation.

### Optional single FOCUS lineage, not a sweep

A0 found no sustained lift and TRAIN closed-loop failures, so the owner is now allowed ONE focused task-fitting adaptation without another routine approval. Trigger: the primary has reached2048 and its1024/2048 screens show no improvement in the tuple (persistent stacks,on-top cases,sustained lifts); or it reaches4096 without passing the development screen. No trigger is evaluated using fresh confirmation results. If genuine progress is increasing, finish the primary before branching.

FOCUS inherits the latest legitimate primary weights and uses a fresh optimizer, same LR/precision/objective/parameter scope, up to4096 updates. Its only new training change is sampling from the **original16 lowest-admitted TRAIN diagnostic episodes**, with the same four phase groups balanced equally, deterministic episode-then-anchor cycling. All labels/images remain existing genuine H40 source commands/observations; no new teacher, no environment rollouts in the training set, no oracle phase input. Freeze eligible UID membership, a full8192-visit stream and a new branch RNG namespace before starting. This is a deliberately narrow memorization/feasibility adaptation; success on those16 episodes alone is not generalization. It is not the old matched CONTROL/GRIP causal experiment.

Evaluate FOCUS at512/1024/2048/4096 with the same small panels, at most four snapshots. No additional coefficient/LR/seed/initializer searches or third training lineage. No autonomous new task, DAgger, data generation, output thresholding, teacher takeover, goal attachment or controller change. A semantic bug affecting labels/state/time/decode still requires a specific amendment; do not silently change it under an engineering label.

All PI0277 formal updates, including any already-started CONTROL and all R2 lineages, share **8192 healthy formal updates total**; the existing9048 cumulative completed-update hard cap still applies. Budget is not restarted or automatically replenished. Set the actual remaining allocation in the adoption receipt. Stop at the first of budget exhaustion, numerical/resource blocker, usable candidate or the morning cutoff.

## 7. Fast feedback, candidate definition and honest selection

Before any new quality result, fix:
- TRAIN4: the first4 original admitted TRAIN diagnostic seeds (10000..10003 in the current frozen index; use the original identities, not a score-selected substitute).
- DEV8: original REPAIR_DEV seeds30000..30007. These are development conditions, never blind test.
- FIT16: four teacher-side groups for each TRAIN4 episode, chosen from the already-frozen PI0277 FIT_CARD. Same recorded input/noise identities across candidate comparisons.
- CONFIRM16: seeds30032..30047, from V1's reserved same-task confirmation namespace. Verify no previous access. If already used, mark the collision explicitly and do not pretend they are fresh or invent favorable replacements.

At each of at most8 candidate snapshots, run FIT16 once and **4 full32s TRAIN4 physical episodes**, with the original100Hz physics/20Hz controls and0.4s replan cadence. Fixed noise packet by condition/query is independent of candidate selection. Report first grasp, longest grasp, sustained lift, on-top, release, first/persistent native task success, gate/IK/transport and raw-versus-actual commands. Loss alone cannot advance a candidate. No success-triggered stopping/freeze or replacing the learned controller with the teacher.

A candidate with>=3/4 persistent TRAIN successes may enter DEV8; at most TWO candidates receive this expanded development test. Freeze the first qualifying candidate per lineage, rather than evaluate arbitrary extra checkpoints until one wins. Require>=6/8 persistent DEV successes and no actual interface/numerical violation for a `LOCAL_DEV_PICKPLACE_PROTOTYPE`. Valid gate-refusal episodes remain failures in their denominators, never removed. If multiple qualify, choose by persistent DEV success, then persistent TRAIN success, then lower action error on the fixed FIT16, then earlier checkpoint; no hidden video selection.

Freeze ONE final candidate before CONFIRM16. Run all16 learner episodes exactly once. It becomes `SOURCE_PICKPLACE_PILOT_CONFIRMED` only at>=12/16 persistent native task successes, with full32s interface-valid execution for each counted success, and reproducible local loading/complete policy identity. This is a small engineering pilot, **not** the original24/32 qualification, not proof of a causal loss effect, not multi-task/cross-body or hardware success. Preserve old0/32 and0/16 results and all failed sprint trials. One favorable video is not a usable-policy certificate. If the candidate fails confirmation, mark NOT_CONFIRMED; do not adapt and re-test that set overnight. Remaining V1 confirmation seeds30048..30063 remain unopened.

Native StackCube score stays unchanged: red on green, static, not grasped, all41 control samples28..30s for persistence. Also retain first-success and task-event traces. The learner always receives only legal RGB/world-TCP/measured-opening/language, not object truth, success, phase or contact-derived hidden state. No shortened horizon, new tolerance, altered gripper saturation, gate relaxation, time-rescaling or human intervention disguised as policy success.

Original expert31/32 and codec/replay checks are reused; new expert evaluation is not a prerequisite for the learner confirmation. At most the16 confirmation source joint/EEF pairs may be executed afterward for diagnosis if time remains, never to filter learner scores or create new training data. Full R1 pairedCONTROL evaluation, offline panels, formal32-case confirmation and all archival audits are explicitly deferred, not auto-resumed by the morning driver.

## 8. I/O, evaluation and deadlines are part of the execution plan

The first meaningful target is an actual learned grasp-and-lift trajectory, followed by a complete native task episode. Do not spend the night only qualifying loaders, writing reports or copying checkpoints. No artificial sleeps or repeated role confirmations. Launch the main learner after compact correctness/resume and local-data admission; reuse completed A0/codec/assets checks.

At each optimizer milestone write a small progress receipt (updates, loss/gripper summary, local snapshot identity, latest completed physical panel, remaining budget). At each completed panel append the per-case table immediately. Summary/report generation must not hold the training allocation. At07:00 SGT finish current atomic update/save and start no new training segment; complete already admitted evaluations within remaining resources, then write the08:00 handoff with whatever is actually complete. The morning goal is not a promise that learning must succeed. If adopted after the cutoff, report actual status and do not silently restart another overnight campaign.

A successful handoff contains: local model/view path or lossless reconstructible package; processor/statistics/mapping/decode and base identity; exact inference command and input example; source-task and runtime limitations; candidate selection trail; all development/confirmation scores; at least one successful and one failed saved-frame example when available (if every measured case passes, say there is no failure in that panel). Use saved frames; no rerender for presentation. Publish small GitHub receipts, not large binary artifacts. Leave an archive backlog manifest, with HF still DEFERRED. No background-monitor or scheduled service is created by this protocol.

## 9. Limits, permissions and deferred claims

Keep the PI0277 cumulative caps: formal8192; all completed updates9048; training heads172032; quality requests14336/heads57344; physical starts234;100Hz physics750000;20Hz controls150000;RGB pairs155000;teacher controls48000;CPU input items65536. All work already performed under V1 is deducted. CPU cache/parity/formal reads and discarded/prefetched entries remain counted. R2 planned extra work at zero adoption usage is bounded by8*4 TRAIN probes +2*8 DEV +16 confirm = **64 learned episodes**, at most5120 rollout queries plus128 FIT queries =5248; optional32 source-confirmation episodes and2 short native fixtures keep within existing physical caps. Add only actually needed same-condition technical recovery inside those caps, never quality reruns.

R2 newly initiated qualification is<=64 updates. Primarily one H200 and the existing separate evaluation resources; no new distributed recipe or multi-platform installs. If only one device is available, serialize at safe segment boundaries, not simultaneous contention. No new TRAIN teacher data, no multisession collection, no WBC or hardware calls.

The owner may proceed autonomously through the explicitly listed branches, pause bulk archives, defer unstartedCONTROL, select intermediate checkpoints on DEV, transfer lightweight local inference snapshots and use the bounded FOCUS schedule. No approval is needed for those authorized steps. Everything outside them retains its previous scientific boundary.

Formal `selected_checkpoint=NONE` and `deployment_selected_checkpoint=NONE` remain. A separate `sprint_candidate` may record the actual local source-simulation prototype and measured scope; it never authorizes sending commands to Go2. Full model identity and local recovery are non-negotiable even when public archival is deferred.

## 10. Sources and publication scope

Reviewed latest branch and Issue #11; actual A0 report at710927f3 and comment6015145955, original PI0277 full protocol and PI0276 physical failure evidence, current PI0277 file inventory. A0 establishes no observed semantic contradiction, not proof all possible learning/integration bugs are excluded. No new reviewer SSH, training, simulation or speed measurement was performed. Static preparation/source reads are not runtime execution. Official PyTorch data/tuning documentation supports the general cache/prefetch/synchronization mechanisms only; the installed project versions remain frozen.

Historical supplied notes remain useful for policy identity (weights+processor+statistics+mapping+decode), separating TRAIN fitting from generalization, and object-level outcomes. Their old statuses, mandatory archive/selection workflows or task-WBC prerequisites do not override this explicit user-priority amendment.

Startup: adopt this R2 canonical/release and actual PI0277 state; defer bulk HF and unstartedCONTROL, reuse A0, qualify the cache/loader briefly, train GRIP first on H200 with early local physical probes; use only the bounded FOCUS fallback if triggered; freeze and confirm one candidate, keep local recovery and all failures, no oracle/WBC/new task/budget reset, hand off actual results by the planning cutoff.

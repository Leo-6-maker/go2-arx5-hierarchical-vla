# PI0285 — First manipulation policy, then one shared two-task policy

Authority: **PI0285_FIRST_POLICY_THEN_SHARED_TWO_TASK_GO_V1**.
Review base: `1e9f223510f4baa6e4bc7ac2282429a9530f3c14`, existing branch `pi0267/executor-decoupled-recovery`.
Predecessor: PI0284 terminal #6057009858, 2026-10-08 17:36:17 SGT; evidence `1d0e16bd8dee9595b55db1685c388947bab95830`.
This protocol and `reports/PI0285/design/PLAN_AND_LIMITS.json` are activated together by a fixed-release Issue11 canonical and readback. Publication is not remote execution.

## 1. Decision and inherited boundaries

User requests an explicit first useful small manipulation policy, followed by a shared two-task policy using the same training mechanism, rather than unlimited StackCube optimization. This is NEW bounded source-task/data/training authorization. PI0284 is complete: both final endpoints have TRAIN0/4 and DEV0/8; more Euler steps did not help under its registered comparison. Historical successful experts, earlier partial task successes and all negative results remain unchanged. Neither insufficient precision nor insufficient data is established as the sole cause.

The new milestone is NOT a relaxed score for historical StackCube. T1 is a separately named, wide-target pick-and-place task. T2 restores stacking in the same new visible scene. Do not claim new-scene scores are the old StackCube benchmark. A qualified T1 model is saved permanently before any T2 training. Failure of the shared successor does not delete that first deliverable.

All target WBC weights/configuration/normalizers, target physics, Go2 queries and hardware remain frozen/unexecuted: new calls0. No target robot teleoperation or filtering task data by target-WBC success. Source Panda/native PD simulation is explicitly NON_REALTIME and does not establish target transfer. Old fine-tuning drivers remain closed/paused. One owner adopts actual processes; no competing goal, duplicate qsub, healthy-job cancellation or subagent-signature requirement.

The earlier09:00 deadline was explicitly removed in the recorded PI0283 CLOCK_RESOLUTION. This goal does not restore that expired deadline and does not authorize unlimited work. Finite gates and budgets below control execution.

## 2. Exactly two tasks in one visible scene

Create an isolated project task module, proposed registry name `PI0285PickPlaceStack-v1` (a new implementation, NOT an already verified built-in environment). Reuse pinned ManiSkill/Panda wrist-camera/native PD assets and integration, rather than changing simulator/robot. Two40mm cubes, original table, and one blue16cm x16cm flat visual target zone are present for BOTH task instructions. The blue zone is a table marking, not a physical bin or invisible attractor; do not call it container insertion. No rim/collision asset is needed. Gravity, masses, friction, servo gains, source control20Hz, integration100Hz and original safety rules stay fixed.

T1 instruction: `Pick up the red cube, place it inside the blue square, release it, and move the gripper away.`
T2 instruction: `Stack the red cube on the green cube, release it, and move the gripper away.`

Both target sites are always rendered. Same layout/reset seed yields the same initial robot/objects/cameras and RGB/state for both instructions. Task identifier and evaluator branch never enter policy observations; only the natural-language instruction changes. Do not correlate a task with a different camera, background, reset posture, hidden state or embodiment tag.

Use a bounded40cm x40cm task-plane workspace centered on the existing visible tabletop workspace; freeze its transform in SCENE_V1.json before teacher scores. Generate positions deterministically from layout ID using PCG64, with red/green yaw in[-pi/4,pi/4]. Keep cube footprints and zone inside the workspace, red initial center at least15cm from zone center and12cm from green center, green footprint outside the zone with at least2cm clearance, and no initial overlap. Geometry-only rejection sampling is allowed before scores, maximum1000 proposals per fixed ID; inability to meet constraints is reported, not replaced with a different seed. No score-driven placement, asset enlargement or later threshold edits.

### T1 success, fixed before any learned-policy rollout

Run the complete32s task; do not stop inference at first success. Require (a) a genuine native grasp episode sustained at least0.30s, (b) red center lifted at least2cm above its own initial center for at least0.30s, and (c) horizontal red-center displacement from initial at least10cm. Durations use actual timestamps, not a loosely named sample count. These historical requirements reject a push-only shortcut.

During every one of the41 control samples at28..30s require: all red-cube xy corners lie inside the16cm square with1cm interior margin; red bottom is within5mm of table surface with original penetration checks retained; red linear speed<=0.01m/s and angular speed<=0.5rad/s; native not-grasped; TCP-to-red-center distance>=8cm. Also green final displacement<=1cm. This is object placement and physical disengagement, not merely TCP proximity or an open-command flag. Score each subpredicate and its first/lost events independently.

### T2 success

Use the pinned original StackCube on_top/static/not_grasped formulas unchanged, plus the same physical grasp/lift history and TCP separation>=8cm for this explicitly new shared-scene task. Hold at all41 control samples28..30s within full32s. The original benchmark and its archived scores remain unmodified; the new marker/layout and disengagement requirement prevent claiming direct benchmark equivalence. Report both the original object conjunction and the new full T2 conjunction.

CPU scorer fixtures must include genuine success, push without grasp, held object, wrong destination, one-frame success, target loss, and gripper remaining close. No oracle success, hidden object pose, teacher phase or fixed timer may stop/switch/freeze the evaluated model.

## 3. G0: reuse infrastructure; repair only real multi-task ingestion requirements

Inspect current manifests and actual imported __file__/hashes. Reuse source controller, recorder, native model loader, exact-resume machinery, source evaluation and local model transfer. Do not repeat previous32/32 native parity or download all weights. New-scene/new-language checks are genuinely new and must not be replaced by old PASS receipts.

The reviewed inherited codec.py hard-codes INSTRUCTION; cpu_inputs.collate assumes equal token lengths, and its cache key omits language because it was single-task. The isolated new loader must accept each episode's instruction, use the pinned original GR00T collator/tokenizer for variable-length padding/attention masks, and include task/instruction/processor/source identity in cache keys. CPU-test mixed T1/T2 batches, different text lengths, batch-size1 versus mixed-batch masks, and same RGB/state with different tokens. Do not merely concatenate preencoded unequal text lengths. The actual runtime must show which new functions it imports.

Policy inputs remain current real128x128 body/wrist RGB (native256 preprocessing), actual EEF pose and measured gripper opening, and episode language. Raw histories/timestamps may be recorded, but no new history, object pose, joints, phase, teacher ID, task ID or success flag is added as a model input in this goal. No new crop, augmentation, state dropout or precision hypothesis.

Output contract remains H40x10: canonical world/task-frame XYZ+row-rot6d and independent gripper command. Network encoding uses native relative EEF, absolute gripper, active10/132 masks, original native inverse and K4 generation. Control uses the first8 references then new feedback. Commanded opening is not measured opening. Source joints may be used inside a teacher planner/FK, never as GR00T supervision or target-robot inputs.

Use clean parameterized modules, not another chain of untested string replacements of legacy runtime scripts. Same-semantics bugs may be fixed at dead-process boundaries with versioned receipts; task, input, label, optimizer or scoring changes outside this protocol require a new decision.

## 4. G1: task-teacher pilot, then a small paired episode bank

Fixed layout families (check absence from all retained ledgers before any score; a collision blocks rather than silently selecting another range):
- PILOT:285000..285007,8 layouts, both tasks.
- TRAIN:285100..285163,64 layouts, both tasks.
- DEV:285200..285215,16 layouts, both tasks.
- CONFIRM_A:285300..285331,32 entirely held-out layouts, T1 only.
- CONFIRM_B:285400..285431,32 new layouts, both tasks from identical reset states.
Split by full layout family before H40 slicing. Pilot is development only, not supervised TRAIN. No confirmation rollouts/rendering/teacher/model queries before the appropriate frozen-candidate gate.

Use a single geometry/state-aware offline teacher framework: approach, align, close, verify actual grasp, lift, carry, lower, release, withdraw, hold; at most two within-episode recovery attempts inside32s. It must reason from actual physical state, not blindly advance on a global timer. Nominal completion includes an explicit collision-free10cm upward withdrawal after release and a fixed terminal reference held through32s; fixed holding target is not repeatedly reset to achieved TCP. The10cm design is a prospective choice, verified by the pilot, not an optimality claim.

The teacher may use simulator object/goal truth OFFLINE to plan. Its commands must be executed in source physics, with new real RGB/state and commanded trajectories recorded. If using a joint-space planner, convert COMMANDED q via fixed FK and re-execute through the EEF source controller to obtain training observations. Do not attach old joint-execution RGB to altered EEF labels. Direct EEF teachers avoid a redundant joint pass when equivalent evidence is available. At most two physical passes per requested teacher episode are budgeted; all unsuccessful starts remain visible.

Pilot admission: T1>=7/8 full successes is required before T1 data/training; T2>=7/8 is required for its data/training, but a T2 teacher blocker must NOT prevent an independently admitted T1 first-policy route. One teacher/scene version is frozen after the pilot; at most one evidence-based semantic repair and its bounded rerun of the same pilot may use technical reserve, never physics/threshold easing. If a substantive teacher strategy change is needed, stop the affected task and report it.

Collect at most64TRAIN+16DEV requested full EEF episodes per admitted task. Require >=56/64 valid TRAIN demonstrations and >=14/16 expert DEV task successes for that task. Preserve all failed source conditions in the manifest and every learned DEV denominator. No extra seeds to top up successful demonstrations. Supervised data include only genuinely valid teacher actions; failure trajectories without valid labels remain diagnostic, not behavioral targets. At least48 shared TRAIN layouts with valid T1/T2 are required to unlock joint training.

Prefer collecting both small teacher banks before first training so one common native processor/statistics contract is frozen. This is teacher-data preparation, not multi-task policy training before T1 works. If T2 teacher/data is blocked, T1 may proceed and T2 remains NOT_ADMITTED; do not let a second-task preparation project consume the first-policy critical path indefinitely.

## 5. Common data release and normalization

Release one immutable episode bank with four distinct namespaces:
1. policy_observation: synchronized raw cameras, actual TCP/gripper, language;
2. action_target: actual sent EEF+gripper commands, control timestamps;
3. feedback/evidence: object poses/velocities/contact/grasp and installed/achieved trajectories;
4. provenance: layout, split, scene/camera/robot/teacher/processor versions, source policy/branch, instruction and episode hashes.

No update needs full raw frames reuploaded to HF. Prefer local immutable arrays and a native-compatible dataset index (GR00T-flavored LeRobot v2 export is allowed, not a second necessary conversion chain). Inputs and targets must share the actual pre-command observation clock. Test grasp/open transitions explicitly, plus no action before observation and no future images. H40 never crosses episode/reset or learner-to-teacher discontinuities; correction suffixes start at the real branch boundary with their own targets. Do not fabricate terminal padding as40 newly observed actions.

Fit common TRAIN-only statistics once on admitted T1 and any admitted T2 TRAIN episodes, including future relative actions over all eligible windows. DEV/confirmation are excluded. If T2 contributed normalization ranges, report it as a known-task-set preprocessing release, NOT T2 zero-shot ignorance. StageA gets no T2 gradient updates. Same mapping/processor/statistics are then used in both stages; no per-task normalizer, row, checkpoint switching or decoder.

Use native normalization and constant-dimension handling. Check raw/encode/decode XYZ<=1e-5m, SO3<=1e-4rad and gripper<=1e-5 on valid labels and representative events; inverse clipping may not silently alter supervision beyond these tolerances. New recovery labels later must satisfy this same contract. If they cannot, retain the raw data and stop that recovery branch for a declared statistics/interface revision; do not silently clip/relabel/exclude all hard states or refit the normalizer during an existing policy evaluation.

Sampler is hierarchical: task -> expert/recovery source -> episode -> behavior bucket -> anchor. Five teacher/evidence-only buckets: approach25%, close/lift25%, carry/lower25%, release/withdraw20%, stable terminal5%. Fractions are total-batch-stream targets with deterministic largest-remainder rounding. In each nonstatic bucket, half the draws prioritize valid anchors whose executed first8 include a relevant phase/contact/command transition, half span the bucket; when no transition exists report the fallback. Use actual teacher events and original command changes, not failed evaluation UID lists. Preserve enough terminal data without letting long static repetitions dominate. Record unique layouts, unique RGB/state, event-first8/tail-only counts and draws, not just number of overlapping windows.

## 6. G2: first T1 policy, native pretrained adaptation, finite training

Start ONCE from the locally verified original nvidia/GR00T-N1.7-3B revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`, with pinned official code `51d4c89f72fda44cbf77285c6a8114b52676b8a1`. The untrained row10 has no claimed known prior coverage. Use its original bytes; do not graft DROID rows. This new-data route does not inherit a selected StackCube-specialized checkpoint. Reuse loader/runtime proofs, not old failed weights. The original model remains an immutable recovery/reference artifact.

PLACE4096: maximum4096 optimizer updates, effective batch8, preferred H200 microbatch4 xaccum2. Native masked flow-matching ONLY; same10/132 masks and full native epsilon/time sampling. Train row10 state/action encoders+decoder and shared DiT, frozen VLM/other31 rows/position/vlln/VL attention as the established scope. Peak interfaceLR1e-4, DiT1e-5; AdamW betas(.9,.999),eps1e-8, projector decay0,DiT decay1e-5,gradclip1,cosine4096 with205 warmup. These are fixed engineering settings, not a guarantee or an exact-all-default official recipe. FP32 trainable masters/BF16 autocast; BF16 K4 inference/FP32 physical decode, TF32/FP16 off. No custom auxiliary loss or a new search over LR/dropout/inference steps.

Initialize optimizer once. Continuations within a phase restore model+Adam+scheduler+RNG+data cursor. Save immutable model checkpoints at2048 and4096; a1024 diagnostic snapshot is optional within the declared DIAG budget and cannot open an extra physical screening panel. Fixed DIAG per task consists of8TRAIN+8DEV layouts times6 event/phase anchors=96 rows, one predetermined native noise each; do not treat those rows as independent demonstrations.

At2048/4096 evaluate fixed first8TRAIN T1 layouts. If TRAIN>=7/8, evaluate all16DEV. The FIRST checkpoint that has TRAIN>=7/8 and DEV>=13/16 is frozen for the single CONFIRM_A32; do not continue training just to fill the budget. Up to two base checkpoint screening rounds. There is no requirement to finish a matched ablation before delivering a usable first policy.

## 7. G3: at most one evidence-based corrective-data phase per stage

If no development-qualified model after a complete primary4096 phase, at most one32-slot learner-state teacher collection is authorized. It is a bounded dataset-aggregation mechanism, not online teacher assistance and not an unbounded second optimizer search.

StageA: take the eight fixed TRAIN rollout trajectories of PLACE4096, branch at2,6,10,16task seconds (32 requested states). StageB: take the eight fixed TRAIN layouts for each task of the final shared4096 model, branch at6 and14s (32 requested states total,16/task). Use actual saved command-prefix replay or a fully validated exact simulator-state restoration at a declared branch boundary; never teleport the object into the intended state. Compare restored actual TCP/object/gripper/cameras against saved source state. An unvisited/early-aborted state is unavailable, not replaced by a different easy state. Exact duplicate branches are not independent data.

Teacher plans from this actual state using the SAME task framework, with new physics feedback and RGB/EEF records. Types include reapproach, regrasp, realignment after grasp offset, placing, release/withdraw and recovery from a disturbed completed placement. It may use truth only offline. Do not label the original failed prefix as expert behavior or use original teacher actions indexed by the same global time. No new objects/layouts/cameras are introduced.

At least24/32 requested branches must be distinct, faithfully restored, safely corrected by the original end time and codec-compatible. StageB also requires>=12/16 per task. Failed/missing branches stay in the32 denominator and diagnostic archive. If this gate fails, do not train on unvalidated corrections and do not replace seeds or enlarge the repair count.

One REPAIR2048 phase then inherits the last primary full recovery, retains Adam moments, declares a fresh cosine2048/warmup103 with interface5e-5/DiT5e-6, same scope/loss/precision. Within each task,50% original expert replay and50% valid correction suffixes, equal episode/branch weighting then behavior-bucket sampling. StageB remains task50/50. At its final2048 only, rerun the fixed TRAIN/DEV screening gate. No second repair/data aggregation cycle. A failure returns an actionable gap, not another default4096 run. This engineering route cannot isolate recovery-data causality without a separate equal-update replay control; no causal claim is made here.

## 8. G4: independently confirm and preserve the first useful policy

CONFIRM_A uses only T1 on previously untouched layouts285300..285331, exactly32 conditions, once, with the chosen frozen candidate and original per-condition noise rule. Pass requires>=26/32 complete T1 successes. Provide per-condition subpredicates, confidence interval and all failures. This is a source-domain engineering prototype threshold;32 trials do not prove a population success rate>=80%.

If confirmation fails, do not retrain using it or select a second candidate. Mark DEVELOPMENT_ONLY_NOT_QUALIFIED and stop model training in this goal. No T2 optimization. Preserve the model and diagnosis.

If it passes, immediately deliver immutable `PLACE_V1_SOURCE_QUALIFIED`: model+processor+statistics+mapping+K4+decode, local load command, full recovery and real videos. Store it in a separate protected location from StageB. selected/deployment remainNONE for target hardware; the SOURCE milestone is explicitly positive. No HF bulk/third-party review is required to start the next already-authorized gate.

## 9. G5: one shared model for T1 and T2, no task-specific switch

Only after G4 PASS and T2 teacher/data/paired-layout gates PASS, initialize SHARED4096 from the qualified PLACE_V1 full state. Keep the identical common processor/normalizer/EEF10 row and observation/action contract. Restore Adam moments with a declared fresh phase cosine4096/warmup205, same peak interface1e-4/DiT1e-5. No architecture, VLM unfreeze or task-specific head change.

Task sampling is exactly50%T1/50%T2 across the frozen draw stream; per-task behavior and expert/recovery weighting follows sections5/7. At inference there is ONE weight set, ONE row10 and ONE decoder. Language selects the task; no dispatcher runs a different checkpoint or supplies a teacher phase. Source joints/object truth remain excluded.

Before shared updates, run PLACE_V1 on T2 DEV16 once as the fixed acquisition reference; prior T1 DEV from G4 can be reused, no extra T1 rescoring. It is a known-task-set untrained-task diagnostic, not an official zero-shot claim. This initial T2 result does not choose hyperparameters.

SHARED2048 and4096 each get fixed TRAIN8 per task and, if desired for the predeclared final diagnosis, DEV16 per task; final4096 DEV is always reported for both tasks, not hidden because TRAIN failed. At most three screen rounds including a permitted REPAIR2048 endpoint. The first endpoint eligible for confirmation must reach TRAIN>=7/8 and DEV>=13/16 on EACH task, plus both tasks succeed on at least12/16 identical DEV layouts. If primary training fails, at most the single section7 correction phase is permitted. Do not sacrifice a failed task behind the average score.

Then freeze ONE candidate and run CONFIRM_B layouts285400..285431 for BOTH instructions, each from the same actual reset state. Require>=26/32 successes per task AND>=24/32 layouts where both requested goals succeed. Record wrong-goal placement and paired initial RGB/state identity; disjoint target sites make a fixed one-goal action distinguishable. A single shared model must satisfy both tasks, not two separately selected models.

A-only confirmation layouts are never added to any later training or tuning set. CONFIRM_B failure ends StageB; no second candidate or repeat. Always retain and deliver the qualified PLACE_V1 regardless of shared-stage failure. Success status `SHARED_TWO_TASK_SOURCE_QUALIFIED` establishes a narrow shared two-skill policy, not unseen-task, broad scene, cross-embodiment or real-time generalization. New scene-specific StackCube results do not overwrite earlier0/4 or0/8 results.

## 10. Execution efficiency and engineering limits

Single owner, up to3 mutually scoped workers: SOURCE/DATA, TRAIN, AUDIT/EVAL. Stages advance automatically when explicit gates pass; no recurring secondary approval. CPU preparation can overlap only without delaying the first-policy path. Model/physics overlap requires actually independent legal GPUs. H200 training+Corelab inference/source physics are preferred, not assumed available. One accepted training lease/job, no duplicate launch, no unrelated-job cancellation, no falsified PBS identity.

Reuse native K4, original source physics rate and two-worker pinned-memory CPU caches. Cache keys now include instruction/task/processor; no neural feature cache unless independently proven semantically identical and separately declared. Avoid worker/compile/precision searches. Direct immutable model-only changed-tensor transfer with full reconstruction checks; complete Adam stays on training host. PI0284 scoped transfer key was revoked, so re-establish an approved read-only scoped channel rather than assuming it remains valid. HF is deferred archival, not upload-then-download feedback transport.

Use reusable recorded-input tests and one short8 vs4+fresh4 proof per genuinely new continuation semantic, not a full proof for every export. Total discarded canary<=64 updates, memory<=32, completed technical rollback<=128, training recoveries<=4 for the entire new goal. Preserve failures and actual counters. Detailed gradient/mode checks in proof, first8, each128 and checkpoints; regular finite-loss/global clipping checks retained. Data/noise/cursor/full identity accounting remains intact without repeated copying/hashing all model tensors on every microbatch.

Save complete local recovery at least1024-update boundaries and all phase transitions/stops. Keep original pretrain reference, first qualified PLACE_V1 and last resumable active checkpoint. Only exactly inventoried redundant exports/superseded nonunique caches may be retired after verification. Check actual quota or an owner-approved local capacity allocation; df alone is not personal headroom. Archive full unique data locally before proceeding; temporary HF errors do not trigger scientific reruns.

## 11. New finite budget (no old-budget transfer)

Main optimization A4096+B4096, optional corrections A2048+B2048: formal<=12288 updates /98304 chunks at batch8. All completed updates<=12512 including64canary+32memory+128rollback. A cap6144 formal, B cap6144 formal; B cannot borrow A unused capacity to search extra models.

Teacher pilot/data at most(8+64+16)layouts x2tasks x2physical passes=352 starts. Corrections<=32states x2stages x2passes=128 starts. Source-model screening/confirmation A<=104episodes (3x8TRAIN+3x16DEV+32CONFIRM), B<=224 (16 acquisition baseline+3x16TRAIN+3x32DEV+64CONFIRM). Nominal full-path maximum808 starts; hard840 including at most32 technical same-condition replacements, valid quality retries0. Each correction prefix/replay is counted, no resetting counters when called data collection.

Quality requests<=32768, actual DiT heads<=131072 (K4 only), controls<=580000, per-environment100Hz integrations<=2900000, RGB pairs<=580000, combined arm/gripper dispatches<=1160000. Teacher decision/control evaluations count separately, teacher control decisions<=400000 and motion-planner invocations<=4096; CPU-only plan candidates cannot conceal extra physics. Proposed maximum model episodes328 at80calls plus up to12 DIAG96 panels and128 load/retention calls total27520 nominal requests, below cap. Optional DROID retention is descriptive with original route/inputs, not a qualification detour.

Resource hard ceilings across devices:32 new GPU-hours,256CPU-core-hours,240GiB owned new outputs/cache. These are ceilings, not estimates or a requirement to use them. Source initializations/neutrals also count. Keep enough reserve to evaluate/save the final required candidate; do not reserve all unadmitted future stages so aggressively that training never runs. Stop on exhausted cap; no automatic next-day/new-goal/reset. Old PI0284 and earlier budgets remain separately sealed.

No target-WBC/model/physics/hardware calls; no extra third task, background/lighting sweep, new large model download, full VLM training, RL, hidden oracle runtime or per-task checkpoint policy is permitted.

## 12. Required deliverables, interpretation and sources

First deliverable is a real separately defined T1 source policy, not a video-selected StackCube success. Second is the same-interface two-task successor or an explicit shared-stage gap while preserving T1. Each includes: local load recipe; full recovery; exact task/scene/teacher/data/processor identity; split and exposure inventory; genuine sent/installed/achieved trajectories; grasp/lift/target/release/withdraw/hold funnel; actual source videos including failures; CPU task recomputation; finite consumed budgets and unchanged frozen-WBC scope.

Reports distinguish NOT_EXECUTED, TEACHER_OR_CONTRACT_BLOCKED, VALID_QUALITY_FAIL, DEVELOPMENT_ONLY, PLACE_V1_SOURCE_QUALIFIED, SHARED_TWO_TASK_SOURCE_QUALIFIED. No target selected/deployment before later independent qualification. No claim that this multi-change new-data milestone proves recovery, normalization, wider target or pretraining as the unique cause. The prospective episode counts, task tolerances and update counts are engineering choices, not literature convergence guarantees.

Actually read: PI0284 FINAL_REPORT and canonical terminal, README/2026-10-05 SIM_FIRST_OBJECT_MANIPULATION_ROADMAP, inherited pi0282 codec.py/cpu_inputs.py, and prior fixed reports in the conversation. README still advertises the superseded09:00 priority: newer Issue11 authority and CLOCK_RESOLUTION govern. Official references: ManiSkill motionplanning and replay docs describe source planning/replay, not a drop-in completed T1; NVIDIA Isaac-GR00T supports native modality configuration and mixed-data adaptation. This reviewer did not run SSH/GPU/physics or inspect all raw videos. Newly proposed task modules, data and policies are NOT yet implemented/completed by this document.

References: https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html ; https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html ; https://github.com/NVIDIA/Isaac-GR00T/tree/51d4c89f72fda44cbf77285c6a8114b52676b8a1 ; project PI0284 final at1d0e16bd8dee9595b55db1685c388947bab95830.

Startup: preserve PI0284 terminal, implement the isolated same-scene T1/T2 task and real instruction-aware native loader; validate teacher/data/common codec; train T1 first from pristine, optionally one qualified correction phase; confirm and preserve PLACE_V1; only then joint-train both tasks and confirm one shared model. No endless StackCube continuation, no WBC/real hardware, no HF-critical transport; advance on gates and stop on finite limits.

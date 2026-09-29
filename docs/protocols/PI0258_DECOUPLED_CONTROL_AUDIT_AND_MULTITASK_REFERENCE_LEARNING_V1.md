# PI0258 — Separate controller capability from multitask EEF reference learning

Authorization: **PI0258_DECOUPLED_CONTROL_AUDIT_AND_MT_REFERENCE_LEARNING_GO_V1**

Issued 2026-09-29 in response to the user requesting careful review of the PI0256 terminal result and a new goal advancing generalization training.

This is a NEW scientific scope, not a repair that silently changes PI0256. It conditionally authorizes actual synthetic-data production, two matched 6000-update models, and fixed offline evaluation. It does not authorize learned-policy physical execution. Controller characterization and reference-generator learning are separate studies with separate claims.

## 0. Source authority and accepted history

Repository: `Leo-6-maker/go2-arx5-hierarchical-vla`; scientific ledger: Issue #11; PR #77 is pointer-only, draft/unmerged.
Source base: `71d9a483897d8f9d9dd0eac4bb433b3c96a327c9`.
Working branch: `pi0258/decoupled-control-audit-and-mt-reference-learning-v1`.

Read completely before implementation:

- Latest terminal canonical `5889109239` and `reports/PI0256/CLOSEOUT.md` at the source base.
- `reports/PI0256/REVIEW_AND_HYPOTHESES.md`, `PREPARATION_V2_TCP_SYNC_CONTRACT.json`, `preparation_stage_summary.json`, `preparation_replay_tcp_sync.json`, `native_canonical_cpu_replay.json`, and `runtime_readback.json`.
- Actual source `scripts/pi0256_collect.py`, its imported preparation code and node-local reference helpers, not just same-named repository files.
- PI0256 authority `fcfd37bf2ced86ae3c3ce9ee43c2a0b9ec810450`, canonical `5887791858`, especially the explicit Motion3-v2, processor and model-training sections.
- PI0255 initializer comparison, capability matrix and actual registry/scorer sources; PI0254 U/R terminal `5887330960`.
- Independent video workflow PI0257 at `1e760ba055106cb85b508711f1466d1cb78a9656`, canonical `5888937489`; use it only where its scope and provenance allow.

PI0256 remains PREPARATION_V2_BLOCKED: A6/B6/Bfix6, 0 qualified; 18 starts, 3356 native control steps, 6676 native actor calls. No task phase, new imitation demonstrations or GR00T query/update occurred. Bfix clean XYZ RMS is 10.564–23.149 mm but SO3 RMS is 0.651731–1.490608 rad. Rotation error exceeds 0.10 rad on 249–250/250 ticks, angular speed exceeds 0.20 rad/s on 229–248, and translation speed exceeds 0.02 m/s on 200–245. The soft-limit predicate is true on every scored tick, including gripper bounds. RMS alone is not the continuous conjunction.

The derived TCP alias correction is real and successful as a cache invariant: camera-induced reset-to-first-actor shift is zero in all corrected cases; independent FK agrees within 0.752 micrometers and 2.2581e-6 rad. Raw/native reference canonicalization also matches numerically. These facts narrow the remaining hypotheses; they do not establish correct actor conditioning or a capable orientation controller.

The original C0 positive result established a finite world-position hit, not arbitrary-pose SE(3) stabilization. B transported both the curve and its orientation to each reset pose and appended a hold. This is a different physical demand; geometric rigid-transform consistency does not prove invariance of a gravity/contact/joint-limited frozen controller. Do not call this observation a proved unique root cause.

U's new-support improvement and old-support failure remain. R is a zero-update integrity block, not evidence that rehearsal failed. Do not restart either, transfer old unused budgets or change old selection. Historical selected_checkpoint and deployment_selected_checkpoint remain NONE.

Reuse the existing PI0256 archive receipt at `leoyuer/pi0246-gate-pass-campaign@c580474a2d91c37807a0dc215b3658cb04f22488`, prefix `evidence/PI0256/preparation-A-B-source-correction-v1-20260929`. Do not duplicate its 14364-member verification solely to create another audit count.

## 1. Mainline amendment: two independent evidence tracks

A. **Controller capability characterization:** bounded source/record analysis and, after contract freeze, at most 12 new reference-only native simulations. It asks whether the residual problem concerns conditioning, reference attitude/support, temporal behavior or safety-predicate semantics. It does not search a new controller or amend the old preparation gate.

B. **KIN-MT reference learning:** construct a separately identified, kinematically verified, rendered multi-behavior corpus; train MT-Base and MT-G; evaluate reference generation and language-conditioned behavior on new held-out synthetic scene families.

Track B does NOT require Track A to pass the 0.5-second preparation gate. Its prerequisite is its OWN geometry/observation/data/model integrity, not stable WBC execution. A bad FK, coordinate, tokenizer or input binding shared with B remains a real B blocker. Failure of controller capability alone is not.

This explicitly replaces the previous serial dependency only for the NEW kinematic research corpus. No failed physical trajectory becomes a successful demonstration, and no old threshold changes. Kinematic placement is allowed only in an isolated renderer; it is NOT allowed inside a physical episode to evade failure.

Required evidence labels:

- DYNAMIC_CONTROLLER_CHARACTERIZATION: actual reference-only native physics.
- KINEMATIC_RENDERED_REFERENCE: prescribed joint/root poses plus independently verified FK and rendered cameras; NOT achieved WBC motion.
- OFFLINE_LEARNED_REFERENCE: model predictions with no dynamics execution.
- PHYSICAL_POLICY_SUCCESS: remains UNMEASURED under this GO.

A learned generator from B is a synthetic-reference research initializer, not a deployment-qualified policy. It must later earn controller compatibility and real closed-loop evidence. No claim that synthetic improvement necessarily transfers to physics, real cameras, new skills or hardware is authorized.

## 2. Track A0 — Finish useful attribution from existing bytes

Produce a compact capability/contract report, reusing the actual Bfix six-case arrays. Do not begin a new all-repository audit.

1. Decompose each joint limit signal by exact joint name, commanded/passive role, hard URDF bound, native soft bound, penetration magnitude, duration, and whether it is a native termination, penalty or external admission check. Determine whether gripper alone explains the all-tick soft flag. Preserve the existing aggregate and all arm/leg violations. Do not remove a joint merely because it is inconvenient.
2. Show same-tick SO3 error, angular speed and XYZ tracking through moving and terminal-hold phases. Express the SO3 log residual in documented frames only for diagnosis; no fitted offset is applied to scoring or actions.
3. Trace the actual desired rotation through canonical reference -> native goal -> arm observation -> rot6d/quaternion convention -> actor output -> actuator/body commands. Inspect native training/config/reward files where available for orientation range/weight, gait and goal overrides. Missing original training support stays UNKNOWN; do not infer it from checkpoint filenames.
4. Examine action clipping/saturation, soft/hard margins, body attitude, goal-error cache and reference time law. Distinguish intact reference loading from correct actor conditioning; one does not imply the other.
5. Compare original C0 evidence scope with transported Bfix: initial/target orientation, root/DOF distribution, moving duration and hold, commanded versus achieved orientation, and original success predicate. Preserve the fact that C0 position-hit PASS did not include the new stabilization criteria.

Do not claim that a gripper soft-bound issue explains 0.65–1.49 rad physical errors. Do not call a timing abort a fall when actual native event fields say otherwise. Do not use post-autoreset state as the pre-reset failure pose.

## 3. Track A1 — One fixed reference-attitude comparison, not another settle search

Optional physical characterization is authorized only after A0, actual-source/read-only recorder tests, numeric reference freeze, and resource preflight. It is not needed to release B.

Use exactly the six old PI0256 Bfix reset/scene conditions. For each, run two fixed references with the SAME corrected cache, initialization order, immutable anchor, actor code/weights, gait, physics, camera cadence and reset distribution:

Let source C0 be (p_c(t), R_c(t)), and measured reset-time independent FK pose be (p_0,R_0).

- N, source-attitude preserved: p_N(t)=p_0+p_c(t)-p_c(0); R_N(t)=R_c(t).
- T, transported attitude/curve: Q=R_0 R_c(0)^T; p_T(t)=p_0+Q[p_c(t)-p_c(0)]; R_T(t)=Q R_c(t).

Both retain the same original C0 time law and the already documented terminal continuation to five seconds. N is NOT an exact replay of the old successful episode; it retains source orientation while translating the curve to a new start. T is the prior Bfix demand under a new explicit comparison. The contrast is a package/reference-support contrast, not a uniquely isolated rotation coefficient effect.

Run the original gate before binding; an N or T REJECT is a valid result, not permission to bridge, smooth or interpolate its start. Preserve native reset/termination/hard limits and original diagnostics. Report the original PI0256 strict conjunction without requiring it for completing this characterization. No thresholds are relaxed to manufacture a new preparation PASS. No gain/weight/dynamics/history reset, actuator, new pose/seed, controller retraining, extra prefix or candidate reference is authorized.

Maximum 12 first-valid starts, each at most 251 native control steps including implicit reset, at most 3012 steps total. An exactly identical fully recorded T tuple may be reused instead of rerun; reference, state, runtime and noise identity must all agree. Reuse never transfers its allocation to another condition. No valid-failure retry. A detected hard integrity defect blocks affected starts; do not patch and relaunch arbitrary physics within this fixed comparison. CPU-only minimal corrections may be proposed for a later physical version.

Regardless of outcome, publish the controller characterization and its limits. Do not launch new Motion3 expert collection, learned-policy physics, or another preparation family from it under this GO.

## 4. Track B0 — A clearly synthetic, kinematic Motion3-v2 corpus

The learning question is: can one GR00T model map the same visual scene and pose to the requested target/route/direction, and do so on held-out synthetic scene families? It is NOT whether the frozen WBC can yet execute that mapping.

Reuse M1 target-pose approach, M2 specified-side route, and the explicitly approved M3-v2 270-degree open contour from PI0256. Preserve their semantic success definitions, dimensions, H40, 0.05-second action samples, natural language distinctions and neutral unexecuted gripper. Preserve coexisting scene elements, so scene background does not reveal the task. Do not silently turn M1 into position-only or replace M3-v2 by a closed lap.

Create a new `PI0258_KIN_MT_V1` namespace: 6 development families, 40 TRAIN families, 10 KIN-VAL families, and two separately sealed 20-family TEST manifests for composition and geometry. Each executable family has three tasks and two command variants: 336 planned kinematic sequences, not 336 dynamic episodes. TEST is metadata-only and remains unrendered/unqueried.

Before producing data, seal one deterministic registry algorithm, all factor values, source IDs, original task-derived geometry, task durations, language assignment, renderer/collision/FK/IK versions and numerical solver settings. Use seed25801 for family ordering and25802 for language; remove cross-version duplicate scene/reset tuples before fixing the finite set, retaining a duplicate log. Do not use old failed preparation conditions as validation. Do not expand an easier candidate pool after seeing IK failures or model scores.

Root/leg configuration is the source-configured nominal robot pose, explicitly recorded; the arm starts from the source-configured default joints. These are KINEMATIC initial states, not asserted settled states. A hard-limit-invalid default requires a source/geometry decision, not hidden clipping. Use the same start for both commands and all behaviors in a family. No task-dependent initial camera/pose shortcut. No new base-motion planner or physical stabilization claim.

### Kinematic reference and image generation

Construct the prescribed continuous task reference, solve its arm IK with the fixed root/leg pose, and independently recompute FK. Warm-start each timestamp from the previous deterministic solution; no random multistart, path search or task-outcome-driven reference redesign. Freeze one bounded solver implementation before the finite data program: reuse an existing verified solver where available, otherwise a deterministic damped least-squares adapter over the existing FK is allowed. Cap iterations at200 per timestamp and record all failures. The numeric damping/Jacobian/step conventions must be in the pre-generation card and cannot change after outcomes. Solvers operate on joint candidates; no model predictions are clipped or repaired.

At each active sample require independent FK position residual<=0.002m, SO3 residual<=0.01rad, hard joint limits, finite continuous joint/pose sequences, source-specified joint velocity limits, and collision clearance for the robot/world using a separately verified geometric check. Record adjacency/self-collision exemptions from the source, not an all-collisions-off switch. IK success alone does not prove collision avoidance, balance, force feasibility or controller reachability. Missing collision capability blocks that family from the kinematically-qualified corpus.

Render original-resolution body/wrist cameras from the SAME robot joint/root pose used for StateV1 FK, using existing asset/TCP and camera conventions. The state-origin field is `KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT`. The desired reference and the FK-implied current pose remain distinct fields. Pairwise commands at a common state must have identical actual RGB/StateV1 before language is supplied. Never paste a new trajectory over unrelated expert frames and call it training data.

Prefer a true step-free pose renderer. One isolated, pinned adapter is allowed; do not upgrade the main native environment or replace WBC. If engine construction unavoidably integrates physics, permit at most ONE explicitly logged zero-policy initialization step per family, with64 total initialization-step allowance including strictly failed setup attempts. Discard any resulting physical pose and render from the explicitly declared kinematic pose; there are ZERO integration steps after trajectory-pose placement. This is not a simulated task demonstration. Do not use a WBCEnv factory that advances an unknown number of steps. Renderer-only state placement is an explicit synthetic-data operation; it is prohibited in Track A physical runs.

Read back/prove actual rendered transforms, FK/TCP and camera projection. If a no-integration renderer cannot show the placed joints correctly, report RENDERER_BLOCKED rather than advance hidden physics. Keep overlays, metadata, coordinate text, future paths and task IDs out of policy pixels. Scene markers described in the language are allowed; floating-point coordinates encoded as pixels are not.

Use the inherited 0.10-second observation sampling and actual synthetic trajectory parameter timestamps, labelled `synthetic_reference_time`, never wall time or measured simulator time. Sample H40 future references at0.05s; document terminal padding and masks. Export complete source sequences, not only favorable snapshots. Maximum60000 active windows and120000 body/wrist image pairs; these are upper bounds, not data targets. Shard arrays instead of generating tens of thousands of synchronous small NFS writes.

### Geometry and observability admission

Run the fixed 12 development sequences per task. At least10 must satisfy kinematic, raw-label, collision, visibility and source-integrity QA before producing that task's fixed80 TRAIN+20 KIN-VAL sequences. No physics-success terminology is used. Keep every failed sequence and reason. No replacement of valid IK/geometry failures.

For each data-qualified task require>=64 usable TRAIN sequences,>=16 usable KIN-VAL sequences, and>=24/6 complete command pairs. Every usable sequence needs>=95% qualified active windows and coverage in all four motion-phase quartiles. Inspect starts, branches, slow sections and terminal approach; do not quietly remove the hard first quarter. Exact label round-trip and gate checks are described below.

All tasks satisfying these pre-model criteria are included. K=3 -> KIN-MT3; K=2 -> explicitly named KIN-MT2; K<2 -> no model training. No choosing task subsets from model outcomes, no MT1 fallback, no replacing failed tasks with two colors of reach.

Before model queries verify that command-relevant cues are visible in actual rendered input and the correct/incorrect command futures are separable on the declared H40 portions. Retain common prefixes as non-discriminative. Check exact input aliases and near-neighbor conflicting futures, including time-driven approach versus traversal. No hidden phase/source ID/target coordinates is added to resolve ambiguity. Structural non-observability blocks that family; merely finding no duplicate in a small corpus is not a proof of full observability.

## 5. Track B1 — Shared processor and real-data integrity

Use the PI0256 proposed C_MT_V2 as a new, explicit `PI0258_C_KIN_MT_V1` identity. Historical C_T is untouched. Preserve relative-action conversion, row-packed rot6d, XYZ units, slot10 mapping, H40,132D padding/masks and neutral gripper convention.

Use deterministic full-field-of-view resize and pinned tokenizer, no crop/flip/color jitter in either arm. TRAIN-only q01/q99 affine statistics are calculated once at the original normalization stage AFTER the unchanged relative-action conversion, from raw-provenance/geometry-qualified TRAIN windows BEFORE codec eligibility filtering. No percentile clamp; retain finite out-of-range values and report them. Range floor1e-6 in coordinate units; explicit constant-channel round-trip. No filtering/statistics iteration using KIN-VAL or model scores.

For every enumerated window validate physical label->raw chunk/gate AND actual processor->inverse decode->same gate, using synthetic FK state and reference time. Require float32 CPU round-trip XYZ<=1e-5m, SO3<=1e-4rad and exact timestamps/mask/dimensions. Failures remain indexed. A kinematic gate PASS is not a WBC execution certificate.

Bind this exact processor to both training physical losses and inference decode; a metadata-only change is insufficient. The actual collator permits only body_rgb, wrist_rgb, StateV1 and language. Negative tests for expert future, task/source ID, success flags, world target, synthetic clock and metadata must fail closed. No extra backbone input or trainable embodiment row is authorized.

Freeze complete source/visit schedules and processed input identities. Check in separate processes and across iterator restore; audit Python/NumPy/Torch CPU/CUDA/worker randomness. Same seed is not the actual noise tensor. Record real tau, native epsilon and pure S4 noise identities under source/visit keys, with no arm-dependent seed.

Passing B's data, processor, source and resource gates automatically releases canaries/training, even if Track A is a valid failure or remains a separately reported unresolved capability question. Do not wait for PI0256 settling PASS.

## 6. Track B2 — Real recovery canaries and matched model training

Fixed initializers:

- MT-Base: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- MT-G: `leoyuer/pi0246-gate-pass-campaign@efecea485ae77000599d19171b02491f8d514f42`, prefix `arms/PI0252/r1-speed-margin-20260928/G/checkpoint-2000`.
- Tokenizer/backbone metadata lineage: Cosmos revision `9ce19a195e423419c349abfc86fd07178b230561`.

Reuse PI0255 actual key/shape/tied-alias receipts. Base retains original slot10 parameters; G retains adapted slot10. Verify complete actual model loads, trainable/frozen scopes and dtype, not merely headers. U/R/FULL/latest-model substitutions are forbidden.

For EACH initializer run continuous8 versus4+FULL-Trainer-save+fresh-process-restore+4 updates using the same first8 real rows and original6000 LR cursor. Total32 discarded canary optimizer updates. Compare model/optimizer/scheduler/RNG/data cursor and actual continued updates; validate all four S4 gradients and frozen tensors. Main experiment starts fresh from the pinned initializers, not these canaries. Numerical tolerance/determinism rules must be frozen before the canary, never widened to accept its mismatch.

Then train exactly6000 new canonical updates per arm, two endpoints total. K=3 ->2000 per task; K=2 ->3000 per task. Shared model per arm handles every admitted task.

- Fresh AdamW, LR1e-5, betas(.9,.999), eps1e-8, weight_decay1e-5, grad norm1, batch1, accumulation1, warmup10, cosine6000; explicit cursor0 accounting.
- BF16 frozen backbone and existing FP32 action-head/freeze scopes. Projector/diffusion trainable; no LLM/visual/vlln unfreezing, new adapters or model-capacity change.
- Original native loss + full pure-noise S4 physical position0.4, increment0.1, rotation0.1, speed-margin0.1; inherited1.8m/s margin is an auxiliary reference-learning target, not a new safety certification.
- Same synthetic corpus, C_KIN_MT, sampler/runtime, input tensors and source/visit/noise schedule in both arms.
- Local schedule RNG25803: balanced task blocks, alternate commands within task, uniform complete-pair TRAIN family, then source sequence, motion quartile and qualified window. Seal6000 actual rows before training. List unused/unpaired sequences honestly.
- Full Trainer at1000/3000/6000 for recovery/archiving. Only step0 and final6000 are evaluated; intermediate endpoints cannot be selected after seeing scores.

This first study intentionally retains one loss recipe to compare initialization; it does not prove this pointwise supervision is optimal for multimodal tasks. If semantically identical allowed observations require different futures, fix the DATA QUESTION before training, not by injecting oracle IDs. No coefficients, seed, curriculum or loss sweep.

G's historical learning cost is additional. Equal new updates is not equal total compute. A single failed arm does not prevent the other valid independent arm from finishing; a complete two-arm comparison requires two complete endpoints. Integrity BLOCKED is not quality FAIL.

## 7. Track B3 — Fixed synthetic-domain generation/generalization evaluation

Evaluate both step0 initializers and both final6000 endpoints under the SAME new processor. This separates processor migration from subsequent learning. Freeze source IDs and two actual noise tensors per input before the first quality query; no best-of-noise or samples selected by model performance.

For EACH admitted task, by fixed SHA order independent of scores:

-32 TRAIN windows from distinct sequences, balanced across four motion quartiles.
-16 KIN-VAL windows from distinct sequences, similarly phase balanced.
-12 KIN-VAL semantic branch inputs from six complete command pairs, using the earliest expert-separable state on a shared prefix. For a start-only shared state with no discriminative H40, report NONDISCRIMINATIVE; do not select favorable model outputs.
-One held-out meaning-equivalent paraphrase per semantic input.
-Empty language once per six shared observations.

Also evaluate O64's64 inputs as a historical-domain regression panel under C_KIN_MT, not as the original C_T qualification. Its loss of old quality does not block this new synthetic learning study, but must be visible.

Total planned queries8*(78K+64):2384 for K3 or1760 for K2. Query logical tuples once. Max64 strictly invalid/no-result query attempts as technical reserve, never resampling valid bad outputs. Keep raw generated S4 states, masks, full physical outputs, noise/input identities and independent CPU metric replay.

Report by task/noise/domain and macro/worst-task: all-row XYZ median/p95, SO3 median/p95, first jumps, peak speeds, gate categories, kinematic feasibility of predictions under the same fixed checker, and changes from each arm's step0. Failed IK of a prediction is a measured failure; do not repair the prediction.

For command pairs report correct-versus-opposite expert margin on predeclared discriminative portions, requested target/side/direction, held-out paraphrase consistency and empty-language ablation. Merely different arrays are not semantic correctness. Preserve all empty/nondiscriminative/missing denominators. Pairwise reference margins are not physical task success.

Per-task absolute trajectory criteria (medianXYZ<=.020m, p95<=.035m, medianSO3<=.050rad, zeroREJECT) are reported, not assumed. A provisional SYNTHETIC_REFERENCE_PROMISING label additionally needs these criteria on every admitted KIN-VAL task under both noises and >=10/12 valid correctly directed semantic branch outputs per task/noise, including the equivalent paraphrases; eligibility and ties are computed from unrounded data. If the fixed corpus cannot supply12 discriminative inputs, semantic claim is INCOMPLETE, not a smaller-denominator PASS. This is a NEW research reporting threshold, not a deployment selector.

Do not select different checkpoints per task. Do not extend training or switch a candidate after an unfavorable result. Report whether Base or G learns and where they differ even when neither meets the provisional label. KIN-VAL is held out from this training but remains development in a synthetic domain; TEST remains untouched. No sim-to-real or physical-generalization claim follows.

Use PI0257-style offline rendering to show both successes and failures. Always watermark KINEMATIC REFERENCE / NO WBC EXECUTION or OFFLINE MODEL PREDICTION. The prescribed robot pose in a synthetic background is not motion caused by the displayed model. No generative success frames or fabricated camera footage.

## 8. Resource and accounting envelope

Old budgets are not transferred. New hard bounds:

| Activity | Budget |
|---|---:|
| Reference-only controller characterization |12 starts /3012 control steps, including implicit resets|
| Isolated renderer initialization physics |64 steps maximum, at most1 per initialization; no WBC actors|
| All new physics steps |3076 maximum; ZERO learned-policy physics|
| Fixed kinematic sequences |336 planned; at most360 generation attempts including24 strictly incomplete technical retries|
| Kinematic active windows / image pairs |60000 /120000 maximum|
| Primary optimizer updates |12000,6000 per arm|
| Discarded paired resume canaries |32 planned|
| Extra engineering/recovery update reserve |64 maximum, no final-step extension|
| Total completed optimizer calls |12096 maximum|
| Raw training action-head calls |109120 maximum =9*12096+256 partial/no-update calls|
| Offline inference |2384 planned at K3; max2448 /9792 S4 heads|
| Hardware, learned-policy WBC, oldDEV/F, TEST execution |0|

A checkpoint recovery may recompute lost suffix updates only within the stated engineering reserve with full raw accounting. Budgets are not permission to omit integrity. Avoid planned interruptions that unnecessarily spend the reserve. Strictly technical import/IO fixes within unchanged contracts have no arbitrary two-submission ceiling; do not duplicate an active job or retry a valid science result. A reserve exhaustion is a truthful resource/integrity partial closeout, not success or an uncounted reset.

PC performs existing SSH orchestration. Corelab runs CPU attribution, native characterization and isolated kinematic rendering; no GR00T training there. Hopper uses legal scheduler allocation for admitted model canaries and training; compatible GPU selection is based on tested runtime/VRAM, not a hardcoded H200 preference. Pure code/CPU tests must not consume Hopper GPU allocations. Offline inference defaults to Corelab after live free-VRAM checks; serialize rather than kill other jobs or change precision.

Use finite existing runners and shard new evidence on verified node-local storage, then durably archive. No watcher/monitor/mail service. No broad home/scratch cleanup or deletion of G/baselines/unique evidence. Check current free capacity and next-write high-water mark plus10GiB reserve; quotaUNKNOWN is not PASS and not proof of no usable space. Preserve a current complete recoverable Trainer, fixed final endpoints and receipts, use already authorized task-local archive/retirement mechanics. Do not require100GiB free AFTER every planned checkpoint write as an arbitrary invariant.

Publish project-owned synthetic data/video/evidence under a unique PI0258 HF prefix with immutable revision, per-file size/SHA and truthful authenticated/anonymous readback. Private WBC assets/source, tokenizer files, secrets and restricted model content stay private. Preserve licence/provenance on derived weights; if public publication is not permitted, use approved private durable storage and publish only admissible metadata. No new public service or firewall change.

## 9. Deliverables and terminal states

Deliver real artifacts, not just more manifests:

- A source-grounded controller capability report, per-joint margin table, orientation/support analysis and all performed N/T outcomes.
- An immutable KIN-MT registry, explicit synthetic provenance, actual image/state/reference samples, IK/FK/collision/visibility QA and per-task admitted counts.
- The actual C_KIN_MT, collator/round-trip/source/noise evidence and real-model resume canaries.
- Two complete6000 Trainer endpoints and fixed offline trajectory/semantic tables, or an explicit per-arm partial result.
- At least one full representative sequence and matched correct/opposite-command prediction visualization per admitted task, labelled as synthetic/offline.
- A gap list for the later physical handoff: orientation authority/support, dynamic start or holding ability, native hard/soft/passive-joint semantics, timing, camera non-interference and learned-policy closed-loop evaluation.

Track A terminal and Track B terminal are independent. Use:
CONTROLLER_CHARACTERIZED or CONTROLLER_EVIDENCE_PARTIAL;
KIN_DATA_BLOCKED / KIN_MT_INPUT_OR_RECOVERY_BLOCKED / PARTIAL_KIN_MT_PILOT;
KIN_MT2_OFFLINE_TRAINING_COMPLETE or KIN_MT3_OFFLINE_TRAINING_COMPLETE, with measured quality and STOP_FOR_PHYSICAL_TRANSFER_ADMISSION.

COMPLETE means the registered experiment finished, not that its scores pass. Dataset construction failure cannot become a model failure. At least two behavior families and valid real-data contracts release training without a new routine GO. Old 6/6 settling is deliberately NOT that gate. Stop only the affected dependency when another independent track can progress honestly.

No physical-data collection or deployment is automatically released by a synthetic positive result or a better N reference. Historical selected_checkpoint=NONE and deployment_selected_checkpoint=NONE remain. A later explicit physical protocol must test the resulting reference generator with the frozen controller before any task-success claim. PR77 remains draft, pointer-only, unmerged.

## 10. Execution handoff

Read the full immutable PI0258 protocol and its canonical. Refresh real resources and use an isolated worktree. Keep PI0256 and PI0257 artifacts/running work unchanged.

First accept and summarize the observed PI0256 negative without rechecking14364 old files. Finish narrow joint/rotation/control-support attribution. If doing the optional fixed N/T characterization, freeze its exact cases and preserve all outcomes; no pose or threshold search.

In parallel implement the separately named kinematic/reference corpus. Do not wait for a controller settle PASS to build it. Kinematic rendering/placement is allowed only here and must never be represented as achieved WBC motion. Keep actual images, FK state and labels geometrically consistent, source-free of oracle leakage, and scene-family splits fixed.

Once at least two tasks satisfy their OWN data/codec/visibility gates, run actual Base/G resume canaries then the two6000-update multitask models under this GO. No redundant PI request, no old-R restart, no extra loss branch, no taskwise checkpoint substitution. Evaluate only the fixed step0/final matrix with paired noise and independent CPU replay. Deliver models, synthetic/offline videos and all measured outcomes. Stop before learned-policy physics; maintain NONE for physical selection. Generalization claims are limited to the registered synthetic support.

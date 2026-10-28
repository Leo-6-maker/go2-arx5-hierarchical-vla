# PI0267 — Executor-decoupled task-space recovery learning

Authority: **PI0267_EXECUTOR_DECOUPLED_RECOVERY_LEARNING_GO_V1**
Date: 2026-10-02. Owner: one main agent, end to end; helpers are optional and never a signature dependency.
Execution branch: `pi0267/executor-decoupled-recovery`.
Source parent: `48048ae37b76534bbbe8a0ba76a9c84ed793e7ea`.
Canonical publication and scientific returns belong to Issue #11. Do not reopen or send routine pointers to historical PR #77. No merge is authorized.

## 0. Decision and scope

The primary question is whether a shared GR00T EEF policy can learn recovery from controller-independent task-space observations, without waiting for the Go2 WBC to successfully visit every trajectory phase. A bounded Go2/WBC transfer diagnostic is secondary and must not block qualified primary data, training, or virtual closed-loop evaluation.

This is an explicit new scientific contract: a procedural virtual tool, task-space reference pacing, controlled recovery states, and virtual execution backends. It is not an engineering relabel of physical robot evidence. It does not prove cross-embodiment transfer or manipulation/contact capability. The tasks remain M1 target/pose approach and M3 directed open 270-degree contour traversal.

Preserve PI0266 PARTIAL / PI0266_FEEDBACK_WINDOW_SHORTFALL, all earlier FAILs, original labels/scores, and formal selected_checkpoint=NONE / deployment_selected_checkpoint=NONE. The already qualified local windows are not retrospectively made into a sufficient training dataset. No PI0266 job is resumed to fill missing phases.

## 1. Evidence motivating the change

Read these exact sources, not old progress snapshots:

- Issue #11 scientific return 5936571754 and completed archive/stop return 5936957413.
- `reports/PI0266/FINAL_CLOSEOUT_RECEIPT.json` at the source parent.
- `reports/PI0266/COLLECTION_MECHANISM_READOUT.md` and `DIAGNOSTIC_MECHANISM_READOUT.md` at `bd250056efe058452981c7460dd3792aa2b23eca`.
- At that same SHA: `scripts/pi0266_expert.py`, `pi0266_control.py`, `pi0266_geometry_worker.py`, and `pi0259_kinematic.py`.

Observed: 96 TRAIN attempts were first-valid, with 79 TASK_CRITERIA_FAIL and 17 IK_RESIDUAL; no recorded safety failure. There are 7,900 locally valid windows, but seven deficient active strata and one missing hold stratum, so no transfer bank and zero GR00T updates. The 17 initial IK failures use a bounded fixed-current-root full-H40 witness, not a global whole-body impossibility proof.

The matched eight-per-teacher DEV diagnostic has CLOCK M1 4/4 versus OBSERVABLE_CORRECTION_V1 M1 0/4. Observable teacher target errors are much larger than its installed-reference tracking errors. This supports examining the reference task-progress mechanism; it does not uniquely identify every failure or qualify the WBC generally.

The old teacher reprojects the current pose onto a globally searched, time-parameterized curve, removes residuals over one second, and executes only the first 0.4 seconds before rebuilding. Existing CPU checks establish anchor/rotation/repeat identity, not repeated-prefix liveness. A supplied constructed 1-D probe demonstrates a possible small-tracking-error stall. It is not a replay of the WBC or a revised historical verdict.

## 2. Primary architecture and prohibited shortcuts

Primary learning chain:

    visible task scene + procedural virtual EEF tool
      -> fresh body-like and tool-mounted RGB + measured task-space StateV1
      -> GR00T H40
      -> unchanged original SafetyGate
      -> declared virtual EEF executor, advance 0.4 s
      -> fresh rendered observation -> next query

Use two 96x96 RGB views, current task-space state (XYZ, row-rot6d, gripper), H40 at 0.05 s, the current universal action mapping and S4 sampling. Do not add a task/family/template ID, elapsed time, curve phase, target coordinates, privileged mask/depth, URDF, joint state, executor ID, or teacher output to model conditioning. A constant existing ABI slot is allowed and must not vary with backend.

Teacher access to task geometry is privileged supervision only. During model evaluation there is no teacher-based output correction, nearest-expert selection, target routing, curve projection, reference retiming, expert fallback, or hidden task future in a buffer. A SafetyGate REJECT fails that episode and is never installed.

The primary executor is mathematical/kinematic, not a robot dynamics simulator. Rendering a tool by setting its kinematic pose is allowed ONLY in this explicitly named backend. It must never be presented as a physical WBC rollout. The secondary physical robot path retains the no-teleport rule.

## 3. S0 — Bounded reconciliation, not another historical full audit

Reuse the complete PI0266 archive receipts; do not re-read all 438,115 historical members. Preserve the existing 11.48 GB evidence and private assets. Main work can start without a storage helper finishing all housekeeping.

From the 16 saved DEV diagnostics, inspect first, middle and final replans deterministically. Check requested versus installed references, the actually executed 0.4-second prefix, current pose, projection phase, and subsequent achieved pose. Record whether the mismatch is prefix progress, phase projection, installation/timing, or tracking. No new GPU queries are needed for this inspection. Missing raw arrays are reported, never reconstructed from a hash.

Run the supplied constructed 1-D counterexample. Clearly distinguish its chosen endpoint bias from empirical WBC dynamics. Do not require a unique neural or controller root cause before the new fixed experiment.

Freeze source/checkpoint/schema/scene/noise identities for this goal. Old protocol text is historical; this GO explicitly authorizes the changes below and supersedes old bans on new labels, a virtual backend and TRAIN-only statistics ONLY within this scope.

## 4. S1 — A teacher must pass executed-prefix progress tests before GPU data production

### Fixed teacher: ARCLENGTH_EXP_CORRECTION_V1

Retain the original task curves and target semantics, but replace their pacing. Build the curve from the source M1 line-plus-orientation path or M3 approach-plus-directed-open-arc path. Use actual source command/colour mapping, not the obsolete PI0255 closed-contour text.

Parameterize the path by cumulative task-space length

    d ell = sqrt(||dp||^2 + (0.05 * d theta)^2).

Discretize the ORIGINAL curve at 0.01 source-second resolution; remove zero-length adjacent samples; interpolate translation linearly and rotation by shortest-path quaternion interpolation. Do not alter its endpoints, direction, obstacle, or geometry. At a current pose, use the nearest pose on this curve in that same metric, with bounded local refinement and deterministic earliest tie handling. If separated equal minima imply materially different H40 futures at the established 1e-5 m / 1e-4 rad label tolerance, mark UNOBSERVABLE_LABEL; do not use a hidden prior phase to resolve it.

For tau_k = k*0.05, define ell_k = min(ell_end, ell_star + 0.02*tau_k). The speed 0.02 is metres of the stated task-space metric per second. Let (p_bar_k,R_bar_k) be the curve at ell_k, e_p=p_current-p_bar_0 and e_R=Log(R_current R_bar_0^T). Output

    p_k = p_bar_k + exp(-2*tau_k) * e_p
    R_k = Exp(exp(-2*tau_k)*e_R) R_bar_k.

This is a new teacher hypothesis, not a proven controller. Its first point equals the measured pose; its first executed prefix has nonzero path progress away from the terminal state. It does not replay the old slow quintic clock or carry secret elapsed time. The terminal rule follows the same equation with clamped curve endpoint, so residual error contracts instead of endlessly restarting a rest-to-rest segment.

### CPU oracle matrix

Use the four already-designated PI0266 controller DEV geometries, two tasks and two commands: 16 conditions. Compare old and new teachers with two mathematical executors, 64 CPU episodes total, at most 40 s each:

- E0: exact tracking of the interpolated candidate at 0.02 s steps.
- E1: first-order task-space tracking with tau=0.15 s: translation advances by 1-exp(-dt/tau) of current desired error and orientation by the same fraction of its SO3 logarithm.

Replan every 0.4 s; never change the teacher's true current pose to a desired pose between replans. Preserve all failures. At the shared approach/arc boundary, report projected phase jumps separately from actual directed path progress. A progress clamp using an unobserved phase is prohibited.

Use the frozen PI0266 V2 task scorer plus its straight-shortcut, reverse, missed-entry and tiny-wrap positive/negative CPU fixtures. Separately assert actual executed-prefix progress, nonidentity rotations, first-point continuity, same observable input/same label, and terminal contraction. Verify task completion at E0 for all 16 new-teacher conditions and at E1 for at least 12/16, at least 6/8 per task. This is a prospective liveness sanity check, not WBC qualification.

No parameter, speed, task, tolerance or seed search follows a valid failure. Mechanical implementation corrections to match this formula are autonomous. If the fixed teacher fails this CPU sanity check, stop PRIMARY_BLOCKED_TEACHER_LIVENESS before rendering/training, with the complete small comparison. Do not spend another 96 physical episodes discovering it cannot progress.

## 5. S2 — Controller-independent, phase-complete visual data

Reuse the frozen numeric scene partitions from PI0266: four controller DEV, 24 TRAIN, eight still-unopened NEW_SIM_EVAL. Preserve actual family identities and protected split exclusions. No extra or replacement scenes. The eight evaluation image/label/score payloads remain unopened until both learned endpoints are frozen; using their predeclared metadata to instantiate the already-frozen evaluator is not training.

Implement a small VEEF_KINEMATIC_V1 backend using the existing renderer/camera adapters, not a new robotics framework. Keep the task markers, obstacle geometry, lighting and admitted camera calibration. Freeze the body-like camera at its canonical initial world pose; the second camera follows the virtual tool via the original fixed tool-camera extrinsic. Render a procedural grey tool with a 40x20x40 mm palm and two 8x8x30 mm fingers at y=+/-15 mm, along the tool's +z axis. Its TCP/orientation mapping and primitive collision union are frozen before dataset production. These are new declared visual/tool semantics, not unchanged robot images. Do not render a fake moving Go2/ARX5 body, call IK to make virtual data look executable, or commit private robot assets.

CPU/renderer fixtures check camera projection and rigid tool-camera transforms, pose/image/state identity, timestamp identity, target/axis/endpoint visibility at the existing pixel thresholds, genuine occlusion counting, and image capture without condition/RNG mutation. Debug masks/depth are stored separately, never painted into policy RGB. Tool-obstacle/ground collisions are checked; target/axis/contour markers retain their original noncollidable role. No claim of whole-body collision feasibility is made.

For each of the 24 TRAIN families, two tasks and two commands, sample 16 equally spaced task-space arclength anchors including both ends. Each produces one nominal state plus six recovery states:

    p' = p +/- 0.010 e_j, R' = Exp(+/- 0.060 e_j) R, j in {x,y,z}.

Signs and axes are paired as written, rotations are left/world perturbations, and gripper stays at the source fixed value. No labels are copied from a different state: run the fixed new teacher on each actual perturbed state and freshly render its RGB. Maximum proposal count is 1,536 nominal + 9,216 recovery = 10,752 labelled observations. Reuse exact duplicate rendered conditions only through a hash-indexed cache; keep proposal and unique-content counts separate.

Admission requires at least 16 of the 24 fixed TRAIN families with both tasks/commands, at least three distinct nominal anchors in each active arclength quartile plus a terminal anchor, and at least one valid recovery counterpart for every selected nonterminal nominal anchor. Each admitted window passes observable-label, original raw gate, task-space geometry, visibility and actual codec checks. Retain all rejected proposals and all eligible families; never replace a failure with a newly generated family. Go2 success, its bounded fixed-root IK, or its joint state is NOT an admission condition for virtual data.

This corpus is CONTROLLER_INDEPENDENT_VIRTUAL_TOOL_SUPERVISION. It is neither real robot demonstration data nor DAgger on learner states. PI0266's 7,900 physical windows remain diagnostic evidence; none are silently relabelled, oversampled to fill missing phases, or mixed into the primary intervention.

### Common processor/statistics rule

Keep the action/state mapping and transformations. First test the frozen PI0264 numerical statistics on the entire admitted TRAIN proposal union with the real processor/decoder. If every label round-trips within 1e-5 m / 1e-4 rad with no destructive clipping, retain them. Otherwise this GO allows exactly one shared TRAIN-only numerical adaptation: XYZ per-axis minima/maxima over the admitted nominal AND recovery TRAIN union, pooled over H40 time, with 5% span padding and a 1 mm minimum span; preserve the existing orientation/gripper conventions. Encode these bounds through the existing normalizer schema, not a second ad hoc decoder. No VAL/EVAL statistics, retries with different quantiles, or per-arm bounds.

Freeze the resulting processor identity before any model canary. Both arms and the donor's virtual evaluation use that same identity. Using a shared range derived from recovery observations gives N range information but no recovery action updates; report this boundary. Donor quality under a new contract is not its historical quality or a pure forgetting estimate. Legacy regression uses its separately pinned OLD contract.

## 6. S3/S4 — Actual paired fine-tuning, automatically admitted after S1/S2

Common donor is PI0264 L2048:

- repo `leoyuer/pi0246-gate-pass-campaign`
- revision `e4290e6ca387abd2de4337b5f12b9c72a5f85c99`
- prefix `arms/PI0264/L/checkpoint-2048`
- complete Trainer tree `2f61d8ed8c8fe8444b7ef23164cb4df2bcacd8a73c93f19be77fcdf459ef6fae`.

Arms:

- N4096: nominal virtual observations only.
- R4096: 50% the exact common nominal replay; 50% recovery observations and their own correct teacher labels.

Both use the SAME new task-space teacher definition on every supervised example. Do not mix old time-paced targets and new arclength targets for an identical input. Balance task, family, command and arclength stage, keep the paired commands in each two-microbatch update, and freeze a deterministic round-robin visitation plan before execution. Corresponding arm slots share family/task/anchor/command/template, initial generative noise and flow time; recovery slots intentionally differ in state/RGB/labels and are not falsely claimed tensor-identical. Select recovery axes/signs by visit order, not model scores. Both arms cover all eligible families and all active stages.

Use exactly the existing eight M1 TRAIN expressions, the fixed current open-contour M3 instructions, the original full-S4 native + .4 position + .1 increment + .1 rotation + .1 speed objective, existing action-side trainable modules and frozen backbone, original precision/masks, fresh AdamW at 1e-5, warmup10/cosine4096, microbatch1 and accumulation2 with the mean of both losses. M1/M3 each receives 2,048 updates per arm; each arm has 8,192 microbatch observations. No adapters, new loss, text expansion or hyperparameter search.

Real recovery canary: continuous8 versus4/save/fresh-process-restore4 for each arm, including exact view/state/label/noise/cursor/optimizer/scheduler identities. Total discarded canary updates=32. Then initialize official arms again from the same donor; do not promote canary weights. Preserve complete 2048 and4096 checkpoints; final-only quality evaluation, N then R. A valid negative N result never changes or cancels R. No ordinary PI reauthorization or helper signature is needed after actual admission.

Training compares recovery-state coverage in a new virtual domain. It does not isolate every consequence of changing the domain from PI0264, and it is not a test of WBC-specific compensation.

## 7. S5 — Closed-loop virtual evaluation, not a prerecorded trajectory video

Freeze the evaluator/noise/scene registry before canaries. Only after both4096 endpoints and four complete Trainer inventories are frozen may donor/N/R quality be queried on the eight held-out geometries.

Primary card: three model points x eight families x two tasks x two commands x two executors E0/E1 = 192 virtual episodes. Use one prospectively fixed actual generative noise stream per family/task/backend/replan; share it between commands and models. The key is SHA256 of `PI0267:VIRTUAL:26741:<family>:<task>:<E0|E1>:<replan>`; first eight bytes unsigned big-endian seed the pinned mature CPU FP32 normal generator. Store actual arrays before querying. Do not use reference labels, command, checkpoint or outcome in the key. One stream is a scope limitation: this does not establish broad sampler robustness.

Every .4 s obtain freshly rendered RGB and actual virtual-tool state, generate original S4 H40, apply original gate, and execute only the installed prefix. Maximum40 s /100 queries per episode. E0 copies the interpolated reference; E1 uses its stated lag law at .02 s. Neither consults the task or teacher. No output smoothing, resampling for a better answer, clock reset, teacher repair, or best-of selection. Label every video VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP.

Use the frozen PI0266 V2 task semantics: M1 actual correct target <=35 mm / .10 rad held .25 s; M3 correct directed entry, all12 ordered arc intervals, no forbidden shortcut/reverse, <=15 mm contour deviation, then correct endpoint/pose/hold. Preserve its implemented deadband and negative tests, not a new fitted tolerance. Keep reference, installed and achieved arrays separate. Report task success, whole-path coverage, gate/collision reasons, endpoint error, movement-only versus hold-only tracking, and per-family/command results.

A virtual research candidate must achieve at least13/16 successes in EACH task/backend group, at least6/8 in each command subgroup, with no dangerous virtual collision or disabled safety checks. Report all failures in the original192-row denominator. N/R treatment comparison is paired and descriptive by independent scene, not by frame or replan. This is local new-geometry transfer in a virtual tool domain, NOT cross-robot generalization.

Offline evaluation: each model point gets256 new-domain windows = eight held-out families x two tasks x two commands x four arclength anchors (.125,.375,.625,.875) x nominal/recovery. The recovery condition cycles the six fixed perturbations by the source ordering. All belong to post-freeze evaluation. Also retain the original440-row legacy panel per model under its OLD processor; reuse the exact donor440 cache. New-domain teacher RMSE and old time-paced reference RMSE are separate diagnostic questions, not interchangeable qualification.

## 8. S6 — Secondary bounded WBC transfer; cannot gate primary learning

After primary model endpoints freeze, use the first two TRAIN families in the frozen source ordering, not whichever looked easiest. Use all four task/command combinations: eight expert controls, then donor/N/R on the same eight =24 model episodes. These are TRAIN-geometry executor-transfer diagnostics, not held-out WBC generalization. New virtual RGB differs from the real body/wrist camera domain; explicitly separate perception/domain shift from controller following error.

Keep the PI0266 already-admitted safe reset, ROW6D research runtime, camera isolation, full physical history, original SafetyGate, current-state collision/limits/native termination, and conservative local-H40 witness. The witness remains a conservative admission rule, not a proof of global reachability. Reuse source-qualified reset checks; do not search seed/pose/gain or drop a failed command. One valid failure is not retried. A system-level safety/identity defect stops only the affected secondary lane; do not kill primary training or pretend virtual outcomes are physical outcomes.

Only model outputs enter model task buffers; no teacher fallback, target-aware projection or teleportation. Fresh real observations every.4 s; maximum40 s/100 queries, with no claim of real-time throughput. Expert/model failure and not-run cases stay visible. WBC weights/gains/dynamics remain frozen. Formal selected/deployment remains NONE even if the limited secondary diagnostic succeeds.

## 9. Budgets and execution autonomy

- CPU teacher progress matrix:64 fixed mathematical episodes; supplied constructed probe is separate CPU-only evidence.
- Virtual labelled proposals:10,752 maximum, no replacement families; at most40,000 new policy-frame pairs overall, including dataset, evaluation and development rendering. Re-rendering solely for prettier videos is not required.
- Official training:8,192 updates +32 discarded canary +256 completed-update technical recovery allowance =8,480 hard cap; raw training-head cap210,000. Failed jobs before an update do not invent updates; record all actual activity.
- Primary virtual episodes:192; invalid no-result recovery allowance16, hard208. Do not count virtual steps as rigid-body physics.
- WBC:32 planned starts, hard36 including only no-valid-result technical attempts; each <=40 s task +1 s normal initialization; hard1,476 actual simulated seconds. Derive tick/substep/actor caps from the exact admitted runtime dt before any physical start. No WBC training.
- Queries: primary virtual <=19,200; secondary WBC <=2,400; new-domain offline768; legacy new880 after donor cache reuse. Planned total <=23,248; hard24,576 attempted new GR00T queries /98,304 four-step inference heads. Failed secondary lane does not release its budget for extra primary cases.

PC coordinates; Corelab handles CPU checks, renderer, virtual feedback and bounded WBC/no-gradient inference; Hopper handles actual GR00T canaries/training on lawful compatible GPUs. Main owner can fix paths, import order, source binding, serialisation, staging, output durability and exact recovery without repeated approval. Preserve valid cells/checkpoints and remaining counts. No changing scientific formulas, tasks, labels, thresholds, text, precision or seeds under the name of engineering.

Storage helper protects the L donor, both new arms, dataset/statistics, active buffers and exact recovery sources. It is not an admission gate. No broad cleanup, private-asset publication, new monitoring daemon/mail, unbounded background work, protected TEST/DEV-F access, real hardware, additional control representation or gain search, or merge.

## 10. Compact durable evidence and completion

Reuse mature raw-first sealing/CPU replay/HF tools. Preserve sufficient raw RGB/state/reference/output data, but package per-episode/per-shard arrays and indexed metadata rather than uploading hundreds of thousands of tiny independent files. Stream-verify fixed public archives and their internal members; do not re-read unchanged historical checkpoints. Do not create another general orchestration framework just to run this goal.

Deliver: finite teacher liveness comparison; actual virtual data with source/exclusion counts; four full Trainer checkpoints; all fixed offline and192 virtual episode outcomes; separate secondary WBC outcome/not-run table; independent-code-path CPU numerical replay (state honestly when the scorer is shared); fixed-HF receipts; videos explicitly separating virtual teacher, virtual learned policy, and actual WBC transfer.

Primary complete requires two actual4096 endpoints and their fixed virtual/legacy evaluations, not DATA_READY or a script-only stub. Quality PASS/FAIL is separate from completion. If real primary data integrity or teacher liveness fails, return PRIMARY_PARTIAL with exact evidence; do not solve it by lowering the admission standard. A WBC-only shortfall is SECONDARY_PARTIAL and does not erase completed primary learning.

Terminal: PI0267_EXECUTOR_DECOUPLED_RECOVERY_STUDY_COMPLETE — STOP_FOR_WIDER_SCENE_OR_SECOND_EMBODIMENT_ADMISSION.

Allow a narrowly scoped virtual_research_candidate only under section7. Keep formal selected_checkpoint=NONE and deployment_selected_checkpoint=NONE. An ideal and a lagged virtual executor are NOT two robot embodiments. No target-robot teleoperation is used in this new fine-tuning corpus, but the GR00T donor's earlier training history remains disclosed. The next scientific decision should be based on recovery transfer, perception shift and tracking decomposition—not another default template-training loop.

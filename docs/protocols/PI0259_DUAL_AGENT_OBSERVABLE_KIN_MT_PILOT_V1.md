# PI0259 — Two-agent handoff: finish media release, repair observable kinematic data, execute the multitask pilot

Issued 2026-09-29 in response to the user's request for separate next execution plans for the video agent and the blocked scientific agent.

Two independent role authorizations:

- **Agent V:** `PI0259_VIDEO_RELEASE_AND_INDEPENDENT_DATA_QA_GO_V1`.
- **Agent M:** `PI0259_OBSERVABLE_FK_CONSISTENT_MT_PILOT_GO_V1`.

This is an explicit prospective scientific amendment to the PI0258 data-generation program, not permission to relabel its failed samples. Agent M may implement the camera/reference/IK changes below, produce real synthetic data and, after the stated gates, run MT-Base and MT-G for 6000 updates each without another routine GO. Agent V closes PI0257-v2 as a media release and independently reviews sealed data and result packets; it does not become a second experiment launcher.

## 0. Sources, accepted status and limits of this review

Repository `Leo-6-maker/go2-arx5-hierarchical-vla`. Issue #11 is the scientific ledger. PR #77 remains draft/unmerged and receives pointers only.

Science source base: `8e1276da662ba2868e704fa96323807c5607efd1`, PI0258 terminal canonical `5890917092`. Read:

- `reports/PI0258/CLOSEOUT.md`, `generation_card.json`, `development_start_visibility.json`, `development_geometry_summary.json`, `development_render_summary.json` and `independent_cpu_replay.json`.
- Actual `scripts/pi0258_kinematic.py`, `pi0258_renderer.py`, `pi0258_geometry.py`, the development/export scripts referenced by their manifests, and the actual private FK/URDF bindings.
- Original PI0258 authority `8944d569d9f83baa74f0cb28c1113ba9659e528a`, canonical `5889346948`, especially its common processor, fixed initializers, real resume and offline comparison rules.
- PI0257 authority `1e760ba055106cb85b508711f1466d1cb78a9656`, canonical `5888937489`, and the actual v2 release/QA manifests reported by the video agent.

The immutable PI0258 CLOSEOUT read for this decision reports M1 start visibility 6/12, with geometry not run after that ceiling; M2 visibility 0/12, geometry not run; M3 start visibility 12/12 and geometry-qualified sequences 8/12. M3 family000 has 61.875% qualified H40 geometry windows and family002 32.5%, with missing phase coverage. The eight qualified M3 streams contain 1448 rendered image pairs and 1280 non-REJECT raw H40 windows. The fixed cohort still fails admission. These findings are neither zero successful geometry sequences nor a trained-model failure. If an execution-side summary differs, reconcile it against these immutable per-sequence artifacts before copying counts; do not silently change either historical record.

No TRAIN/KIN-VAL production, common new statistics, model canaries, Base/G6000 training or model evaluation has been admitted. Original U quality FAIL, R zero-update integrity BLOCKED, PI0255/56 failures, and selected/deployment NONE remain unchanged. No remaining allocation from those experiments is transferred.

The user reports PI0257-v2 complete, zero new inference/training/physics, archived at `leoyuer/pi0246-gate-pass-campaign@88a9dc144cd0bb3263279f952fcf2f6fa8a0b4dd`, prefix `evidence/PI0257/corelab-video-review-v2-20260929`. This authorization reviewed the report, not a newly downloaded v2 video package. Agent V must bind its actual release receipt; the PI's publication is not independent visual acceptance of every v2 frame.

Two concrete source observations inform the new design:

1. The existing cameras keep fixed attachment offsets and a fixed 135-degree local rotation. Actual start images fail to contain some command-relevant markers. Changing a video canvas or drawing an overlay does not put those cues into policy observations.
2. `pi0258_kinematic.reference()` starts from a nominal position and an identity quaternion, while the renderer obtains the arm pose from actual URDF/FK. Source correctness requires checking their actual equality; an identity quaternion is not a general substitute for measured or FK attitude. The solver warm-starts from prior timestamp solutions, including failed ones. These are specific contract/design issues to examine, not proof that they uniquely caused all failed sequences.

The new plan fixes the data question before spending model compute. It does not require old WBC settling, does not declare kinematic data physically executable, and does not authorize a WBC convention/gain/weight fix.

## 1. Responsibilities and isolation

Use separate worktrees and output roots:

- Agent V: `pi0259/video-data-qa-v1`; owns visualization adapters, portable QA, independent read-only checks and its release reports.
- Agent M: `pi0259/observable-kin-mt-execution-v1`; owns camera/data contracts, synthetic renderer, references, solver, dataset, processor, model training and inference.
- The publication branch contains this authority. Execution branches may be created from the actual required source commits, recording exact dependencies. V must bring its completed v2 implementation by immutable commit, not assume that it is present in the science branch.

Do not share a writable worktree or mutate a running producer. Exchange only sealed manifests and read-only arrays. No second scheduler, persistent watcher, background polling service, mail or automated monitor is required. An explicit handoff or a finite stage check is sufficient.

Only Agent M may submit a PI0259 Hopper model job. Only Agent M may invoke the isolated synthetic renderer. Agent V runs CPU validation, decoding and re-rendering of recorded arrays: zero model loads/queries, zero renderer scene construction, zero physics, zero WBC actor calls, zero optimizer/backward.

Agent V findings on correctness, missing provenance, leakage, visibility or inconsistent pose/label identity can block the affected dataset. Cosmetic layout issues do not block collection/training once machine-verifiable contracts pass. V does not choose examples, camera rigs, labels or checkpoints to improve scientific scores.

## 2. Agent V goal — freeze video v2, turn the workflow into independent data QA

### V0. Close the already delivered release; do not start an unnecessary v3

Bind the actual v2 source commit, output manifest, portable QA, Corelab-runtime QA and fixed HF receipt. Check that the PC index points to existing outputs, normal-speed/slow-motion are distinct, H40 versus native sampled metrics are labelled, and the physical entry is honestly `GUARD_STUB_ONLY`. Reuse completed full hash/readback/decode evidence when file identities are unchanged. Do not download and re-verify all old evidence simply to report a larger audit count.

Preserve v1 and v2. Fix only a new reproducible defect in an additive patch release. Otherwise publish `PI0257_V2_RELEASE_FROZEN` and close the video beautification task. A `D:/...` location is a PC path, not evidence of a path in another runtime. No new query, checkpoint download or inference is permitted to improve a movie.

### V1. Implement one small reusable evidence-review adapter

Extend existing v2 tooling to read the Agent M packet in section 8. Do not build a separate simulator or analytics platform. Produce:

- source-resolution body/wrist panels, with display scaling clearly labelled;
- a separate diagnostic segmentation/depth panel, never merged into policy RGB;
- cue visibility counts, frame/FK/camera identities and source/sequence/task/command IDs;
- independent FK versus requested reference, residual/velocity/collision/phase-coverage summaries;
- matched-command views at the same actual RGB/StateV1, with independent correct-command semantics;
- all failures and missing data visible, not just the best sequence.

Validate RGB/BGR, field names, units, full SE(3), actual camera matrix convention, source dimensions, timestamps and noise/model identity fields. Keep masks/depth/calibration counterfactuals out of the policy-input allowlist. The masks must be saved raw, unlike the previous pixel-count-only evidence; re-reading producer PASS booleans is not an independent check.

Use deterministic metadata/hash selection for representative cards, before model scores: include each admitted task, both commands, all four motion quarters and any observed failure class. At least one complete correct/opposite-command sequence per task goes into the review package. A review cohort is not an estimate of generalization success rate.

### V2. Three finite handoffs

**H0, data design review:** once Agent M seals camera/FK/solver calibration arrays, check the actual native-resolution frames and independently reconstruct cue counts/projections. Return `DESIGN_VISUAL_QA_PASS`, `CORRECTNESS_BLOCKED`, or a named partial result. Do not infer visibility from a 3D bounding box alone.

**H1, data-release review:** once all registered data attempts and data QA are sealed, independently recompute split grouping, pair identity, required-cue and phase coverage on saved arrays, and verify the stated qualifying counts. Tests must detect changed target ID, swapped command, dropped initial quarter, mixed native/world frame and target metadata entering the collator. Return `DATA_QA_PASS` only for the exact data/processor identities reviewed. Review source packets, not a lossy MP4 as a substitute for the inputs.

**H2, model-result review:** use only Agent M's already budgeted and sealed step0/final predictions and CPU receipts to render results. Show each arm's common-processor baseline and final, per-task and worst-task metrics, correct/opposite commands and paraphrases. No model calls by V. Return `OFFLINE_RESULT_VISUAL_QA_PASS` or named discrepancies, not a deployment decision.

H0/H1 require correctness closure, not a finished presentation video. Agent M can continue to independent CPU work while V reviews; neither overwrites the other. When no new packet is available, V ends with `WAITING_FOR_SEALED_HANDOFF`, not infinite polling, a false overall BLOCKED, or another cosmetic goal.

### V3. Acceptance

Deliver the frozen v2 release index, portable data-review command, CPU tests, actual H0/H1/H2 receipts when available, and source-labelled cards/videos. All synthetic material must say `KINEMATIC REFERENCE — NO WBC EXECUTION`; model panels say `OFFLINE MODEL PREDICTION`. Historical physical videos retain their original outcomes and timing caveats. No new dynamic success, best deployed model or high-level closed loop is claimed.

## 3. Agent M goal — explicit observation/reference amendment, not retrying PI0258

### M0. Reuse the block and finish the narrow source checks

Accept PI0258's failed cohort and do not rerun it under its old IDs. Read the actual source/config and the failed and successful M3 numeric streams. Separate:

- required marker outside image, marker occluded, marker too small/ambiguous;
- fixed IK residual/branch failure, joint-velocity failure, collision and missing window coverage;
- reference start versus actual nominal root/arm FK pose;
- nominal world coordinates versus immutable native anchor and row-packed model rotations.

Use existing source/arrays for diagnosis. Do not reopen old controller A1, apply the native WBC two-column/two-row shim change, restart R, or repeat model fitting to O64. The controller import-convention uncertainty remains in a separate physical-transfer gap list.

### M1. Explicit new reference identity: actual FK start and smooth global phase

Create `PI0259_KIN_REF_V2`. Fix the original nominal root/leg pose and original nominal arm joints; do not optimize the start pose per task. Compute the full initial world TCP pose `(p0,R0)` from actual URDF/root/DOF FK. Use that same state for the renderer, StateV1, immutable native anchor and the first reference pose.

Do not set a pose to identity merely to match a convention. Require exact physical equality within the inherited FK/round-trip tolerance, and record the numerical comparison against the old nominal/identity start. The old values remain preserved; this is a new reference version even when a particular equality happens to hold.

Retain the three behaviors and the existing task durations:

- M1: approach the language-selected target and match its VISIBLE frame. Target orientation is prospectively defined as `R0 @ Rz(yaw)` with the existing two yaw offsets (-0.2,+0.2 rad). The rendered axis markers use this exact world orientation. This is a changed orientation distribution, not evidence of arbitrary-world-orientation control. Do not silently turn it into position-only reaching.
- M2: retain the specified left/right route, forbidden obstacle region and required corridor order. Keep the constant attitude R0 throughout. Existing semantic position offsets and obstacle/collision geometry are not shrunk to rescue a sequence.
- M3: retain the 270-degree open arc and distinct black/white endpoints, requested direction, ordered traversal and correct end. Keep constant attitude R0. This does not test closed-lap counting.

Use one quintic progress law `s(u)=10u^3-15u^4+6u^5`, u clipped only to the declared segment time bounds, for M1's six-second motion, each existing M2 segment (3,3,4 seconds), M3's four-second approach and twelve-second arc traversal. Use SO(3) interpolation for M1. This deliberately supplies continuous zero endpoint velocity/acceleration at segment boundaries without lengthening task durations or changing inference dt. Record it as a new expert time law, never retime a failed model output. Terminal holds remain explicit audit-only windows when the original minimum-arc requirement rejects a constant future.

Derive all data windows from the same continuous reference and synthetic time. No per-window reanchoring, target injection, oracle clock/phase/source ID, hidden previous reference or outcome-selected interpolation is allowed. New statistics do not change the meaning of the raw pose.

### M2. One fixed deterministic IK-continuation program

Reuse existing FK, collision assets and numerical libraries, not a new simulator/controller. Implement the new solver as a versioned adapter with these fixed rules:

- start at original nominal joints; warm-start from the last ACCEPTED solution, never present a failed solver candidate as a valid continuation state;
- position residual and SO(3) log residual in a documented common frame; stable rotation-log/quaternion handling around identity and pi; quaternion sign continuity is representation handling, not an altered physical reference;
- independently finite-difference the ACTUAL residual function with step1e-6 and verify sign/frame against the geometric Jacobian. Solve the corresponding negative-residual least-squares step rather than assuming the signs coincide;
- original weight0.2 m/rad, damping1e-3, max joint step0.1 rad, max200 iterations per internal waypoint; line-search scales [1,.5,.25,.125,.0625,.03125,.015625,.0078125]; no random multistart or manual per-case joint seed;
- deterministic subdivision of each 0.05s reference interval, at midpoints from the SAME analytic reference, up to depth4 (at most16 internal subintervals). Subdivide if adjacent requested poses exceed5mm or0.02rad, or a direct continuation solve fails. Record every attempted internal solve, not just the accepted endpoint. This is a solver continuation strategy, not extra output duration or a change to H40;
- check accepted internal and output-grid samples for original hard joint/velocity limits, original indexed collisions and finite/continuous FK. If no valid continuation exists, retain a failed sequence with its last valid prefix and all residuals. Do not skip the failed interval, interpolate invalid joints, or replace the sequence.

Final geometry tolerances remain XYZ<=2mm and SO3<=0.01rad. The existing independently verified collision checker and source exemptions remain unchanged. No obstacle removal, new collision exemptions, safety-threshold changes or automatic joint clipping of model predictions. Solver projection of its own candidates to original hard joint bounds is allowed and recorded; it is not deployment postprocessing.

Before new sequence generation run CPU fixtures for analytic FK round-trip, true default-start equality, nonzero root rotation/translation, near-pi rotation branch, residual-Jacobian sign, failed-candidate propagation, quintic joins, velocity limits and collision negatives. This tests one algorithm, not whether an untested pose is globally reachable.

### M3. Bounded observation-rig calibration, then one frozen common rig

New identity `PI0259_KIN_OBS_V2`. This is explicitly a changed synthetic observation distribution. Both Base/G receive it; the legacy physical camera configuration stays untouched. Preserve two 96x96 RGB streams and StateV1/H40 dimensions. The wrist attachment remains original. The first/body channel uses a fixed base-mounted look-at workspace view, not a task-dependent tracking camera or a third model input.

Use only these three body-mount position candidates in base coordinates, in this fixed order:

1. [1.15,-1.15,0.95] (old mount position, newly correct look-at orientation);
2. [0.80,0.80,0.80];
3. [0.40,0.00,1.15].

For each, orient its optical axis toward the centroid of the prespecified non-TEST workspace/reference/marker envelope. Fix horizontal FOV70 degrees, use the source renderer's verified camera-axis/up convention and record the complete extrinsic/projection matrices. The orientation is resolved once for the common support, not per family, task, trajectory sample or model prediction. Convert the base-mounted transform through the fixed root as required. No zoom based on a desired future.

This is bounded, disclosed DEV-only sensor design, not three model arms. Agent M may evaluate at most these three rigs on six quarantined calibration families and fixed metadata-selected start/quarter/branch/end poses, up to512 rendered image pairs total. Save original-resolution RGB, raw segmentation/depth, current joints/FK and camera matrices. The old PI0258 six development families may be used here because they are already development-exposed; they may not enter new TRAIN/VAL/TEST.

Select the FIRST candidate in the fixed order that meets the following on all calibration states relevant to each candidate task; do not select the rig by model loss or later TRAIN/VAL outcomes:

- selected target/goal/arc endpoint markers each occupy>=4 actual pixels in at least one stream; M1's two necessary visible axis markers each>=2 pixels, with distinct centroids and correct colors;
- M2 obstacle occupies>=16 pixels in at least one stream and the orange goal is visible; side semantics use the frozen world convention and actual camera matrices, not an implicit image-left substitution;
- M3 endpoints and direction-defining visible contour are present; at least8 of the12 ordered sectors have actual contour pixels at the common start, plus endpoint visibility at the task's discriminative states;
- robot/link placement, RGB/current FK and sensor calibration agree; show natural RGB to a reviewer because pixel-count presence alone is only a necessary condition for discernibility.

One global camera must support at least two distinct behavior families. Evaluate every candidate task, include ALL that later meet full data admission, and disclose any unsupported third family. No per-task camera selection. If more than one rig qualifies, use the lowest index, not the prettiest video. If none qualify, record `OBSERVATION_DESIGN_BLOCKED`; do not add a fourth rig, move targets, enlarge markers or paste cues under engineering permission.

Save masks/depth explicitly. Optional diagnostic marker-hide counterfactual renders are allowed inside the512-pair calibration budget and never become policy RGB. An analytic bounding box or a producer count without its raster evidence is insufficient. Agent V independently verifies the sealed H0 packet before production; a correct data contract is the gate, not cosmetic video readiness.

## 4. Calibration, fixed new data population and qualification

Only one prospective reference/solver version in M1/M2 is tested. Up to36 old-development-condition kinematic calibration sequences are allowed under NEW PI0259 IDs, with all failures retained. They are development engineering/scientific calibration, not additions to old success counts or data for training. Freeze all numerical camera/reference/solver settings before the fresh data program. A source-proven implementation defect may be corrected prospectively within this defined algorithm; preserve its version and affected attempts. No new pose/time-law/parameter candidate beyond this document.

Create a fresh geometry pool in native world axes around the source nominal scene center with offsets:

- x,y each in [-0.04,-0.02,0,0.02,0.04] m;
- z in [0,0.015,0.03] m;
- contour radius in [0.035,0.045,0.055] m.

This gives225 geometry tuples before removing previously development-exposed duplicates. Keep existing within-scene target/obstacle/route offsets and physical sizes, subject to the same solver/collision checks; do not resize a particular failed family. Explicitly changing this support grid is part of the new data experiment, not comparable seed replay of PI0258.

Group by complete geometric scene tuple, fixed root/DOF, camera identity and asset identity BEFORE adding colors/language. Assign all color-only aliases, all tasks, both commands and all windows of a geometry family to the same split. Merely changing a color or ID cannot create independent held-out geometry. Exclude old inspected calibration geometries from new validation/test. Record overlap with old data; held-out claims are about this new training and declared domain, not all historic development knowledge.

Use deterministic SHA ordering namespace `PI0259_GEOMETRY_SPLIT:25901`: six fresh DEVELOPMENT families, forty TRAIN, ten KIN-VAL, and twenty composition-TEST families from the remaining pool. Verify that individual factors in composition-TEST occur in TRAIN but complete tuples do not; a fixed seeded assignment solver may enforce these combinatorial constraints before any new renders. Separately seal twenty geometry-TEST metadata families using the prior explicit out-of-range design, without rendering/IK/model queries or statistics. No data generation from TEST is authorized.

Assign colors and four existing instruction templates with seed25902, balanced within the chosen population; training/development use templates0/1 and KIN-VAL uses2/3. Opposite commands share the same actual image/state at the same start/prefix. Same instructions and identical allowed observations must not require incompatible futures; investigate structural aliasing without adding privileged time/history.

Fresh development: three tasks x six families x two commands =36 fixed sequences. For each task >=10/12 usable development sequences releases its fixed80 TRAIN and20 KIN-VAL sequences. Evaluate the registered set, not only until enough successes have accumulated. Failed geometry, missing visibility, invalid reference or terminal-phase failure stays in the denominator. Do not substitute another family to reach64 successes.

For usable sequences require >=95% qualified active-generation windows and at least one qualified window in every motion quartile, plus correct semantic/reference construction. In the common initial and predeclared discriminative branch states the M3 visibility rules above are mandatory. Required cues must also remain available in>=95% of sampled active windows under the combined two-camera rule; record longer missing intervals and any structural ambiguity. This is a NEW observation-support criterion, not retrospective scoring of PI0258.

A data-qualified task requires >=64 usable TRAIN and>=16 KIN-VAL sequences and>=24/6 complete command pairs. Include every task satisfying the common-rig/data program. K=3 gives KIN-MT3; K=2 gives explicitly named KIN-MT2; K<2 blocks training. No two colors of the same reach task as an MT2 fallback.

Preserve successful and failed full references, requested/FK poses, IK solve ledger, joints, velocity, collision, RGB/masks/depth, complete transforms and phase flags. Synthetic state is `KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT`, time is `synthetic_reference_time`. Use the existing proven zero-integration link renderer: ZERO new physics steps/WBC actor calls under this goal, including constructors. No fixed-base kinematic sequence is called balanced, dynamically executed or controller-compatible merely because its gate accepts it.

## 5. Actual processor/data freeze and automatic model admission

Agent M owns dataset generation; Agent V validates the immutable H1 package independently. Use machine-readable correctness checks; do not block training for video styling. The actual data package and sampler are frozen only after fixed attempts finish.

Reuse PI0258's intended processor conventions under NEW identity `PI0259_C_KIN_MT_V2`:

- original relative-action conversion, row-packed GR00T rotations, slot10 mapping, H40/.05s,132D padding and active masks, neutral unexecuted gripper0.5;
- TRAIN-only q01/q99 affine statistics at the actual original normalization stage, after relative conversion, from provenance/geometry-qualified active TRAIN windows BEFORE codec filtering; one calculation, no VAL/TEST inputs, no iterative excluding-hard-samples/refit loop;
- original affine convention with1e-6 range floor, NO percentile clamp; deterministic full-view resize/tokenization, no random crop/flip/color jitter;
- raw physical label->original gate, actual processor->actual inverse decode->same gate, float32 round-trip position<=1e-5m and SO3<=1e-4rad; timestamps/masks/conventions exact;
- identical processor binding in physical training losses and inference decoder, not just config filenames;
- actual input allowlist: body_rgb,wrist_rgb,StateV1,language only. Reject target pose/ID, task/source ID, phase/time, future labels, mask/depth diagnostics or success values as policy inputs.

A label stored as a native state must match the recorded full world/native transform and actual FK; never feed the earlier raw-world `state_v1` preflight field directly. Keep desired versus current FK pose distinct.

Freeze6000 actual source/visit rows per arm using local seed25903: balanced task blocks, alternate command within task, uniform complete-pair TRAIN family, then source sequence, motion quartile and eligible window. Same exact schedule and processed tensors for Base/G. Record actual native tau/epsilon/S4 noise by source/visit keys; verify Python/NumPy/Torch CPU/CUDA/worker effects across independent processes and iterator restore.

Passing >=2-task data/processor/H1 correctness and resource gates automatically releases the model canaries and training below. No old settling/G-sim/O64-PASS prerequisite and no further routine PI GO. A new unrelated WBC capability question is not this synthetic-data gate.

## 6. Model experiment: exactly two matched6000-update endpoints

Fixed initializers:

- MT-Base: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- MT-G: `leoyuer/pi0246-gate-pass-campaign@efecea485ae77000599d19171b02491f8d514f42`, prefix `arms/PI0252/r1-speed-margin-20260928/G/checkpoint-2000`.
- Existing Cosmos tokenizer/backbone metadata lineage `9ce19a195e423419c349abfc86fd07178b230561`.

Resolve the original full actual weight/header/processor receipts; preserve slot10 original versus adapted tensors and the one documented tied-key reconciliation. No latest-model, U, R or FULL substitution. No arbitrary missing keys. New observations/normalization can alter step0 behavior; step0 must therefore be measured for BOTH models under the new common contract.

Real recovery test for each initializer: continuous8 updates versus4 + full-Trainer save + fresh-process restore +4 on the same first8 rows and the6000-step learning-rate schedule. These are32 discarded canary updates in total. Bind actual weights/optimizer/scheduler/RNG/cursor, all four S4 state gradients and unchanged frozen tensors. Canaries do not become research initializers; both main runs start fresh from the pinned original checkpoint.

Train6000 canonical optimizer updates per arm. For three tasks each gets2000 updates; for two each gets3000. Use one shared model per arm, not per-task checkpoints. Same original G loss: native flow matching +0.4 physical position +0.1 increment +0.1 rotation +0.1 peak-speed margin, full four-step S4 with proper backward recomputation. Auxiliary speed margin remains1.8m/s, not a new dynamic safety guarantee.

Fresh AdamW LR1e-5, betas(.9,.999), eps1e-8, weight_decay1e-5, gradient norm1, batch1, accumulation1, warmup10 and cosine6000; cursor0 accounting explicit. Existing BF16 frozen backbone and FP32 action-head/trainable projector/diffusion scopes remain; no LLM/visual/vlln unfreezing or new model adapter. No loss/seed/rate sweep.

Preserve complete Trainer at1000,3000,6000 with rolling verified durable storage and an actual recoverable working copy. Only step0/final6000 are quality-screened. No intermediate checkpoint rescue, endpoint extension or task-specific winner. G's prior adaptation cost remains extra; equal new updates is not equal total historical compute.

An independent arm can finish if the other is integrity-blocked; report partial results, not an unearned paired causal result. Quality FAIL is distinct from integrity BLOCKED. No fresh third arm or old R restart.

## 7. Fixed offline evaluation and interpretation

Retain PI0258's bounded evaluation structure under the NEW observation/data identity. Before the first model quality query freeze full source IDs and two actual noise tensors per input; no score-based selection.

For each admitted task:32 TRAIN windows from distinct sequences,16 KIN-VAL windows,12 semantic inputs from six paired KIN-VAL shared-prefix scenes,12 meaning-equivalent reserved paraphrases,6 empty-language inputs. Also64 old O64 regression inputs. Four model states (two step0 and two final6000) x two noises gives8*(78K+64):2384 queries for K3,1760 for K2. Each actual source/noise tuple is evaluated once. H40 shared prefixes without discriminative future are reported NONDISCRIMINATIVE; expert geometry determines the earliest separable state before model inspection.

Independently recompute raw decode, same-frame XYZ/SO3, first jumps, speed, original gate issues and fixed IK/collision feasibility without repairing predictions. Report absolute metrics on all rows, by task/noise, macro and worst-task, versus the corresponding step0. Correct/opposite-command trajectory margins and requested side/direction/target checks are separate from merely different arrays. Empty-language and paraphrase checks are limited evidence, not proof of universal language understanding.

Keep original numeric quality criteria visible (medianXYZ20mm, p9535mm, medianSO3.05rad, zeroREJECT). Reuse the PI0258 synthetic research label only when every admitted KIN-VAL task/noise passes those criteria and>=10/12 semantic directions and equivalent-paraphrase cases are correct with unchanged denominators. Missing12 discriminative inputs yields INCOMPLETE, not a smaller passing denominator. No deployment selection results from this label.

Old O64 under a new processor/observation domain is a regression panel relative to each new step0, not recreation of historical G/C_T qualification. KIN-VAL is held out from this training but remains synthetic development; composition/geometry TEST stays untouched. No new native WBC policy rollout or physical transfer is authorized.

Agent M provides H2 prediction/CPU evidence to Agent V. V only renders those outputs. Videos explicitly distinguish requested/FK kinematic motion from model predictions and from historical actual physics. The background prescribed robot is not motion caused by the displayed model.

## 8. Minimal handoff schema and source ownership

Each H0/H1/H2 packet contains a JSON manifest with: schema/version, producer role, immutable code commit, protocol ID, data/processor/camera/reference/IK hashes, cohort/split and full source IDs, array paths/size/SHA, state-origin and evidence class, counters, missing fields and a complete outcome index.

For H0/H1 include saved RGB, segmentation/depth, renderer cue-ID map, complete camera matrices, root/DOFs and independent FK, immutable world/native/TCP transforms, desired poses/timestamps, solver residuals/internal steps, velocity/collision flags, phase coverage and actual raw/codec gate records. Include all available rows needed to reconstruct counts. The reviewer is not required to guess a private drive path from a URL.

For H2 add model/initializer and complete checkpoint identities, actual raw noise, model outputs, decoded references, inference latency, per-cell metrics and first-valid/reuse status. No ground-truth future is passed to the model; its use in supervision and independent scoring is separately logged.

Agent V writes only a companion receipt containing its own code hash, accepted producer manifest hash, recomputed results, reviewed example IDs and PASS/BLOCKED findings. Corrections to producer data are made only by M, with a new immutable version and affected-scope re-review. Do not overwrite a sealed PASS packet in place.

A packet is delivered once at its stage boundary; use existing file transfer and GitHub pointers. No generalized service, live mutable shared directory or endless file audit is needed.

## 9. Hard bounds, resources and permission limits

NEW PI0259 allocations; old unused budgets are not rolled over.

| Activity | Maximum |
| --- | ---: |
| Agent V model/renderer/physics/WBC/optimizer calls |0|
| Agent M calibration kinematic sequences on old exposed conditions |36|
| Camera/visibility calibration image pairs |512|
| Fresh development sequences |36|
| Conditional fixed TRAIN+KIN-VAL sequences |300|
| Strict no-valid-result sequence technical reserve |24|
| All new sequence attempts |396;372 planned|
| Active H40 windows / rendered image pairs outside calibration |60000 /120000|
| New dynamics integration, native episodes, WBC actor calls |0|
| Main optimizer updates |12000;6000 per arm|
| Discarded model resume-canary updates |32 planned|
| Extra engineering/recovery completed-update reserve |64|
| Total completed optimizer calls |12096|
| Raw training action-head calls |109120, including256 partial/no-update calls beyond9*12096|
| Planned offline queries atK3 |2384|
| Strict no-valid-result query reserve |64|
| Offline query attempts / S4 inference heads |2448 /9792|
| TEST/oldDEV-F/hardware/learned-policy physics |0|

Calibration is development, not success-counting on old protocols. A valid scientific data failure is never retried under the same identity for a better score. A demonstrated implementation correction within the specified algorithm can be versioned prospectively, with every started attempt and compute counted; it cannot invent another camera/pose/reference/solver parameter or refresh an exhausted scientific budget. Pure import/IO/code corrections have no arbitrary two-submission ceiling. No duplicate active PBS jobs, loss of exact resume, or uncounted recomputation.

Corelab: M's isolated zero-step rendering/CPU geometry and later no-gradient inference; V's CPU review/rendering. Coordinate actual GPU UUID/free VRAM and node-local disk before use; no competing native process or killing unrelated workloads. Hopper: only M's admitted canaries/training on a compatible allocated GPU; legal auto scheduling is acceptable after runtime/VRAM checks. No CPU debug job on H200, no requirement to wait for H200 when another tested allocation suffices. V never submits Hopper jobs.

Use compact sharded arrays on verified task-local storage, not synchronous per-step NFS file trees. Preserve historical G, active full Trainer and unique evidence. Reuse fixed HF public archives and already verified identities; do not rerun broad cleanup or all-history hash scans. Check disk free, quota separately, and next-write high-water plus10GiB reserve. UNKNOWN quota is neither PASS nor proof of no capacity. A GPU queue is a resource wait, not a scientific failure.

Only project-owned data/media/allowed derived weights may be published to the authorized HF repository under unique PI0259 prefixes. Preserve licences and private WBC/mesh/tokenizer restrictions. Fixed-revision full-content checks must distinguish anonymous from authenticated reads. No public server, secrets in Git, email, monitoring automation or PR77 merge.

## 10. Terminal states and immediate next deliveries

Agent V:

- `PI0257_V2_RELEASE_FROZEN` after its existing receipt binding;
- `DATA_VISUAL_QA_TOOL_READY` with real CPU negative tests;
- actual H0/H1/H2 result per received packet;
- `WAITING_FOR_SEALED_HANDOFF` is a normal finite handoff state, not a reason to fabricate model/video progress.

Agent M:

- `OBSERVATION_DESIGN_BLOCKED`, `KIN_REFERENCE_OR_GEOMETRY_BLOCKED`, `KIN_DATA_K_LT_2`, or `MODEL_INPUT_OR_RESUME_BLOCKED` when the relevant new gates fail;
- `KIN_MT2_PILOT_COMPLETE` / `KIN_MT3_PILOT_COMPLETE` only when the admitted data, two6000 endpoints and fixed full screen actually exist;
- an unfinished arm/screen is `PARTIAL_MT_PILOT`, not COMPLETE;
- all completed pilots stop at `STOP_FOR_PHYSICAL_TRANSFER_ADMISSION`, with model quality stated separately.

Agent V's release completion is not Agent M's experiment completion. Agent M's source-preparation PASS is not data readiness or training progress. Both preserve selected_checkpoint=NONE and deployment_selected_checkpoint=NONE.

The immediate deliverable from V is a frozen v2 release plus an independent source-resolution cue/FK review tool. The immediate deliverable from M is a small actual H0 packet demonstrating the new common rig and FK-consistent reference/solver on both passing and failing calibration cases. Do not spend weeks on media polish or restart old single-task tuning; after the new data gates pass, execute the already authorized two-arm multitask pilot.

## 11. Role-specific startup instructions

**Agent V:** read this full protocol and the PI0259 canonical. Bind the actual PI0257-v2 release, reuse existing QA and close it without a new cosmetic version. Implement the small read-only H0/H1/H2 adapter and CPU negatives. Review M's sealed packets, not its live worktree; source-resolution cue visibility and physical frame identities are correctness, not presentation. Never load/query models, run the synthetic scene generator or advance physics. Produce readable cards/video from existing evidence with honest synthetic/offline labels. If no new packet exists, end at WAITING_FOR_SEALED_HANDOFF. Keep run-episode GUARD_STUB_ONLY.

**Agent M:** read this full protocol, PI0258 canonical5890917092 and actual8e1276d source. Preserve old failure; implement the prospective common-camera calibration, exact FK initial reference/orientation and smooth time law, deterministic bounded IK continuation. These changes are explicitly authorized here; do not request another GO for the same old observability/fixed-solver restriction. Produce H0 for V, freeze one data design, then execute the fresh fixed development/TRAIN/KIN-VAL program, actual common processor and H1. At least two qualified tasks and real integrity gates release Base/G recovery canaries and6000 updates EACH under this GO. Run only step0/final fixed offline/semantic evaluation; pass sealed H2 to V. No old settling prerequisite, no R restart, no controller rewrite or learned-policy physics. Report actual data/model evidence and partials, not another manifest-only completion.

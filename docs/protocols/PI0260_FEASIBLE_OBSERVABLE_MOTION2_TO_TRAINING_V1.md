# PI0260 — Feasible, observable Motion2 data to matched Base/G training

Status: AUTHORIZED_NOT_EXECUTED at publication.

## 0. Authority, objective and precedence

User requested a careful review of Agent V's final H0 audit and authorization to continue towards training. This is a NEW prospective data-design and training authorization, not a reclassification of PI0259 failures.

- Agent M: `PI0260_FEASIBLE_OBSERVABLE_MOTION2_TRAINING_GO_V1`.
- Agent V: `PI0260_INDEPENDENT_MOTION2_DATA_AND_RESULT_QA_GO_V1`.
- Canonical scientific ledger: Issue #11. PR #77 remains draft/unmerged and receives pointers only.
- Reviewed producer: `74aa7bcb75e2ca37dc6c50bb1a0c7f15f9c7d7f0`, comment5892641889.
- Reviewed independent auditor: `653d280dacdd3588f937281a4f59997bebbaec5a`, comment5892796587.
- Parent protocol: `240dedfd3954dd4cbbb48645a2aad5b19738e2c4`, `docs/protocols/PI0259_DUAL_AGENT_OBSERVABLE_KIN_MT_PILOT_V1.md`.

The deliverable is a real two-behavior synthetic dataset, two matched 6000-update model endpoints and their fixed offline/semantic evaluation. Correct new data, actual input/recovery integrity and resources release training automatically under this authority. Old WBC settling, old single-task perfect fit, M2 repair and cosmetic video completion are NOT prerequisites.

This protocol supersedes the parent ONLY for the new active task set, visual marker/camera design, upstream scene-eligibility procedure, splits/keys and budgets specified below. It restates model/data invariants and evaluation requirements to avoid an ambiguous implicit inheritance. Parent failures and old identities remain unchanged. No old unused budget is silently added to the new budget.

## 1. What the source evidence actually establishes

H0 manifest SHA256 `94954516f9b1e9159e7fe468c04d7155e6cc20eb4d21c601b8389e8ab2dcac62`; V receipt SHA256 `1e16c79ad6d7c1f98318cf8d151378b27b050dcaeb2068c3874b815c8fabcde7`.

1. 144 original image pairs were audited numerically; six representative natural-image cards were visually inspected. This is not 144 independent episodes or exhaustive human inspection.
2. Rig1 M1 has40/48 saved later poses,9 visible-cue failures and8 geometry-missing poses. Rig1 M2 has36/48 saved later poses, no observed cue failures but12 geometry-missing poses. Rig1 M3 has32/48 saved later poses,14 black-endpoint visibility failures and16 geometry-missing poses. Missing geometry is NOT a zero-pixel image, and passing the available images does NOT prove full coverage.
3. Geometry calibration was M1 10/12, M2 0/12, M3 8/12. M2's fixed route reaches actual x5_link7/8–grey-obstacle collision in its final segment. M1 family002 and M3 families000/002 fail the bounded residual continuation. No task currently has an admitted fresh training dataset.
4. Actual default FK was p0=[.2901,0,.5966057], R0=I; the old nominal/identity start matches it. The earlier default-start-mismatch hypothesis was NOT observed. Do not cite it as the established cause.
5. The corrected contour-sector check applies eight visible sectors at common starts, not every later pose. Correcting that report did not eliminate black-endpoint failures. The post-save renderer destructor failure was separately disclosed; do not rerun already-valid arrays because of that exit.
6. PI0257-v2 is frozen and V's review tool is delivered. Generic H1/H2 views are not yet a source-specific independent admission implementation; completing their narrow adapters is authorized below.
7. No new Base/G training, model-query evidence or model generalization failure exists in this H0. `OBSERVATION_DESIGN_BLOCKED` and `CORRECTNESS_BLOCKED` remain the correct OLD results.

Publication reviewed source/report evidence, not the local Windows ZIP or the raw 144 rasters anew. Agents must use the sealed originals for execution checks.

## 2. Scientific change: a feasible Motion2 starter, not an easier retrospective score

For this first shared model experiment, the task set is FIXED to:

- M1: language-selected target with the target's displayed pose, with original target yaw offsets relative to actual R0.
- M3-v2: approach the specified endpoint and traverse the visible 270-degree OPEN contour in the specified direction to the other endpoint.

M2 is `DEFERRED_FULL_LINK_COLLISION`, NOT PASS and not relabelled as M1. No M2 planner experiment or training occurs in this goal. Its original grey obstacle and orange goal remain in the common scene as distractors/geometry, identically for M1/M3. Their existing collision treatment remains. Do not delete/shrink the obstacle or hand collision geometry to evade the reported contact.

M1 and M3 are distinct non-contact motion behaviors, not evidence of grasping/pushing or general-purpose manipulation. This reduced task scope is explicit and decided BEFORE model results. No MT1 fallback; both behaviors must supply qualified data.

Unlike the parent program's fixed-attempt success-rate pilot, this program expressly defines a CONDITIONAL scene population before learning: scenes for which BOTH commands of BOTH behaviors have complete, visible, geometrically valid references. A bounded, deterministic upstream eligibility filter may select scenes from the declared proposal lists. Rejected scene proposals and all reasons remain in a separate denominator. A curated accepted dataset is NOT reported as 100% success on the unfiltered scene distribution.

This is permitted data construction before model testing; it is NOT permission to reroll a learned-policy failure, replace a fixed model-evaluation case, omit a difficult valid score or alter the old PI0259 results. The output claim is limited to the declared feasible-and-observable synthetic domain.

## 3. Freeze one new observation design; preserve reference/IK physics semantics

### 3.1 Fixed prospective cue pack

Use actual in-scene 3D visual assets, rendered into both raw RGB streams. No pasted overlay, x-ray visibility, segmentation-as-RGB or instruction-dependent highlighting.

Set one common pack `PI0260_CUE_V1`:

- target sphere radius .018 m, unchanged;
- target axes length .060 m; axis endpoint sphere radius .010 m;
- black/white arc endpoint sphere radius .018 m;
- yellow contour point radius .006 m, original24 point locations/order;
- all cue centers, labels, colors and target orientations follow the original scene/reference geometry; no relocating the endpoint independently of the task;
- grey obstacle, robot/hand meshes, collision shapes, collision exemptions and physical dimensions are UNCHANGED;
- visual cue actors retain the source's visual-marker role. Do not add a new collision exemption to any robot or real obstacle.

These size changes are explicit new visual task assets, not an image-quality correction of the old rasters. Larger spheres may still be occluded; acceptance must be measured. No second marker pack or per-case resizing is authorized.

### 3.2 Camera calibration

Both policy streams remain actual96x96. Wrist attachment, convention and FOV remain the source values. Body camera uses exactly this fixed ordered candidate list of BASE-frame positions:

1. [0.65,0.00,0.75]
2. [0.65,0.45,0.85]
3. [0.65,-0.45,0.85]

Horizontal FOV70 degrees. Resolve a common look-at target once from the declared non-TEST scene envelope using the actual root transform and source camera convention. Record numerical matrices. No per-task/family/trajectory tracking or model-dependent zoom. This is a synthetic sensor design; do not claim that the physical body camera has been changed or validated.

Reuse the OLD calibration FK joint arrays, including all available valid prefixes, rather than resolving the same IK again. Re-render only because the NEW assets/camera identity differs. Common states can be shared across commands; preserve repeated-render determinism tests. Total calibration <=512 image pairs.

For each rig:

- All six original common-start states must support both M1 commands and both M3 endpoint/direction requirements.
- Check the original registered later times for every available valid M1/M3 calibration pose, not a favorable subset. Keep missing-geometry rows explicit. At least four complete paired geometry families must contribute later-pose coverage for both behaviors; the reported calibration should provide these, but verify the actual IDs.
- Retain targets/endpoints >=4 actual pixels in at least one stream, M1 required x/y axis cues >=2 pixels with correct colors and distinct centroids; M3 >=8/12 contour sectors at common start and both endpoint cues at discriminative states. No threshold relaxation.
- V inspects actual RGB, masks/depth, full transforms, occlusion/failure records; pixel presence is necessary, not sufficient to prove the semantic cue is distinguishable.

Select the FIRST rig meeting these calibration requirements. No fourth rig or implicit resolution increase. Missing invalid geometry is neither camera PASS nor camera FAIL; it is excluded only from camera measurement and MUST be rejected by the fresh scene-eligibility filter below.

Freeze the chosen rig and cue pack BEFORE examining fresh DEVELOPMENT, TRAIN or KIN-VAL outcomes. If none qualify, deliver specific obstruction/projection evidence and stop this design branch, not a speculative training run.

### 3.3 Reference and solver remain fixed

Reuse `pi0259_kinematic.reference` and its verified source semantics: initial root/nominal joints and actual world FK p0/R0; M1 target pose R0@Rz(original +/-0.2), M3 constant R0; original quintic timing, motion and terminal-hold durations; H40/.05s and native/world/TCP conventions. Do NOT contract workspace, shorten the arc, move a failed goal or optimize the initial joints under this authorization.

Retain actual-residual centered-Jacobian DLS, weighted rotation.2,200 iterations, damping1e-3, jointstep cap.1, original fixed line search, last-ACCEPTED continuation depth4 and all hard limits, per-joint velocity and collision checks. Original2mm/.01rad geometric tolerances remain. No random restart, per-sequence solver tuning or alternative IK branch search.

The new intervention on geometry is upstream conditional selection, NOT another unbounded solver development campaign. A demonstrated coding defect within these definitions may be corrected prospectively with versioned evidence; changing the defined algorithm requires new PI authority.

## 4. New data population and explicit denominators

### 4.1 Proposal registry before outcomes

Use the parent225-tuple factor grid: x/y shifts[-.04,-.02,0,.02,.04], z[0,.015,.03], radius[.035,.045,.055]. Keep original within-scene offsets/obstacle sizes. Exclude every previously inspected calibration/development geometry and every reserved old TEST geometry from the new proposal population by actual numerical geometry, not just ID. Reading split/geometry metadata to prevent overlap is allowed; no TEST IK/render/model call.

Sort remaining tuples by SHA256 of canonical JSON under `PI0260_GEOMETRY_PROPOSALS:26001`. Freeze the whole ordered registry and exclusion ledger before fresh data execution. Require at least160 remaining tuples; if fewer, report the exact shortage rather than duplicate or expand the grid.

Assign positions BEFORE feasibility checks:

- first20 -> DEVELOPMENT proposal stratum;
- next100 -> TRAIN proposal stratum;
- next40 -> KIN-VAL proposal stratum;
- all others -> UNUSED_RESERVED, not executed.

Color aliases, both commands, both tasks and all windows of a geometric family remain in the same stratum. No transferring a rejected VAL scene into TRAIN or borrowing extra candidates from another stratum to fill a quota. New namespace does not erase prior exposure; attach the overlap report.

Within each stratum use the fixed order. Select the first6 eligible DEVELOPMENT, first40 eligible TRAIN and first10 eligible KIN-VAL families. Stop drawing from that stratum when its quota is filled; remaining IDs are `NOT_EVALUATED_AFTER_FIXED_QUOTA`, not failures or successes. If the proposal list is exhausted first, report `ELIGIBLE_SCENE_SHORTFALL` with exact counts; do not change the design or quota.

### 4.2 Eligibility is joint and pre-model

A family is eligible only if ALL FOUR complete requested sequences (M1 c0/c1 and M3 c0/c1) satisfy:

1. Complete finite FK/reference sequence through the original motion endpoint, valid hard joints/velocities and original full-link collision checks, no failed-IK suffix replaced with interpolation/zeros/NaNs represented as usable motion.
2. Original semantic/reference construction correct: M1 correct target AND pose; M3 correct ordered270-degree arc AND direction. Preserve desired versus FK and exact time grids.
3. Current image/state/FK transforms agree. Start and registered discriminative frames satisfy necessary cues; required cues remain available in at least95% of active-window observations and in every motion quartile. Full real RGB/mask/depth evidence is retained. Missing cue intervals and sequence failures stay explicit.
4. Raw physical label -> same provider/gate at the actual FK/state/time is non-REJECT in at least95% of active-generation windows, with valid coverage in all quartiles and the required starts/discriminative states. Terminal-only holds remain audit records, not a manufactured source of motion windows. No source scores or model predictions participate in eligibility.
5. Opposite commands have genuinely identical observations/state at their common start/prefix. The task and command are expressed only by permitted language; no command-specific actor visibility or layout. Same language/observation must not silently require incompatible futures.

Perform cheap complete geometry and raw-gate checks before expensive dense rendering. Geometry-invalid proposals need not render nonexistent states; record why later checks are NOT_RUN. Do not use partial valid prefixes from ineligible families as training data. Reuse same raw frame across matching commands where actual state is identical, not merely similar.

All proposal attempts, outcomes and first-failure categories are archived. Report `eligible families / evaluated proposals` by stratum and individual behavior, plus selected-family counts and true usable-window counts. Selection feasibility is NOT WBC task success. Exact solver calls and rendered frames are counted separately from proposed sequences.

### 4.3 Development then fixed dataset

H0 camera/source review is done once for the selected design. The first6 eligible DEVELOPMENT families provide a fresh12-sequence-per-behavior confirmation of the conditional domain. Because acceptance was conditioned on known feasibility, report this as confirmation/QA, NOT an unbiased12/12 success estimate. No further camera/cue/solver changes after this point.

After DEVELOPMENT and V correctness closure, execute the frozen TRAIN and KIN-VAL lists. Joint selection yields80 requested TRAIN sequences and20 KIN-VAL sequences PER behavior. Do not count these as80 or20 independent scenes: there are40 and10 paired families.

H1 carries dataset freeze and all rejected-proposal indices. Data may feed learning only after the actual codec checks below leave at least64 usable TRAIN and16 usable KIN-VAL sequences per behavior,24/6 complete command pairs and all motion quartiles. No post-codec refill, reselecting a better family or iterative statistics refitting. Both M1 and M3 are required, no single-behavior training fallback.

## 5. Actual processor and independent training admission

New shared identity `PI0260_C_KIN_MOTION2_V1` for both models:

- original relative conversion, row-packed rotations, native/world/TCP binding,10 valid dimensions in132,H40,.05s; neutral gripper.5 is unexecuted;
- original q01/q99 affine convention with1e-6 range floor, NO percentile clamp; fit ONCE on provenance/geometry/raw-gate-qualified active windows of the selected TRAIN families before codec filtering, never VAL/DEV/TEST;
- deterministic full-view processing, no random crop, horizontal flip or color jitter;
- raw physical label->actual processor->actual inverse decode->same provider/gate; float32 position roundtrip<=1e-5m, rotation<=1e-4rad; finite and exact masks/time/frame conventions;
- actual physical loss uses the SAME statistics/decoder identity as inference. No metadata-only switch;
- policy inputs only body/wrist RGB, correctly converted measured-style FK StateV1 and natural language. No target coordinates, task/source/phase IDs, segmentation, depth, success, time or future expert values enter the collator.

V must implement the real source-specific H1 adapter rather than promote its parent generic PARTIAL view. Audit all counts/IDs, split intersections, frame and FK consistency; replay original collision checks on saved accepted output states using the authorized private local bindings, with the exact replay scope stated. It must not claim independent collision proof merely from producer flags. No uploading private URDF/meshes/WBC/tokenizers to make the audit easier.

V signs the exact dataset/processor manifest, or lists precise unresolved correctness defects. It does not edit M's source data. Cosmetic cards/video are not a training prerequisite. Once BOTH behavior data requirements, actual codec/input integrity, V's H1 receipt and resources pass, THIS authority releases the canaries and BOTH6000-step runs. Do not request another routine GO or return to old observation-design restrictions already amended here.

## 6. Fixed matched model pilot — actual training authorized

Initializers are unchanged from the parent:

- Base: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- G: `leoyuer/pi0246-gate-pass-campaign@efecea485ae77000599d19171b02491f8d514f42`, prefix `arms/PI0252/r1-speed-margin-20260928/G/checkpoint-2000`.
- Cosmos/tokenizer lineage: `9ce19a195e423419c349abfc86fd07178b230561`.

Resolve existing full receipts/header reconciliation and slot10 original/adapted state. No latest-version or substitute initializer. Both use the new common observation/normalization; neither step0 inherits old G quality by name alone.

Freeze6000 actual source/visit rows per arm under local seed26003: M1/M3 exactly3000 updates each, balanced paired commands, uniform selected TRAIN family/sequence, then motion quartile and valid window. Same schedule, processed tensors and source/visit-keyed actual native tau/epsilon/stock noise for both arms. Actual identity mismatches are detected before the affected backward/update; record observations, not just recomputed expectations.

For each initializer: actual GR00T continuous8 updates versus4 + complete Trainer save + new-process restore +4, with the6000-step scheduler.32 discarded canary updates total. Compare actual input/randomness, model/optimizer/scheduler/cursor and mode histories with declared numeric tolerances in the sealed runtime card; no silent relaxation after a mismatch. Main arms restart from original initializers, not canary weights.

Both main arms train6000 canonical updates. Fresh AdamW LR1e-5,betas(.9,.999),eps1e-8,WD1e-5,gradnorm1,batch1,accumulation1,warmup10,cosine6000. Existing BF16 frozen backbone and FP32 trainable projector/action-head diffusion scope; no visual/LLM/vlln unfreezing.

Objective stays native flow matching +.4 physical position +.1 physical increment +.1 physical rotation +.1 peak-speed margin, on the original differentiable fullS4 rollout. Speed margin remains1.8m/s; it is not a dynamic safety certificate. Expert is a loss label, not model conditioning. No new loss/seed/LR branch.

Save complete Trainer1000/3000/6000 with a verified durable rolling source. Onlystep0/final6000 are quality-screened. Do not pick an intermediate checkpoint or switch model per task. Both arms finish unless a genuine integrity/resource blocker makes a partial outcome explicit. G's historical training cost remains extra; equal new updates does not imply equal total compute.

## 7. Fixed evaluation, not a post-hoc camera/scene selection loop

Freeze before first quality query, by full source IDs and actual two-noise packets:

For EACH of M1 and M3:32TRAIN windows from distinct sequences;16 KIN-VAL windows;12 semantic inputs from six paired VAL shared-prefix families;12 reserved meaning-equivalent paraphrase cases;6 empty-language cases. Add64 historical O64 regression inputs. Four model states (Base/G step0 andfinal6000) times2noise gives `8*(78*2+64)=1760` planned query tuples. Exact duplicate tuples may be reused and must not be double counted as new computation.

The12 semantic and paraphrase cases need an expert-defined discriminative H40 before inspecting model outputs. Shared prefixes that do not yet separate are not mistaken for model failure; missing the prespecified full evaluable set means INCOMPLETE, not a smaller passing denominator. No phase or future-path field is introduced to make the model decide direction.

Report per behavior and noise, macro/worst behavior, train/VAL/semantic/O64 panels: decoded same-frame XYZ/SO3, first jumps, speeds, gate issues, correct versus opposite-reference margin, requested target/direction/ordered-contour predicates, fixed geometric feasibility without repairing predictions. Include all invalid/rejected prediction rows in outcome rates. Report metrics before and after training, never merely output inequality.

Synthetic research quality is PASS only if EVERY behavior/noise KIN-VAL group meets medianXYZ<=.020m,p95XYZ<=.035m,medianSO3<=.050rad,zeroREJECT and at least10/12 correct semantic directions AND equivalent-paraphrase cases with fixed denominators. Model quality FAIL does not mean experiment incomplete; an integrity failure does. O64 under the new processor is regression relative to newstep0, not old historical qualification.

VAL results apply only to the explicitly conditioned, new synthetic development domain. No result is called WBC tracking, actual task success, broad physical generalization, original-M3/M2 success or deployed readiness. TEST, oldDEV/F and all learned-policy physical episodes remain locked.

V receives H2 raw predictions/metrics once sealed; no additional inference to improve a video. Use PI0257-v2 renderer and source-scale axes, with `KINEMATIC REFERENCE / OFFLINE PREDICTION` provenance. Existing GUARD_STUB_ONLY physical entry stays unchanged.

## 8. Two-agent handoff and autonomy

M owns all generation, source datasets, model computation and Hopper submissions. V owns an independent read-only review worktree. V's model load/query/backward/optimizer/scene-construction/physics/WBC counters all remain0.

Three handoffs only: H0 new cue/camera evidence and design; H1 selected dataset, rejected proposal ledger, splits and actual codec/input results; H2 Base/G model outputs and scores. Reuse existing schemas/transfer/manifest tools with versioned adapters, not a new service or orchestrator. A metadata/readback correction must not automatically trigger a model or scene rerun.

Every handoff has source commit, protocol/camera/reference/solver/processor/model IDs as applicable, SHA256/size, complete IDs and outcome index, raw arrays and missing fields. V returns a companion receipt bound to exact producer hash and states manual-inspection scope. Already frozen evidence is not overwritten.

Engineering defects within defined algorithms may be repaired, narrowly tested and resumed without a fixed retry-count rule, while respecting remaining raw-compute budgets. Examples: import/path/schema/serialization/device/cached-buffer/recorder fixes. No redefinition of thresholds, cue pack, camera list, source support, IK branches, data quotas, task set or training recipe to force success. Resource waits are not quality failures; unknown scheduler status never authorizes duplicate qsub.

No M2 work or video cosmetics block already-correct M1/M3 data and model stages. V having no new package means finite WAITING_FOR_SEALED_HANDOFF, not endless polling. A reviewer must not sign unavailable evidence merely to maintain momentum.

## 9. Budgets, storage and terminal states

New allocations; no unused old budget is added:

| Activity | Limit |
|---|---:|
| New model calls/scene construction/physics by Agent V |0|
| Camera calibration, using existing valid q arrays |512 RGB pairs|
| Fresh family proposals |160 total:20DEV/100TRAIN/40VAL|
| Fresh kinematic sequence attempts |640 maximum,4 per proposed family|
| Selected full datasets if eligible |6DEV+40TRAIN+10VAL families;224 sequences|
| No-valid-result sequence technical reserve |24, total actual sequence attempts<=664|
| New active H40 windows / production render pairs |60000 /120000|
| New physics integration/WBC/hardware calls |0, including constructors|
| Main optimizer updates |12000,6000 per arm|
| Discarded canary updates |32|
| Extra completed-update engineering/recovery reserve |64|
| Total completed optimizer calls |12096|
| Raw training action-head calls |109120|
| Planned model quality queries |1760|
| Strict no-valid-result query reserve |64|
| Total query attempts / inference heads |1824 /7296|
| TEST/DEV-F/learned-policy physical transfer |0|

The640 possible geometry runs are a proposal-search maximum, not a target to execute after quotas fill. No model action scores influence which proposals are selected. Missing/failed records and zero-scene import attempts are recorded, not erased. A valid scientific output is never retried merely for a better score.

Corelab owns zero-integration rendering, CPU geometry/QA and no-gradient model evaluation; Hopper owns admitted real-model canaries/training on a compatible allocated GPU. Legal auto scheduling with tested memory/runtime capability is allowed; do not wait for a particular host by name, repeat known OOM configurations, or use H200 to debug ordinary CPU paths. Check live owners/UUIDs, no killing unrelated work.

Use task-local compact arrays, not huge per-frame NFS trees. Preserve historical G and current complete resume/evidence; no broad storage cleanup. Existing approved HF repo with unique PI0260 prefix and fixed revisions; anonymous and authenticated readback reported separately. Do not upload private URDF/mesh/WBC/tokenizer/credentials. Capacity checks separate df/quota and next-write overlap plus10GiB reserve; quotaUNKNOWN is not quotaPASS. No mail, monitor or PR merge.

Successful experiment terminal: `PI0260_KIN_MOTION2_PILOT_COMPLETE — STOP_FOR_PHYSICAL_TRANSFER_ADMISSION`, only with BOTH actual6000 endpoints and the complete fixed evaluation; report each model's quality separately.

Possible blockers: `PI0260_CAMERA_DESIGN_BLOCKED`, `PI0260_ELIGIBLE_SCENE_SHORTFALL`, `PI0260_DATA_CODEC_OR_OBSERVATION_BLOCKED`, `PI0260_MODEL_INPUT_OR_RESUME_BLOCKED`, `PI0260_PARTIAL_MODEL_PILOT`. M2 stays deferred. `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`. A synthetic research candidate may be recorded under a separate research-only field, never a formal/deployment selection.

## 10. Immediate role instructions

**M:** Read this full protocol and both final H0 records. Do not reclassify the old audit or blame the disproved nominal-start mismatch. Build the explicitly new fixed cue pack/camera calibration using saved valid q, then freeze one rig. Implement deterministic preassigned proposal strata and joint M1/M3 eligibility, retain rejected families and label the conditional domain. Complete H0/H1 with V. On correct data/codec/input/recovery/resource admission, automatically run the fixed Base/G6000 experiments and their1760-tuple evaluation. Do not stop for another routine PI GO; do stop on actual unmet gates. No old single-task restart, M2 collision search, WBC settling or physics is required.

**V:** Keep PI0257-v2 frozen. Reuse its package/visual tools; add only source-specific H0/H1/H2 checks. Verify original raster evidence rather than new producerPASS booleans, distinguish missing geometry from occlusion and task scope from success. Audit proposal/accepted denominators, no TEST leakage, full family pairing and actual codec identity. Sign exact accepted hashes or give precise blockers. Never generate scenes/models or change source data. Deliver understandable source-resolution cues and results without prolonging training for media polish.

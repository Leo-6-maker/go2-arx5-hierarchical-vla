# PI0276 — Mature-environment EEF task fine-tuning and physical closed loop

Date: 2026-10-06 SGT.
Authority: **PI0276_MANISKILL_EEF_TASK_FINETUNING_AND_PHYSICS_CLOSED_LOOP_GO_V1**.
Research charter: GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1, commit `ed398fee17151cfec47864fed9d48ce149b28d61`, Issue #11 direction #5994405991.
Reviewed project SHA: `abcf772c49d84d71339f4c76f1d65b850f90ebd9`. Working branch remains `pi0267/executor-decoupled-recovery`.
The release containing this protocol and `reports/PI0276/design/REVIEW_AND_BUDGET.json` is bound by the new Issue #11 canonical.

## 0. Decision: authorize the complete learning experiment, not another preparation-only pilot

**GO for implementation, a pinned mature manipulation environment, scripted-source data, EEF conversion and replay, a new task processor, real GR00T fine-tuning, and fixed physical closed-loop evaluation.** Passing preconditions automatically unlock the next already-budgeted phase. Do not stop at installation, DATA_READY, an offline loss plot, or a saved checkpoint. A concrete integrity/resource/feasibility failure is a legitimate bounded stop; success is not guaranteed by this authorization.

Choose ONE environment for this goal: **ManiSkill StackCube-v1 with panda_wristcam**, pinned to `mani-skill/ManiSkill@62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`. Do not install and sweep LIBERO, RoboTwin and ManiSkill in parallel. This source already has a pick/lift/transfer/release planner, an actual rigid-body success predicate, and two cameras. It avoids beginning with a bimanual macro-planning API or human-teleoperation datasets. It is not asserted to be intrinsically superior to those platforms.

The concrete objective is **grasp the red cube, move it onto the green cube, release it, and retain a stable physical stack**. This first experiment is one manipulation skill with randomized layouts, not a multi-task or language-selection claim. Do not add a retreat requirement that the source demonstration does not teach. Door opening, bookshelf insertion, mobile-base coordination, language counterfactual tasks and cross-embodiment transfer remain subsequent experiments.

| Charter field | Binding answer |
|---|---|
| object_task | Stack the red cube on the green cube and release it; physical object poses, velocities and contact determine the result. |
| high_level_contract | body/wrist RGB + measured task-space TCP and gripper state + language -> H40 XYZ/row-rot6d/gripper references. No robot joints or privileged goal/phase/success inputs. |
| frozen_wbc_identity | Senior's Go2+ARX5 WBC is unchanged and **NOT CALLED** here. Native Panda controllers are explicitly SOURCE-ENVIRONMENT executors, not the senior's WBC. |
| task_data_provenance | Official scripted motion-planning source, converted and physically replayed into task-space references. No target-robot teleoperation or target-WBC success filtering. Source-robot kinematic bias is disclosed. |
| single_intervention | First end-to-end object-task adaptation in a mature environment. BASE0 versus TASK8192 isolates additional task training within the NEW source/processor contract; it is not an ablation claiming one new component explains all gains over historical geometry experiments. |
| evidence_and_controls | Native joint-teacher versus converted EEF replay; four open-gripper controls; fixed BASE0/TASK8192 physical tests; TRAIN and VAL fit diagnostics. |
| budget_and_stop | Sections 10–12. One initializer, one formal training arm, one final checkpoint. No quality-driven seed replacement or extra healthy steps. |
| claim_boundary | At most physical SIMULATION task learning in this fixed-base source robot. Not Go2 WBC success, real-robot transfer, multi-task autonomy, unseen skill learning or CVPR readiness. |

## 1. Previous goals, actual state and safe handoff

At review, PI0275 `CPU_ENTRY_RECEIPT.json` says native checks NOT_RUN, zero physical starts/ticks/renders and zero GR00T/WBC calls. The latest commit only preserves exact scene values across a 1–2 ULP derived norm difference. This does not establish physical grasping. Issue #11 incremental read after #5999317212 returned no newer comment, so the owner must refresh actual runtime logs before acting; the reviewer has no SSH or live scheduler access.

This is an explicit scientific **source-platform/task-plan replacement**, not an engineering patch to PI0275. If PI0275 still has zero native starts at adoption, preserve its CPU work and mark its remaining custom-tool production `SUPERSEDED_BEFORE_NATIVE_RUN_BY_PI0276`; do not spend its remaining budget building a competing custom physics benchmark. If a PI0275 native job is already genuinely running, do not kill or reset it: let its already-authorized bounded path close, archive it honestly, then start PI0276 when resources are free. PI0276 does not require its quality PASS. Do not report unstarted conditions as failures or convert old counters into this budget.

PI0274 retains its own authority, data, losses, precision and evaluation. Finish any healthy finite evaluation/closeout without modification. Source preparation for PI0276 may be isolated while it runs; no GPU contention or competing automatic owner. Issue #47 is a reusable-code backlog, not a prerequisite approval or permission to merge historical implementations.

Use one Codex main owner, separate runtime roots and `scripts/pi0276_execution/`, `reports/PI0276/`. No new long-lived branch, PR77 pointer, scheduled monitor, mail change or broad cleanup. Actual runtime file hashes and `__file__` paths must be recorded. Existing valid archives and exact source proofs are reused; do not repeat all historical model downloads or experiments.

## 2. Pinned source and runtime setup

Read and bind these exact upstream files at the pinned SHA:
- `mani_skill/envs/tasks/tabletop/stack_cube.py`;
- `mani_skill/agents/robots/panda/panda.py` and `panda_wristcam.py`;
- `mani_skill/agents/controllers/pd_ee_pose.py`;
- `mani_skill/examples/motionplanning/panda/solutions/stack_cube.py`;
- `mani_skill/examples/motionplanning/panda/motionplanner.py` and its two-finger/base planner imports;
- `mani_skill/utils/structs/types.py`, plus the resolved native dependency/asset inventory.

Install in an isolated compatible Python environment, not into the sealed IsaacGym or GR00T environments. Resolve the pinned repository's declared dependencies, record a lock, import versions, Panda asset hashes and selected controller configuration BEFORE scored source episodes. Download only required Panda/stacking assets and source packages. No environment migration for the senior's controller. Missing drivers/assets/permissions are concrete blockers, not reasons to silently switch benchmark or public GR00T release.

Use one CPU-physics environment (`sim_backend=cpu` or the pinned equivalent), GPU graphics, native scene/material/drive settings, sim_freq=100 Hz and control_freq=20 Hz. Source teacher uses **pd_joint_pos**, not pd_joint_pos_vel; converted/reference and learned rollouts use **pd_ee_pose**, absolute, `use_delta=False`, `use_target=False`, `normalize_action=False` on its arm. Native arm and finger gains/force limits stay at pinned Panda defaults. Physical Panda simulation is authorized; Go2 WBC and real hardware are not.

Use default task geometry/randomization and panda_wristcam assets. Freeze camera defaults, native 128x128 RGB, `base_camera` and `hand_camera`. Preserve raw frames; a single fixed bilinear RGB resize to 96x96 supplies the already-tested project interface. No target-dependent markers, masks or camera movement. Record the exact resize implementation/version. Depth, object IDs, cube poses, contact-derived is_grasped, robot qpos/qvel, teacher stage, reward and success stay evidence-side.

Explicit horizon modification: registered StackCube-v1 has max_episode_steps=50. This goal uses **640 control steps =32 s**, scored task interval 0..30 s and a fixed 28..30 s persistence window, with the remaining tail recorded for data and safety. Disable only automatic termination/reset on success in the evaluation wrapper; do not feed success to control or freeze the policy when it first succeeds. Report this as `StackCube-v1 / PI0276 extended-horizon EEF protocol`, NOT an official leaderboard-equivalent score. Keep the original per-tick task predicate unchanged and separately report first success and persistent success.

Two counted native fixtures, each <=10 s: (i) sensor/TCP/native initialization and finite-state/clock checks; (ii) nonidentity orientation plus open/close transport and FK/EEF-controller target consistency in clear space. These prove interfaces, not task success. Preserve invalid runs. No physics tuning, pose setters during execution, welded grasp or hidden object forces. Reset/state setting is permitted only to establish the recorded initial state before t=0.

## 3. Fixed data and test conditions

Pre-register seeds BEFORE any source or model results:
- DEV: integers 0..7 (8 conditions).
- TRAIN: 10000..10095 (96 conditions).
- VAL: 20000..20015 (16 conditions).
- TEST: 30000..30031 (32 conditions).

One source-teacher attempt and one EEF replay per DEV/TRAIN/VAL seed; 120 source slots, at most240 full episodes. Do not collect until a target number of successful demonstrations is reached. Preserve source failure, planner failure, EEF conversion failure and missing/corrupt evidence separately. Only the official planner's fixed internal grasp-candidate procedure is allowed; it is part of the source teacher, not a search over evaluation seeds. Freeze its native planning limits and record plans/calls. No added retry/reset loop.

All streams from a seed remain in the same split. Initial state includes robot, cubes, randomization and RNG identity; store it rather than assuming the same seed guarantees identical physics. Source and EEF replay must start from matching measured physical states. Preserve exact values and numerical comparison results. A difference beyond1e-6 m/rad or declared native float precision must be diagnosed before using a paired replay, not hidden by broad tolerance changes.

TEST is not rendered, teacher-labelled, queried or inspected until dataset, training schedule and final TASK8192 weights are sealed. Even later TEST teacher data cannot enter training/statistics. These are held-out fine-tuning seeds of a known task, not proof the foundation model never saw related pretraining data. Same cube assets are not unseen object categories.

Constant instruction: `Stack the red cube on the green cube and release it.` It is a required interface field, not evidence that language grounding was newly learned. Do not claim two colors mean two skills.

## 4. Source teacher -> true commanded EEF -> physical EEF replay

Wrap the actual official solver's calls to env.step, without changing task goals, motions or native grasp selection. Capture each **commanded** joint target, gripper action, actual q/TCP, physical clock, both cube states and source phase annotation. Instrument the actual commanded target path, not a field merely named endpose/action. Record all source source-file/asset hashes. At each20 Hz source call, compute TCP FK at the commanded arm target using an independent native FK model and the fixed source root transform. Never set the live articulation to the target just to obtain FK. Store world-frame commanded pose and actual achieved pose separately.

Source phases for offline grouping are fixed by solver call sites: (0) reach and descend; (1) close gripper; (2) lift and align; (3) open and terminal continuation. They never enter the learner. When the native script returns normally, continue the last actually sent joint/EEF reference with its current opening to32 s, regardless of success. This is an explicit teacher terminal policy, not invented padding or an evaluation oracle. If its script/planner fails, record invalid future labels and use a disclosed safe passive continuation where possible, not a fabricated successful reference.

The bridge replays the full saved commanded EEF and gripper sequence through the SAME absolute pd_ee_pose controller that later receives GR00T output. Use fresh **replay** observations as training inputs; do not pair original joint-controller RGB/state with a different executor's achieved state.

Critical rotation detail: at the pinned ManiSkill source, absolute pd_ee_pose takes BASE-FRAME position and **XYZ Euler angles** via `euler_angles_to_matrix(..., "XYZ")`, NOT axis-angle. Convert world TCP through inverse robot-root pose, use a matrix->XYZ-Euler inverse consistent with that exact function, and verify recomposition on nonidentity rotations to1e-5. Merely writing scipy.as_rotvec or silently assuming lowercase/extrinsic Euler is prohibited. ManiSkill Pose is wxyz; project canonical storage is xyzw. Project raw labels/state are row-rot6d; go row->SO(3)->xyzw chunk, not through the column-packed convenience constructor. Record transformed native target matrices after compute_target_pose and FK at installed IK targets.

Native IK may return None and hold the starting joints. Retain this as an IK failure/hold count and tracking result; do not label it a perfect reference execution. Use current actual joints only inside the executor. They are never GR00T inputs. No privileged planner is called during learned evaluation to choose grasp targets or a different task path.

Gripper transport:
- Source native command is -1 closed, +1 open. Dataset action channel is `g_cmd=(a_native+1)/2`; it is **an actuator fraction, not measured physical width**.
- Native Panda internally maps this command to finger targets from-.01 to.04 m; the slight commanded overclosure is part of its published controller. Preserve that native actuator model. Do not falsely interpret g_cmd=0 as a measured negative opening.
- State gripper is actual inner opening, from calibrated finger geometry/q, divided by nominal .08 m. Preserve measured values and numerical excursions, no clipping state to make it look like a command.
- At every executed reference tick translate g_cmd back to2*g_cmd-1. New, predeclared learner transport permits actuator-only clipping of a finite g_cmd in[-.10,1.10] to[0,1], with raw/applied values and clipping rates/magnitudes saved. Values outside[-.10,1.10] or nonfinite fail the episode. This saturation applies identically to BASE0 and TASK8192; it is NOT a pose correction, not an old-gate change, and not evidence the raw model is always in range. Source labels themselves stay[0,1].

Keep the project's non-corrective full-H40 TrajectorySafetyGateV1 numerics (2 m/s,4 rad/s rejection, original first-point/path-length thresholds) with the new frame/TCP binding. No clipping, smoothing, retiming or endpoint overwrite for XYZ/SO3. Gate raw predicted chunks before use; ACCEPT/WARN can execute with logged reasons; REJECT is a valid quality failure and is not retried. A stationary-path WARN is not an automatic stop. The gate is a benchmark envelope, not real hardware safety certification. Source EEF replays use actual recorded future commands; the last source records are executed from the saved sequence and not padded into artificial H40 labels.

Run DEV source+replay first. Automatic continuation requires at least6/8 positive source episodes and6/8 persistent EEF replays, trustworthy observation/target transforms, and no setter/privilege/integrity violation. Otherwise stop as a concrete SOURCE_OR_BRIDGE_FAIL, with TRAIN/VAL unstarted, rather than train against an unproven command converter. Native source and EEF success rates/differences are both reported. No 3.5 cm tracking criterion is silently substituted for object success.

Four fixed anti-attachment controls use DEV0..3 EEF root-command records, same initial states, and gripper forced OPEN for the whole replay. No new teacher planning. Keep failed parents; disclose when a negative is not paired with a positive parent. All four remain outside training/statistics and must not establish sustained grasp/stack success for DATA_READY. Do not require an object never to move through an ordinary physical collision.

## 5. Data validity, new task normalization and sampling

A complete32 s replay gives640 observation/command records at t=0,.05,...,31.95, with physical states retained through32 s. A window starts at t and uses the40 actual future commands at t+0..1.95 s, maximum601 anchors per complete episode. Never splice episodes, fabricate future source stages, use achieved poses as if commands, or fill missing tails. All failures and excluded windows retain manifest rows.

Positive BC bank: successful persistent EEF replays from TRAIN only, with complete genuine command windows that pass the original full-H40 gate and valid input schema. Source Panda success filtering is disclosed; **no Go2/WBC result is available or used**. Failed source/replay data remain for failure analysis, not automatic bad-label imitation. Require at least72/96 positive TRAIN seeds and12/16 positive VAL replays, and representation/phase coverage in the admitted TRAIN bank. These are one pilot's readiness targets, not universal sample-size requirements. At least two eligible windows in each of four source phase groups per admitted TRAIN episode; otherwise report that episode's coverage failure and the exact stage. Do not sample future TEST to make counts pass.

Policy input allowlist is exactly body_rgb/wrist_rgb/state_v1/language. State is10D measured world XYZ, row-rot6d, measured opening fraction. Output is H40x10 commanded world XYZ/row-rot6d/g_cmd, embedded in the existing132D head with only those10 channels active. Fixed policy position anchor[0,0,0] in `PI0276_MANISKILL_WORLD`; TCP `PI0276_PANDA_HAND_TCP`. The original relative-SE(3) forward/inverse transformation is retained; do not apply the anchor twice. Sensor metadata may be stored for evidence but is not an unregistered condition.

**New processor contract explicitly authorized: PI0276_TASK_SE3_NORMALIZATION_V1.** This is a new task model identity, not permission to apply new stats to historical policies and relabel their old results.
- Preserve the qualified actual processor, modality mapping,132D masks and relative-SE(3) transform order.
- Compute pre-normalization effective state/action XYZ bounds from admitted TRAIN data ONLY, using the same transformation used in training. Use actual min/max; for any XYZ channel narrower than.10 m, symmetrically expand the interval to.10 m about its observed midpoint. Store actual extrema and true quantiles separately; low/high stored in q01/q99 fields are declared affine calibration bounds, not falsely claimed percentiles.
- Effective valid rotation-matrix row6d channels use the analytical range[-1,1], not near-constant empirical quantiles. The codec must prove these are physical row6d channels at that boundary; if the actual transformed representation differs, stop for a source-contract contradiction rather than guessing.
- g_cmd affine range[0,1]; state opening uses TRAIN measured min/max with minimum interval.10 in opening-fraction units. Require actual commanded OPEN and CLOSED examples and nonconstant measured opening. No normalization from DEV/VAL/TEST, no global z-score/clipping, no modification of PI0274/75 files.
- Actual native StateActionProcessor roundtrip must preserve XYZ<=1e-5 m,SO3<=1e-4 rad,g_cmd<=1e-6, active masks and padding. Test all admitted unique input/label conditions deterministically; preserve every gate-before/after discrepancy. Known old q01=q99=.5 produced approximately million-scale gripper features despite exact roundtrip; this must not recur.

Build a manifest-backed map-style dataset and exact8192-update/16384-microbatch schedule before learner construction. For microbatch index m, phase_group=m mod4. Within that group, cycle admitted TRAIN episodes in seed order; within an episode cycle valid window anchors ordered by timestamp. Each phase thus gets25% of consumption; release/terminal data cannot dominate merely because long terminal holds produce more windows. Save actual per-window/episode/group exposure. Do not feed grouping metadata to the model.

Offline panel, frozen from saved records:128 TRAIN windows (32/group, selected by increasing SHA256 of seed/anchor within group) and64 VAL windows (16/group). No TEST windows or teacher data here. Report whole-H40 and executed-prefix XYZ/SO3, opening fraction MAE, OPEN/CLOSE errors, gate and all failures. These are192 conditions, not192 independent task episodes.

## 6. Model initializer and fixed training recipe

One initializer: **PI0270 AGG4096 weights only**, already qualified for recoverability, NOT assumed best for new manipulation.
HF repository `leoyuer/pi0246-gate-pass-campaign`, revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`, prefix `arms/PI0270/AGG/checkpoint-4096`, full tree SHA256 `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`.
Reuse closed fixed-file proof and verify local model inputs; do not redownload optimizer to initialize a fresh optimizer. No selection among BROAD/TARGET/EQUIL/DIRECT based on task scores. A frozen BASE0 is these weights under this NEW processor/source executor; it is explicitly NOT the historical AGG policy contract or a zero-shot foundation-model result.

Use the actually sealed project GR00T N1.7/Cosmos model sources, not today's upstream main. Use source-bound isolated versions of PI0273/PI0270/R3 helpers. Hardcoded old task membership/status strings must be replaced by corresponding PI0276 evidence, not faked as historical PASS. Preserve mathematical functions, real mode/gradient checks and parameter inventory. Actual import paths/hashes are required; copying one matching filename is insufficient.

Formal arm **TASK8192**:8192 updates, batch1 x gradient accumulation2, one GPU, fresh AdamW/optimizer state; lr1e-5, cosine scheduler, warmup10, weight_decay1e-5, max_grad_norm1.0; retain qualified optimizer betas/epsilon and record them. Dataset/model seed27603. Same471 trainable/559 frozen tensor name/shape inventory as the selected initializer; no LLM/vision unfreeze, added head or joint target. Check that optimizer contains exactly the permitted tensors and some permitted weights actually change.

Retain the qualified objective:
`L_native + .4 L_position + .1 L_increment + .1 L_rotation + .1 L_speed` on the fully generated four-step S4 trajectory. Retain existing80/20 temporal weights for position/rotation/increment and unchanged all-H40 native and speed terms. Use the source-qualified physical definitions and scale; no added gripper loss, history, stop token, output smoothing, task-oracle projection, learned simulator or differentiation through physics. Native loss/mask supervises the nonconstant gripper; save its gradient and action-error diagnostics explicitly.

Precision is R1 phase-specific: original mixed parameter storage, native BF16 epsilon/autocast, generated training query/features FP32, physical losses FP32; official model-only evaluation BF16 query and original physical decode boundary. Explicit phase, not .training, selects the contract. Do not force training query to BF16 or demand cross-phase bitwise equality.

One actual resume canary: continuous8 versus4-save-fresh-process-resume4, total16 completed updates, full8192-step scheduler context. Compare actual input/label/noise sequences, same-phase native/generated parity, gradients, model/optimizer tensors and RNG/scheduler. Native saved/restore comparison uses the qualified exact tolerances; no loose tolerance to admit an optimization. Then restart the formal arm from the fixed initializer/fresh optimizer, not from canary-trained weights.

The first128 formal updates are an engineering/learning-health observation of the SAME continuing run, not another reset experiment. Check finite losses/gradients, live gripper supervision and changed trainable weights. An arbitrary low loss is not an early gate and is not a reason to alter the recipe. Stop only for actual invalid input/numerical/gradient/freeze/recovery defects or resource safety. Do not look at TEST to decide whether to continue.

Save complete4096 and8192 Trainers; evaluate ONLY8192. No mid-checkpoint tournament. Independently archive completed immutable checkpoints to fixed HF revisions; optimizer/scheduler/RNG/provenance must be included. Do not delete the only recovery copy. End of training is not the end of this goal.

Engineering accelerations authorized for this new driver: immutable decoded-array/CPU-input caching, incremental JSONL cursors, one-time verified control-log loading, ordered two-worker prefetch (CPU only), pin_memory where available. Dataset loader has a separate RNG, original sample order is explicit, only actual consumption advances the checkpoint cursor. Bind these BEFORE canary. If exact continuation fails, use the logged technical allowance to repair or return to zero workers; no batch/accumulation/precision changes, no cached cross-step random noise. Do not patch old live trainers. Report actual walltime/resources rather than claiming earlier microbenchmarks are training speedups.

## 7. Learned physical rollout: no teacher, no goal fields, no success stopping

Run model server in the qualified GR00T environment and the physical client in the isolated ManiSkill environment. Use the established finite process/serialized-array architecture, with fresh same-tick images and actual TCP/opening. Do not import IsaacGym into ManiSkill or vice versa. Simulation pauses for inference: **physical simulation closed loop, NON_REALTIME**, not a proof of real-time throughput.

At20 Hz native control, request a fresh H40 once every8 control steps (.4 s); execute decoded indices0..7, then reobserve. Last unexecuted predicted tail is superseded, not appended. Exactly80 maximum queries per32 s episode. Each executed XYZ/SO3 is converted to native absolute pd_ee_pose as in the admitted expert bridge, with separate per-tick gripper commands. No phase/goal/success information crosses the process boundary. The controller is allowed current Panda joints for IK, never object truth for task correction.

Save raw model H40, native installed targets/IK status, commanded and measured opening, actual EEF and cube states, all source transforms, pair contacts and native success at every physics/control sample as applicable. Record model latency but do not convert wallclock waiting into simulated object freezing claims about hardware.

Stop valid episode on nonfinite/invalid SO3, full-H40 gate REJECT, out-of-envelope gripper, unrecoverable physical state or technical failure. Record the reason and remaining time, never rerun for a better outcome. Native IK hold behavior is logged; the old native fallback is not hidden success. Reset only once per declared start. At first native success, continue policy queries/physics through32 s; no target freeze, oracle stop or success-triggered action replacement.

## 8. Evaluation matrix and metrics

After final8192 checkpoint, data/normalizer/schedule and all evaluation noise tensors are sealed:

| Lane | Conditions | New model calls |
|---|---:|---:|
| BASE0 offline |192 frozen TRAIN/VAL windows|192|
| TASK8192 offline |same192|192|
| Native joint expert TEST control |32 held-out seeds, new source traces|0|
| EEF expert TEST control |same32, converted replay|0|
| BASE0 physical TEST |same32 fixed initial states|<=2560|
| TASK8192 physical TEST |same32 fixed initial states|<=2560|
| TASK8192 physical TRAIN diagnostic |16 lowest-seed BC-admitted TRAIN episodes|<=1280|

Use one fixed noise packet per condition/replan, same BASE0/TASK8192 TEST sequence; retain actual tensor bytes and hash, not just seed. Noise is generated once from a local independent generator seeded27611; it never depends on success or model outputs. BASE0 and TASK8192 will visit different states; matched noise is not a claim those inputs remain equal.

For each TEST seed, run and report all four policies/controllers in the table. TEST expert failures remain in the denominator of both model rows; do not define the test subset using expert outcome. Original expert and EEF upper bounds localize conversion/execution limits. TRAIN rollouts reveal memorization/closed-loop-fit problems, not generalization.

Primary original task predicate: exact pinned StackCube `evaluate` (red on green, red static, red not grasped). Report first-success counts/time and fixed28..30 s predicate persistence. Persistence is evaluated at20 Hz samples including endpoints (41 samples); retain100 Hz physical state/contact arrays for replay and safety diagnostics. Native static thresholds remain lin .01 m/s,ang .5 rad/s; on-top geometric tolerances are from source, not replaced by project reaching tolerances. This is the native source's operational success definition, not a universal grasp guarantee.

Registered completion objective: TASK8192 persistent success>=24/32 TEST, no nonfinite or reference-gate failures in those successful cases, and no setter/privilege/source-integrity violation anywhere. Report all collision/impact/contact/penetration, cube drops, actual joint-limit excursions and gripper saturation separately. This objective is a task-learning feasibility gate, NOT blanket whole-body/hardware safety qualification. Do not assert zero dangerous collision if the relevant contact channel is not measured.

Learning evidence is separate: paired per-seed success changes versus BASE0, net gain and discordant counts; fixed offline error differences. A useful planned effect target is at least8/32 additional persistent successes, but no significance or superiority is guaranteed; report exact counts and binomial uncertainty. If BASE0 already meets the task gate or both policies are similar, disclose that fine-tuning has not established the claimed gain. Never rewrite thresholds or evaluate additional seeds because the difference is small.

An executed negative result is still COMPLETE_VALID_QUALITY_FAIL. If source expert is high but converted EEF low, the bridge is the bottleneck. If admitted data/codec are valid but both TRAIN fit/rollout are poor, inspect fitting/representation. If TRAIN works but TEST fails, only then prioritize coverage/generalization. If first success improves but persistence does not, report terminal behavior separately. No automatic DAgger, second arm, new loss or physics retuning within this goal.

## 9. Data/model admission and automatic state machine

Required runner interface to IMPLEMENT (these commands are a deliverable, not a claim existing files already provide them):
`python scripts/pi0276_execution/run.py all --release <canonical-release> --resume`
Subcommands: audit, setup, native, source, bridge, dataset, admit, canary, train, evaluate, replay, archive, closeout.

Each successful stage atomically writes its receipt and the runner advances. Source/bridge gates, nondegenerate processor and true canary are substantive automatic gates, not additional human approvals. No stop at DATA_READY requiring a new training GO: this protocol already grants8192 updates and the fixed evaluations. Persist a single owner/active-process lock; a reconnect adopts active work, does not start a second coordinator.

If a native dependency, bridge, insufficient usable data, exact-resume check or resource cannot be resolved within its fixed allowance, close with concrete evidence and deferred stages. No promise to continue invisibly, no waiting for optional storage/helper signatures. Report which task objective remains untested. Finite failures must not be converted into new physics attempts or counted as model quality retries.

## 10. Budget, counted from first actual calls

| New work | Planned maximum | Hard bound |
|---|---:|---:|
| DEV/TRAIN/VAL source teacher slots |120|fixed seeds, no added scientific slots|
| Corresponding EEF replays |120|same fixed slots|
| TEST native/EEF controls |32+32|fixed32 held-out seeds|
| Open-gripper controls |4|fixed DEV0..3|
| BASE0/TASK TEST and TASK TRAIN rollouts |32+32+16|same fixed conditions|
| Native fixtures |2x10 s|2 scientific conditions|
| Full32 s physical episodes |388|technical reserve below only|
| Total physical starts |390|406, at most16 same-condition technical replacement starts|
|20 Hz control steps |248720|<=260000|
|100 Hz simulator steps |1243600|1300000, including native/setup/reset physics|
| RGB frame pairs |248720|270000, no extra videos rerendered|
| Source-teacher executed control commands |<=97280|110000; planner dry-run calls separately logged|
| Formal optimizer updates |8192|8192 healthy updates|
| Canary updates |16|included in total below|
| Total completed optimizer updates |8208|8464 including256 technical repeated/discarded updates|
| Training action-head calls |count actual|155648|
| Quality/inference requests |6784|8192 attempts including binding probes/technical allowance|
| Quality S4 heads |27136|32768|
| Go2+ARX5 WBC / real hardware runs |0|0|

Arithmetic:120*2 +64 TEST experts +4 negative controls +80 learned rollouts =388 full32 s episodes, plus two10 s fixtures. A32 s rollout has640 control calls,3200 physics steps, and up to640 RGB pairs; fixtures add400 control calls/2000 physics steps. The data source has at most96*601=57696 overlapping TRAIN windows, NOT that many independent demonstrations. Quality=80*80 +2*192=6784. Model-loading has no hidden free forward allowance; binding/probe queries consume the quality cap. All failed/in-flight calls and low-level steps count. Do not equate source planner calls, control calls and neural queries.

Valid quality FAIL never permits replay. The16 technical starts may repair same semantic execution/recording defects on the SAME registered condition, preserving all failed attempts. No arbitrary native-physics bitwise-resume demand: same-seed fresh technical replay is explicitly logged and may differ numerically. Training technical repeated steps count under8464; changes to scientific recipe require a new decision. Do not spend unused PI0275/PI0274 allowance here.

## 11. Required evidence and closure

Source/setup lock; actual source/asset/controller hashes; split/initial-state/noise cards; all source and EEF success/failure records; command-versus-achieved traces; paired FK/native-target checks; data membership and TRAIN-only processor; effective consumed schedule; real canary and training audit; complete4096/8192 Trainer fixed-revision readbacks; BASE0/TASK offline panels; all32 TEST seeds with expert and learned rows;16 TRAIN diagnostic rows; gripper/open/closed/contact/saturation metrics; and final budget.

Recompute final object results from saved physical arrays, not just success JSON. Use source math definitions with a separately written aggregation/reconstruction and disclose shared primitives. Never call a second invocation of the same scorer fully independent. Preserve original per-tick source success beside the CPU reconstruction and explain discrepancies. Model audit can replay saved S4/decode without fresh neural calls.

Videos: saved frames for TEST seeds30000..30003 under BASE0/TASK8192 and the four fixed open-gripper controls, including all failures. Additional figure selection must list its deterministic rule and cannot replace the full table. No rerender for prettier success. Archive new datasets/cards/source wrappers/metrics/video to the existing authorized HF project location with pinned file/member hashes; retain source licenses. Reuse old closed model proofs. Do not remove unique data or full Trainer until durable readback passes.

Write FINAL_RESEARCH_RETURN.md, CLOSEOUT.json, SOURCE_INDEX.json and per-condition CSV; publish Issue #11 terminal and read it back. Distinguish execution, physical task quality, source/codec validity and archival completeness. Do not return only a summary of planned stages.

Successful or valid negative terminal: **PI0276_PHYSICS_TASK_FINETUNING_COMPLETE — STOP_FOR_FROZEN_WBC_COMPOSITION_OR_TARGETED_TASK_FAILURE_DECISION**. This grants no actual WBC/hardware deployment. `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`; a separately named source-domain task candidate may be recorded only with its full new policy identity and this limited qualification.

## 12. Non-goals and provenance boundaries

No new senior-WBC training, no Go2 task teleoperation, no hidden task-state actor input, no qpos labels to GR00T, no benchmark success oracle used to stop or hold, no label padding, no camera/physics/seed sweep, no direct-versus-S4 comparison, no test-set tuning and no automatic second8192-step run. Claims about opening doors, bookshelves, mobile navigation or another embodiment are deferred. Source fixed-base controller success cannot be relabelled as whole-body control.

The provided historical research notes motivate: (i) expert-through-gate and roundtrip before diagnosing a learner; (ii) weights+processor+statistics+mapping+decode as policy identity; (iii) object outcomes separate from trajectory/control metrics; and (iv) no target-WBC-success data filter. Their older selected models, statuses and WBC-dependent curriculum suggestions do not override the current charter or this protocol.

Startup: read the new canonical and fixed release plus charter#5994405991; adopt actual PI0274/75 status without interruption or duplicate starts. Prefer the pinned mature StackCube environment, execute source/EEF replay and data checks, then automatically canary -> TASK8192 -> fixed learned physical tests -> CPU/video/HF/terminal. This is a training-and-physics GO, not preparation-only. Keep the senior WBC frozen and unused, no quality retries or budget reset, selected/deployment NONE.

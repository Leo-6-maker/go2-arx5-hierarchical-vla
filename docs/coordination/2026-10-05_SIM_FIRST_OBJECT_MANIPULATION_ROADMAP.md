# Sim-first object manipulation with GR00T and the supplied frozen WBC

Date: 2026-10-05 SGT.
Status: **implementation roadmap / coordination; NOT a new scientific GO, run budget or hardware authorization**.
Initial code-review base: `cb6e17d3a478bbd953b8c5fb5b49cb5f842fa824`.
Latest execution commit observed during review: `0083f010882cc9cc3b54319ceccdd0f2ff0feb6c`.
Existing authority remains PI0274 canonical5990552027 / release495b17078cb57fdbd47d8257d33202c57beb058e. Do not alter its source seals, data, optimizer, losses, masks, precision, evaluation or budgets. No new numbered goal is introduced here. One existing main owner; no competing job or permanent branch. selected/deployment remain NONE.

## 1. User constraints and current evidence

The senior researcher supplied the WBC used from the start. The upper-level team has kept that WBC frozen and developed/fine-tuned GR00T. The senior now reports mostly performance improvements with no major interface change; this is a handoff statement, not an independently verified new binary/ABI. The user currently has no real-robot dataset. Do not require robot task teleoperation before starting synthetic task development, and do not attribute WBC training to the upper-level team.

The useful next objective is **one physically simulated object-transfer task, then a controlled frozen-WBC composition**, not another assertion that geometric pose reaching already implies book sorting or door opening.

During review new PI0274 receipts appeared: at2026-10-05T11:39:53.221764Z (19:39:53 SGT), EQUIL has1534/5120 formal updates,3068 matched consumed microbatches,1550 completed updates including16 canary. BROAD field2496 is complete with CPU replay; EQUIL quality is NOT_YET_STARTED, full terminal NOT_COMPLETE. The earlier canary receipt reports exact resume PASS. This is repository evidence, not reviewer live PBS/SSH. New object-task work must not interrupt healthy PI0274.

## 2. What was actually read and what can be reused

At cb6e17d3:
- `README.md`, blobbd5b556c8d250429115f263b4ac9f1489296ba83: repository contains contracts, adapters, runners and provenance; large/private WBC/model/assets can remain outside Git. A repository URL alone is not proof that all runnable weights are present. Restore only actually missing dependencies from their manifests, not all historical artifacts.
- `src/go2arx5_vla/adapters/groot_to_wbc.py`, blob3695b4c7bc437ded5f49b1da7ea24bf923c0468a: reference-only adapter, EEF_XYZ_ROT6D, explicit TCP/frame checks; independent current-time GripperCommandV1. It does not load/control a robot. Presence of a gripper field does not prove future gripper commands are correctly synchronized to execution.
- `src/go2arx5_vla/trajectory/preview.py`, blob2e94cb0f35609308baf786161a5fc9a32dfd33cc: nine spatial SE(3) preview offsets, lambda0.15m/rad, row-rot6d, pure-rotation support, terminal pose clamping and stationary paths. These semantics must be compared with the ACTUAL loaded sink, not assumed solely from the helper name.
- `scripts/pi0273_execution/quality_virtual.py` lines1-163, blobcbf3e6af74a8f8c9cf323177aecd30dd34d32bf4: .4s feedback, native S4, original gate, ideal/lagged mathematical EEF execution; explicitly NOT_WBC/NON_REALTIME, physics_steps0, nonarticulated gripper fixed.5. This is not a grasp/contact/object-motion environment.
- `scripts/pi0267_virtual_renderer.py`, blob1ade1400be03e35e4081c60e913b41c32aa208ab: fixed boxes for palm/fingers, rigid placement, no simulation-time advancement. The body camera is fixed using the registered root pose; the wrist camera follows the tool. The source body camera is not a demonstrated moving-camera mobile-robot model.
At0083f010:
- `reports/PI0274/execution/BROAD_FIELD_AND_TRAINING_PROGRESS.json`, blobacee3542a54d9a0c422268a5a34557a32135a82f: latest partial training and completed baseline field.
- Commit diff includes `CONTINUATION_R1_ACTUAL_PROGRESS.json`: exact canary and metadata-only continuation; not a completed EQUIL model-quality result.

Reusable: current task-space action/state codecs, proven trainer/inference mechanics, reference contract, original safe handoff infrastructure and saved-record audits. Not automatically reusable as new-task semantics: static gripper labels/statistics, geometric task teacher, marker-only success scorer, static body-camera distribution, or zero-physics executor as an object-task simulator.

## 3. Target runtime: task reference, not joint control or one-shot long execution

Keep the external action structure as H40 rows of [XYZ(3), row-rot6d(6), gripper(1)] with explicit timestamps, frame, TCP and validity. H40 is0..1.95s at.05s in the existing configuration. The gripper is a parallel channel; it must actually be commanded and measured in the new task.

Runtime intent:
RGB+language+measured task-space EEF/gripper -> GR00T -> complete decoded reference -> gate -> actual native trajectory installation/progress -> FROZEN WBC and gripper controller -> fresh measured pose/images -> next chunk.

A whole door-opening/sorting task is not one fixed two-second prediction. Receding-horizon chunks compose into the task; the exact online handoff cadence must be established for the native backend. Do not blindly impose the E0/E1 .4s lockstep semantics on a spatial-progress WBC.

Before a new integration run, check only the meaningful ABI delta: TCP/frame/units and left/right rotation convention, time versus spatial progress, actual native first-point/chunk replacement behavior, parallel gripper command scheduling and measured opening, stale-reference handling. Same dimensions do not prove these. No need to redesign a mostly unchanged ABI.

A new WBC release is a new executor identity. Test it under its frozen code/weight/config binding; do not rewrite older experiment identities. The high-level does not produce leg/arm joint labels. The WBC does not receive hidden object-goal truth to repair a wrong high-level task choice.

## 4. First task and extension order

Recommended first learned object task: in a clear, bounded environment, move the language-specified rigid object from one support to a destination bin/large slot, release it and withdraw. Start with an easily grasped rigid shape and generous placement tolerance. Include a second object/second destination so instructions have actual behavioral meaning.

At later native-WBC composition, arrange source and destination so both are reachable through allowable base relocation but not both by the stationary arm; measure base movement. A free-tool task alone is not evidence of locomotion or obstacle-aware navigation.

Extension order, conditional on actual results:
1. Single-object transfer with release and retreat.
2. Placement on an open shelf at different heights/positions; then a rigid book-shaped item into a roomy slot. Dense book extraction and tight insertion are separate harder contact tasks, not a rename of simple placement.
3. Simple hinged cabinet/door without a latch. Script handle approach, grasp and hinge-compatible path; the door angle must change through physical contact. A later latch/twist stage needs separate support.
4. Multi-object rearrangement by repeated learned transfers. Report whether a human supplies subcommands, an explicit high-level planner selects them, or one learned policy autonomously manages the whole instruction. Do not conceal a privileged task-phase oracle as VLA autonomy.

For a door, a geometry teacher may use p(theta)=c+R_axis(theta)(p0-c) and a handle-consistent orientation. This constructs an intended contact trajectory, not a guarantee of opening. The simulator must include hinge/damping/friction and actual handle contact. A pose-only controller can be adequate for some low-force tasks but is not a general force/compliance guarantee. Do not compensate with arbitrarily high servo forces or fake door-angle updates.

## 5. No real data: generate task supervision with a physically simulated tool

Preferred task-data boundary:
scene/object geometry + language + physics-driven articulated parallel gripper + task teacher -> actual RGB/EEF/gripper observations + expert task-space references.

The task data generator does NOT call the target WBC and does NOT retain only tasks that the target WBC successfully executes. It is allowed to verify task success and safe contact in the independent tool environment; that is distinct from filtering on target-robot execution. It remains synthetic tool-domain data, not proof of robot feasibility.

Implement a new environment beside the sealed geometric evaluator, not by silently changing it. Use the existing simulator ecosystem when suitable; do not migrate every training/WBC dependency merely to generate a rigid-object task. Required features:
- movable rigid bodies and articulated door/shelf assets where applicable;
- actuated fingers, gravity, contact and friction, bounded tool actuation/velocity rather than teleporting through objects;
- actual object motion from physics, no success-triggered parenting, magnetic attachment, welding or manually assigned object/door poses advertised as learned manipulation;
- fresh cameras rendered from the same scene and actual tool/object state;
- explicit contact-intent rules: grasp contact is allowed; dangerous impacts/interpenetration and non-target collisions are failures. Do not reuse an all-contact-is-collision geometric criterion as an object-task definition;
- separate source of expert proposals and achieved tool/object trajectories, with failure reasons retained.

A bounded supervisory task teacher can use exact simulator state OFFLINE: choose target; pregrasp; align; close fingers; verify grasp; lift; transfer; lower; release; retreat; recover from a miss/drop when a valid solution exists. Transitions depend on actual tool/object state rather than elapsed frame index alone. The teacher's phase, hidden target pose, object identity and success flags remain label/diagnostic metadata, not policy observations.

A static RGB/EEF observation may be insufficient to distinguish some task histories. Check such aliases rather than training contradictory labels. Observation history or learned stage representations are legitimate future choices, but are a separate scientific change, not a silent feature addition to the current protocol. Do not use simulator teacher phase at policy evaluation to solve aliasing.

## 6. Dataset and GR00T adaptation contract

Store complete episodes first, then split by source scene/layout/object instance before slicing overlapping H40 windows. Dataset records should separate:
- policy inputs: synchronized body/wrist RGB, measured EEF and gripper state, exact language;
- training actions: teacher COMMANDED task-space pose/gripper trajectory, defined relative/absolute at the actual observation anchor under one frozen mapping;
- evidence only: object poses, teacher phase, contact/grasp status, intended and achieved trajectories, task outcome, camera transforms, timestamps and all source IDs.

Do not label a delayed achieved pose as though it were the command that caused it, or copy nominal images onto perturbed states. Teacher recovery uses the current actual state. Counterfactual language pairs and layout variation must make visual/language selection necessary rather than correlating each instruction with a fixed object position.

Explicitly audit gripper semantics: measured opening versus commanded opening, open/close polarity, physical width normalization and command scheduler; active action/state masks; nonconstant positive examples of close, carry, release and post-release withdrawal. Existing data used .5 constantly. Do not assume old percentile statistics can encode open/closed actions: test representative raw->processor->decode round trips, actual min/max support and masks. Any required new gripper scale or other statistics are a new TRAIN-only frozen task contract with a new model identity; no silent change to PI0274 or historical readbacks. Reuse pose statistics only if appropriate on the new TRAIN data; avoid needless global refits, clipping or reinterpretation.

Reuse the verified S4/native/physical-loss training implementation and R1 phase precision where applicable. Gripper currently has no verified object-task learning evidence; audit actual gradients/supervision instead of assuming a reserved output dimension has acquired grasp skill. Do not add several losses or architecture changes simultaneously. New task learning needs a separate fixed data/budget/evaluation protocol after the teacher/data pilot is runnable.

Initialization is not settled by the geometric scoreboard. BROAD has useful interface adaptation but heavy geometric-task fine-tuning. Compare only a bounded, predeclared initialization baseline if budget permits; at minimum preserve a frozen original/earlier qualified initializer and do not call BROAD an established best initialization for contact manipulation. No unbounded checkpoint sweep.

Planning milestones, not already executed results or time guarantees:
- a small set of complete scripted physics episodes establishes contact, data schema and task scorer;
- same expert labels pass the intended codec/gate and preserve physical meaning;
- a small TRAIN-fitting check establishes pose AND gripper learning;
- expand independent scenes, objects and camera conditions, not only overlapping windows;
- test learned policy with no teacher control in held-out tool-physics tasks;
- then run scripted-reference and learned-reference comparisons on frozen WBC separately.

## 7. The moving-camera and sim-to-real problem

Current body camera is fixed. A mobile robot's body camera moves with its body, while wrist camera follows its tool. Synthetic task observations must cover legal camera trajectories, occlusions and extrinsics, not only random colors. Tool-state history does not uniquely determine base-camera pose; do not infer it from TCP. Use independently declared rig paths in tool-domain data, then actual camera mounting/state in native-WBC evaluation. This preserves the distinction between nuisance viewpoint variation and body-specific policy input.

No real data now does not block scene/teacher/codec/simulator work. Begin from the old declared calibration/model geometry; represent unknown real mounting and hardware differences honestly and vary plausible camera/light/material/object/contact parameters. Domain randomization is mitigation, not evidence of real transfer. Do not aggressively mask all body/hand appearance without verifying task information is retained.

Later minimum real information need not be a task-teleoperation dataset: camera model/mount/calibration, a few synchronized RGB+measured TCP/gripper recordings, tool geometry, scene dimensions and a known reference/achieved trace. Receiving such data for calibration/testing is distinct from training on target-robot task demonstrations. The broader pretrained GR00T may already use robot data; the claim is no NEW target-robot task teleoperation for this adaptation, not no robot data ever.

Without any real observations or trials, the honest result is synthetic training + simulation validation with expected, unverified transfer. Neither a simulator score nor unchanged ABI guarantees door/bookshelf performance on hardware.

## 8. Separate task physics, WBC competence and learned high-level behavior

Use this conceptual evaluation ladder, with future run budgets authorized separately:
1. Scripted task teacher + independent physics gripper: the task, contact model and labels are valid in the data domain.
2. Learned GR00T + same physics gripper: high-level object-task learning, no target-WBC dependency.
3. Same scripted task-space reference + frozen Go2/ARX5 WBC in dynamics: feasibility/tracking/base relocation/contact upper bound in the robot domain. Do NOT remove these failed conditions from the high-level training set and then claim independent task data.
4. Learned GR00T + same frozen WBC: system composition, new actual observations and feedback.
5. After protected readiness checks, real-system evaluation with a qualified local operator and existing emergency/fallback controls.

Report raw, installed and achieved references; time-aligned pose tracking and gripper delay; grasp/lift/release; final object pose/container occupancy/door angle; collateral collisions, falls, interventions, base displacement and task duration. Expert geometry or reference proximity is not object-task success. Positive expert control with negative learned control localizes an upper-layer gap; failure of both does not justify blindly adding GR00T training.

Old35-40s geometric reliability tests and failures remain unchanged. New object-task success/termination must be preregistered from object semantics; evaluation can score an object result but must not feed hidden success back to the policy or use it to disguise drifting control. Any deployed completion/hold mechanism must be observable/learned and explicitly part of the compared system.

## 9. Concrete implementation ownership and preparation output

The upper-level owner can prepare proposed isolated modules (names below are recommendations, NOT claims that they exist):
- task specs/scorer for single-object transfer;
- tool-object physics environment and moving-rig observations;
- scripted task teacher, including recovery and validity diagnostics;
- episode recorder/converter with independent policy/evidence fields;
- learned-policy evaluator and frozen-WBC adapter contract tests.

Start by reusing old manifests/runtime inputs and the existing reference-only adapter. Ask the senior only for unresolved differences: fixed new checkpoint/config identity, actual TCP/gripper meaning, camera mounting, contact/load/velocity limits and a compatible runner. Do not request re-delivery of already available artifacts or require nonexistent real demonstrations as a condition for writing the task simulator.

Immediate deliverable before the meeting is a concrete first-task design, data schema, exact gripper/time/frame contract and runnable CPU-only format/roundtrip fixtures where current source data permits. A future bounded task-physics pilot is the next experiment to authorize; this roadmap itself authorizes no new simulation, rendering, teacher/model query, training or motors. Existing PI0274 continues on its own budget. No experiment is claimed executed by this document.

## 10. External implementation references and limits (checked2026-10-05)

- ManiSkill official motion-planning guide: https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html . Built-in Panda task solvers and recording demonstrate a non-teleoperation path using scripted substeps and a planner. They are examples to adapt, not a Go2/ARX5 drop-in. A surrogate-arm generator is a different source embodiment; its geometry/visual/kinematic biases must be disclosed.
- MimicGen official getting-started: https://github.com/NVlabs/mimicgen/blob/main/docs/tutorials/getting_started.md . Standard workflow starts with source HUMAN demonstrations. It is not automatically a demonstration-free solution; public demonstrations avoid new target-robot collection but do not mean no human/robot data.
- Isaac Lab Mimic official overview: https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/imitation-learning/teleop_imitation.html . Reuses source demonstrations and task subsegments; the version is a referenced example, not a requested migration of the sealed IsaacGym runtime.
- UMI on Legs author Q&A: https://umi-on-legs.github.io/ . EEF interface does not guarantee reachability and force/contact execution; explicit body-side limitations remain.
- NVIDIA synthetic-data domain-gap explanation: https://developer.nvidia.com/blog/closing-the-sim2real-gap-with-nvidia-isaac-sim-and-nvidia-isaac-replicator/ . Appearance/content randomization can mitigate gaps, not establish target-system transfer without experiments.

The reviewer did not run SSH, the simulator, new neural or teacher queries, training, motors or real-data acquisition. Public implementations were read, not installed or experimentally reproduced.

## Preparation prompt

读取本路线文档、10月7日meeting记录与Issue11现役PI0274授权。复用师兄旧版冻结WBC及已有EEF接口，不等真机示范：先准备单物体抓取—搬运—释放—撤离的物理仿真任务规格、算法教师、数据字段、夹爪/时间/坐标契约和CPU-only样例检查；明确静态body相机、固定.5夹爪与零物理评估不能直接当作操作环境。任务数据不按目标WBC成功过滤，actor不读特权目标/阶段。形成可单独定预算的最小任务物理pilot方案，不修改或重启PI0274、不新增模型/教师/渲染/物理/真机运行，不重置预算；selected/deployment保持NONE。

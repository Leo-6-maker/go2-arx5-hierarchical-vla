# Next Data Goal Draft

Status: **NOT_AUTHORIZED**. Draft date:2026-10-12.
This is a proposal, not GO, canonical protocol, resumed budget or launch instruction.

## Proposed Question

Can an independent dynamic gripper, with finite actuators and actual contact physics, produce trustworthy complete pick/place/release/withdrawal episodes in target-relevant task geometry without depending on target-WBC success?

High-level contract: dual RGB + language + actual TCP/gripper -> desired/committed EEF SE(3) + gripper.
Frozen WBC identity: NOT_USED_IN_THIS_PROPOSED_DATA_GOAL.
New data domain: INDEPENDENT_DYNAMIC_TOOL, SIM_ONLY, NON_REALTIME; not Panda-to-Go2 transfer or real hardware success.
Single intervention: construct and qualify independent contact-task data/recording, keeping GR00T neural work and WBC work0.

## Reuse And Required Work

Reuse bounded task geometry/scoring and SE(3)/row-rot6d math, four-stream recording concepts, hash manifests and existing causal episode evidence.
Inspect saved TRAIN/DEV source history for current grasp vs ever-lifted, release/withdrawal and late hold; exposed CONFIRM_A is historical evidence, never new training or blind validation.
Panda source bank/27 admitted correction suffixes remain their original provenance and can inform design. Do not automatically mix them into a new task dataset.

Existing target_mujoco.dataset.validate_manifest is specific to frozen native WBC/G1-G3. Implement independent-tool admission in a distinct module/manifest rather than relaxing its gates.
Existing recorder/teacher checks also bind GO2_MJ_PLACE_V1 and deployment gripper qualification; reuse logic selectively with explicit new source semantics, not by relabeling old receipts.
The Core task runtime installation is incomplete/unverified in the bounded check. Resolve environment/package versions through isolated CPU metadata checks before any later physics work.

Required: finite-drive gripper/contact calibration, no weld/snap/teleport object attachment, explicit force/speed/travel/geometry/TCP/collision identity, independent scene, two calibrated actual cameras, object-milestone scorer and recorder.
Do not modify the WBC package or infer certified force/friction from URDF declarations.
Privileged teacher/object/contact events are supervision/scoring only.

## Initial Data Contract

Record complete20Hz episodes before windowing:
- PolicyObs: synchronized dual RGB, actual EEF/gripper, language, masks/snapshot ID.
- Causal history: episode order, actual prior observations/actions, reset/abort/takeover boundaries.
- Supervision: raw desired and planned H40, accepted/installed/actually committed setpoints and timestamps.
- Execution: same-command achieved EEF/gripper, actuator/constraint failures and clock identities.
- Privileged: actual grasp/lift/transport/release/withdrawal/hold and failure/retry events.
- Geometry: camera intrinsics/extrinsics/per-frame pose, metric world frame, object poses and fixed local point IDs; optional depth/visibility.
- Identity: code, scene, teacher, executor, gripper, data order, dtype, processor/stats/decode, seeds/noise and file hashes.

Split layouts/objects/episodes before overlapping H40 windows. No H40 across reset/abort/takeover/invalid-reference.
Maintain current predicates separately from ever-achieved milestones. Save failures; only successful/qualified teacher commands enter positive action supervision.
Prove same-time decode/actual anchor and requested-vs-committed-vs-achieved distinctions.
Fit any future statistics on TRAIN only.

## Proposed Finite Pilot Budget

All numbers below require approval and an exact future protocol/code SHA:
- Up to4 commissioning +8 independent pilot starts, at most32s each, no valid-quality retries.
- At20Hz inclusive endpoints: at most7,692 observation pairs/teacher decisions. At2ms physics timestep: at most192,000 integrations, excluding separately bounded setup steps that must be explicitly registered before GO.
- Corelab physics/render reservation at most1 GPU-hour and CPU at most24 core-hours; no resource reservation in this adoption.
- Output cap10GiB; stop before ENOSPC, never delete prior artifacts to meet it.
- At most2 explicitly versioned teacher variants; preserve original first-valid failures.
- New PBS jobs0, GR00T load/forward/optimizer0, WBC0, hardware0 for this first data-only proposal.
- No formal TRAIN64/DEV16 collection or neural fitting hidden inside pilot. Expansion and H200 fitting require a later gate/budget.

Pilot must separate4 fixed seen preparation layouts and4 held-out layouts. Publish counts by layout and all original attempts, including failures.
Readiness requires real-contact task completion, sustained release/withdrawal/hold, safety, consistent camera/state/command timing, complete provenance and CPU re-scoring. A visually plausible trajectory is insufficient.
The reviewer must choose fixed task dimensions, physical success thresholds, exact seed sets, setup limits, split and acceptance counts before GO.

## Delivery And Stop

Publish small protocol/readback/budget/data-readiness receipts in #11.
Transfer allowed data through HF revision/prefix/file hashes, with READY committed last; reuse credentials.
Resolve HF write scope and dataset visibility before upload. Keep private WBC assets out.
Stop on safety/numerical warning, clock/reset mismatch, invalid labels, unsupported contact/gripper identity, storage limit, exhausted cap or terminal pilot outcome. Do not use unused old budgets.

After trustworthy data exists, propose B0 from official pretrained GR00T on allocated H200. B1 history and B2 predicted task state are later equal-budget interventions; UNITAS-style geometry is initially recorded supervision only.
Future model closed loop requires the relay in NETWORK_AND_EVAL_PLAN.md to be separately qualified.
Independent-tool/Panda success is not Go2/WBC or hardware success.
selected=NONE; deployment=NONE.


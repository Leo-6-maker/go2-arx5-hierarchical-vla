# 2026-10-07 GR00T / frozen-WBC meeting preparation

Date prepared: 2026-10-05 SGT. Meeting: Wednesday 2026-10-07, time to be agreed by the participants.
Review base: `495b17078cb57fdbd47d8257d33202c57beb058e` on `pi0267/executor-decoupled-recovery`.

**Status: coordination and read-only preparation, NOT a new scientific GO or hardware authorization.** This document does not supersede PI0274 canonical #5990552027 / release495b1707, change any experiment, restart work, add neural/teacher/render calls, or reset budgets. Existing PI0274 work continues under its own authority and single owner. No new numbered experiment is created here. Selected/deployment stay NONE.

## 1. Where the high-level work actually stands

The user confirms the project has always used the WBC supplied by the senior researcher, frozen during the upper-level experiments. Our learning contribution to date is GR00T task-space adaptation and evaluation, not training a new WBC or proving an improvement to his controller. His report that a newer real-robot WBC is nearing convergence is an external handoff claim; its released identity and task-level logs have not yet been received in this audit.

Latest read branch HEAD remains the review base; Issue11 returned no comments after 2026-10-05T08:06:57Z; reports/PI0274 contains design only. This establishes no later published result, NOT that the remote owner is idle or a job is not running. No reviewer SSH or live scheduler access.

PI0273 is the last complete quality result read. Both GR00T endpoints completed5120 updates with real exact-resume canaries and four full Trainer archives. BROAD original48/64, reliable9/64, collision0; TARGET original39/64, reliable3/64, collision7. BROAD target-pose first completion32/32 but reliable5/32; directed-contour completion16/32, reliable4/32. Both original-ROT LEGACY PASS; neither policy qualifies for full virtual task reliability. These are developed-geometry mathematical EEF-executor results, NOT frozen-WBC or real-robot task success rates. BROAD can be retained as a diagnostic reference, not promoted to deployment.

The actual PI0273 evaluator declares VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP and a fixed nonarticulated gripper value0.5. It does not demonstrate learned physical grasp/release, object relocation, door contact, or book manipulation. A prescribed open-contour path is not evidence of door opening. No current real-camera task-data/real-system qualification should be inferred from the geometric training metrics. Pretraining may contain broader knowledge, but its survival and usability under this adaptation are unverified.

Historical Stage40/41 has a sealed engineering fixture with4 GR00T forwards and245 native WBC ticks; it also reports joint-limit violations and explicitly disclaims robustness, real time and hardware readiness. It proves a prior integration attempt existed, not that the latest GR00T and the senior researcher's latest WBC are already integrated or safe.

Assessment: reproducible task-space fine-tuning and useful geometric closed-loop behavior exist; reliable task feedback, actual grasp/contact task supervision, and real-system transfer remain separate open work. Do not give a numeric percentage of paper completion or treat remaining work as cable connection alone.

## 2. What our side should finish before the meeting

### By the end of Tuesday October6: evidence and reproducible interfaces

1. Refresh the existing PI0274 owner's actual receipt and preserve exact progress if partial. Continue its existing recipe; do not launch extra training for the meeting or require its completion before discussing integration. Package a one-page status with natural-language task names, explicit virtual/WBC/hardware labels, and first-completion versus final-window retention. Reuse existing all-condition videos/contact sheets and saved failure traces rather than rerendering or querying models. New visualization of already saved records is descriptive and cannot change eligibility or scoring.
2. Prepare the upper-level handoff manifest: diagnostic checkpoint HF revision/prefix/tree; processor/statistics; state/action mappings and masks; relative-to-absolute decode; row-rot6d convention; R1 phase precision; environment/dependency lock; launcher instructions; one already-saved observation/prediction pair with hashes; representative saved evidence for successful arrival and later drift. Use model-only artifacts where appropriate; do not copy all historical optimizer payloads. An instruction to invoke inference is not evidence it has been newly executed.
3. Draft the joint reference/feedback contract and mark every unknown, rather than guessing from the old WBC. Build or inspect CPU-only parsing, coordinate round-trip and log-only transport stubs using saved fixtures. Nothing in this preparation path connects to motors or sends live motion commands. No model execution beyond the current PI0274 authority is granted here.

### Wednesday October7 before the meeting: runnable-data handoff and task choice

4. Using any files already supplied by the senior researcher, check metadata, timestamps, image/state pairing, TCP/frame/rotation conventions, and version differences offline. If no packet is available, report NO_REAL_PACKET_RECEIVED rather than synthesizing a fake real observation. Real images may support a future explicitly budgeted shadow-inference experiment, but receiving logs alone does not authorize a new neural sweep or motor actuation.
5. Produce one integration/task sheet: first protected engineering milestone, first object-manipulation task, required observations/labels, current missing capabilities, named responsible people and next shared machine slot. Proposed first engineering milestone is visual approach/pregrasp with non-contact reference replay; it is not a paper-level manipulation result. Prefer a single easy object transfer involving clear base relocation before full bookshelf sorting or a door requiring forceful contact, unless the senior already has stronger demonstrated contact skills. Decisions should follow actual hardware capability and available task supervision, not the most attractive demo name.

Deliverable at the meeting: one evidence page, one upper-level replay/identity bundle, one interface-difference checklist, one first-task/data/owner decision sheet. Finishing another arbitrary1024/4096 update run is not a meeting deliverable.

## 3. What to request from the senior researcher

Send the request before Wednesday. Ask for existing files/logs first; this is not a request to teleoperate target-robot task demonstrations for high-level training.

| Priority | Requested item | Minimum usable content / decision |
|---|---|---|
| P0 | Frozen real-ready WBC release | Actual code SHA, all actor/normalization/config artifacts, robot/tool assets or their hashes, motor/control gains and limits, state-estimation dependencies, SDK/environment, one launch command. State what changed versus our delivered frozen version. If changes remain, provide a stable integration release and a changelog, not a moving latest folder. |
| P0 | Reference and feedback contract | Absolute/relative frames and transforms, TCP physical point and tool offset, row/column rot6d and quaternion ordering, metres/radians, timed-trajectory versus spatial-progress semantics, preview count/offsets, progress owner, new-chunk replacement policy, stationary/endpoint behavior, command-age/timeout behavior, actual-state timestamp and coordinate conventions. Show one real accepted input packet and its output feedback. |
| P0 | One existing synchronized sensor/control recording | Raw body/wrist RGB where present, per-image acquisition timestamps, measured TCP pose with source time, measured/commanded gripper opening, robot/base state available to WBC, raw and installed EEF reference, achieved pose, run configuration, and whether observations share a clock. State unavailable fields explicitly. Supply camera intrinsics, extrinsics, distortion/resolution/crop settings and TCP calibration. No demand for robot-specific joints in the VLA input. |
| P0 | Actual capability and safety boundary | Existing logs/video for fixed-pose hold, independent rotations, arm-plus-base motion and repeated reference updates; known failures, admissible speeds/accelerations/workspace/tool loads, maximum tolerable data age. For contact, report what was demonstrated, with compliance/force sensing/control and gripper behavior, not only free-space tracking. Identify the qualified local operator, emergency stop, workspace constraints and network-loss fallback. |
| P0 | Integration resources and first slot | Who runs the robot, where the GR00T inference will run, GPU/RAM/network/runtime availability, who owns logging/calibration, and a dated shared protected-integration slot after the meeting. Latency must be measured on that stack later, not inferred from H200 training or an old simulation fixture. |
| P1 | Task/data and publication boundary | Which single manipulation task is feasible now; object/door/shelf geometry and assets; whether grasp/release and task state labels can be generated independently of robot teleoperation; permissible use of sensor logs; publication/data rights; low-level versus high-level contribution and any separate WBC submission overlap. |

The three smallest useful files before the meeting are: release/launch manifest; one accepted trajectory plus synchronized achieved-state log; one synchronized RGB+TCP(+gripper) packet with calibration. They reduce more immediate uncertainty than another best-case demo video alone.

Important: the repository's current preview.py uses nine spatial SE(3) arc-length offsets and row-rot6d. Do NOT assume the senior's new release still uses that ABI. Ask explicitly who controls time/progress. R1 FP32-training/BF16-quality bindings apply to GR00T and are not an instruction to cast the senior's WBC.

## 4. Decisions the meeting must close

- Fix one WBC deployment identity. A new senior-supplied version is a new executor version, not a silent replacement in prior reports. Upper-level controlled experiments continue to use frozen declared backends.
- Agree the reference/feedback ABI using a concrete packet, not just the phrase XYZ+rot6d.
- Choose one protected integration milestone and one real object task; list dependencies and owner/date for each. Scripted reference success establishes executor readiness only. GR00T-generated reference remains separately evaluated.
- Agree how task labels will be produced. Current geometric samples are not a door-opening/grasping dataset. Target-robot sensor recordings used for calibration/diagnosis are not automatically task-training demonstrations; any such training use must be declared. No-task-specific-target-robot-teleoperation is the intended fine-tuning claim, not no robot data anywhere in GR00T pretraining.
- Decide whether the paper claim is frozen-WBC multi-task composition with robot-independent task supervision, or actually demonstrated cross-embodiment transfer. Do not claim the latter without a second suitable body and frozen shared-policy evidence. An EEF interface alone is not a novelty result.
- Set the next small, explicit integration protocol only after the actual release/ABI is known. This meeting memo does not authorize WBC/physics/hardware runs, new real-input model calls or broaden PI0274. It must not generate a second competing scientific owner.

Do not wait for perfect virtual scores to exchange files, review recorded real observations, prepare interface mocks or reserve hardware. Conversely, do not put a known unqualified policy in unattended closed-loop motor control to meet a deadline. Integration readiness and deployment qualification are distinct gates. Old35–40s geometric hold scoring stays unchanged; future object-task metrics must be separately preregistered from task semantics, not post-hoc relaxation of old failures.

## 5. Conference planning context (external, not experimental evidence)

Official CVPR2027 CFP, checked2026-10-05: registration2026-11-10 AoE; paper2026-11-16 AoE; supplementary2026-11-23 AoE. Main deadline is fixed. Wednesday's purpose is to reduce integration/task-data uncertainty on this critical path, not certify submission success. Source: https://cvpr.thecvf.com/Conferences/2027/CallForPapers .

## 6. Sources and read limits

All project paths below were actually read at the review SHA; former chat summaries are not substituted for current receipts.

- `reports/PI0273/final/FINAL_RESEARCH_RETURN.md`: training, archived identities, original/reliable/safety/legacy and virtual scope.
- `scripts/pi0273_execution/quality_virtual.py` lines92–163: fresh-state feedback, original gate, no physics, fixed nonarticulated gripper and NONE selection.
- `reports/CORELAB_STAGE41_M2_BASELINE_PROFILE.md`: full historical4-query/245-tick fixture and its explicit limitations.
- `src/go2arx5_vla/trajectory/preview.py` lines1–135: spatial preview and row-rot6d.
- Current ref and Issue11 since2026-10-05T08:06:57Z; `reports/PI0274` directory listing: no later published execution result observed.

No reviewer SSH, new training, model query, data capture, renderer, WBC or hardware operation was performed. The user's statement about the senior's current real-system convergence is not independently verified here.

## 7. Preparation prompt

先读取本会前协调文档与Issue11现役PI0274授权。保持师兄提供的WBC冻结；复用已完成证据，整理自然语言任务结果、当前真实收据、上层模型完整身份与保存样例、参考/反馈接口差异和首任务数据缺口。用已有文件做CPU解析与log-only接口准备，不新增训练/模型调用/渲染/WBC/真机实验，不重置预算，不改PI0274。周三带着四份交付物对齐固定WBC版本、同步RGB/TCP样例、首任务、负责人和机时；缺失资料如实列出。selected/deployment保持NONE。

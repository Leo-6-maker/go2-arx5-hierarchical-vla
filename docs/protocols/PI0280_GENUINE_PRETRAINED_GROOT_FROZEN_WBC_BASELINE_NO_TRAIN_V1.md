# PI0280 — Genuine pretrained GR00T + frozen Go2/ARX5 WBC baseline

Date: 2026-10-07, Asia/Singapore.
Authority: **PI0280_PRETRAINED_FIRST_FROZEN_WBC_BASELINE_NO_TRAIN_GO_V1**.
Reviewed project SHA: `86a0d44f752b7d42546db1898e50d4bd14b50972`.
Reviewed official NVIDIA/Isaac-GR00T SHA: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
Immediate user-priority pause: Issue #11 **6037170888**, published 2026-10-07 19:42:26 SGT.
The final Issue #11 canonical binds this protocol and `reports/PI0280/design/AUTHORITY_AND_LIMITS.json` at one immutable release.

## 1. User decision and precedence

The user explicitly changes priority: first execute a baseline using the genuine official pretrained GR00T and the senior researcher's frozen WBC; only afterward judge task generalization and whether fine-tuning is needed. **No fine-tuning is authorized in PI0280.** This is not a new name for fresh-base SFT, and not another CLEAR/FOCUS/COVER training run.

Supersede the not-yet-executed scientific permissions of PI0279 canonical6036062568/release4afdb1401650353092428dcb69d0cb353ee74164 and any automatic continuation of previous task-repair/training/preparation plans. They are `PAUSED_BY_USER_PRETRAINED_BASELINE_PRIORITY`. PI0277/PI0278 results remain sealed; preserve everything PI0279 actually completed. Paused does not mean quality failure. Do not revive unused previous budgets, treat a completed data gate as permission to train, or resume after this baseline without a new user/PI decision.

The task-space research charter remains the objective. This baseline is a prerequisite audit of model/interface transfer, not evidence that cross-embodiment generalization has already been obtained. A source-native diagnostic may consume genuine source-robot joint observations required by the official pretrained tag. That exception is confined to the source diagnostic: it does not authorize target-robot joint supervision, invented source joints on Go2, or a claim of embodiment-unaware behavior from a joint-conditioned run.

One existing main owner. Continue the current execution branch, preserve concurrent commits, do not create a competing goal/owner, permanent branch or automatic monitor. Publication itself does not cancel a PBS job or run inference.

## 2. A0 — Actually pause the old work before using its resources

The owner must inspect current jobs, PIDs, locks and ledgers, rather than infer remote state from the last commit. Reviewer saw PI0279 CPU data-preparation code at the review SHA and no later Issue #11 execution comment after6036062568; this does not prove that no pilot or training process is running.

- Queued or unstarted old training/data jobs: cancel only the positively identified project-specific requests. Disable the predecessor driver's automatic transition into training.
- Active training: do not start a new update after recognizing this priority change; finish only the already in-flight atomic optimizer update and save its complete local model/optimizer/scheduler/RNG/cursor. If not safely finishable, preserve the previous valid recovery and record the interrupted attempt. No additional scientific canary/update is allowed.
- Active simulation: safely complete only the current bounded episode or record `USER_INTERRUPTED` with actual counters. Do not start the next teacher/learner episode. Hardware is not involved.
- Do not kill an atomic save, remove unique data, delete old failures, cancel unrelated jobs, or reset historical counters. Pausing supersedes the old instruction to let an entire healthy campaign finish.
- HF bulk and old exhaustive archives remain deferred. Already saved parent and latest/previous full recovery are protected.

Write and publish `reports/PI0280/execution/PAUSE_ACK.json`: observed timestamp; exact affected job/PID; queue action; last completed/partial counts; recovery file identity; disabled continuation; process recheck. Separate `AUTHORITY_PAUSED` from `REMOTE_PAUSE_CONFIRMED`. Any in-flight predecessor completion belongs to that predecessor ledger, not to the new zero-update budget. Owner acknowledgement is an execution receipt, not a second scientific signature. CPU source review may proceed while awaiting a safe boundary; conflicting GPU/physics execution may not.

## 3. What is, and is not, the baseline

Primary model: **nvidia/GR00T-N1.7-3B**, with its original official weights, serialized processor/modality configuration, statistics, embodiment/projector mapping, tokenizer/Cosmos dependency, official decoding and inference settings. All are part of the policy identity.

Not primary pretrained baselines: AGG4096, TASK8192, GRIP, FOCUS, COVER, CLEAR, Stage28 task-adapted weights, NEW_EMBODIMENT placeholder outputs, or NVIDIA's separately task-finetuned LIBERO/DROID/SimplerEnv checkpoints. Do not substitute those when the base performs poorly. An official task-finetuned checkpoint is still fine-tuned and is not authorized as a fallback here.

Use a fresh isolated Python process/check-out for official inference. Do not import project training overlays, patch native forward/loss/normalization, inherit R1 precision overrides, or reuse the old task statistics. No action-head replacement, slot copying, learned mapper, LoRA, prompt optimization, output smoothing, external success controller, extra denoising search, or model selection. All parameters are frozen, eval/inference-only; no backward, optimizer, update, statistics fitting or adaptation data production.

The existing code already uses relative EEF internally. World/task-frame values at the WBC boundary do not prove absolute neural supervision. Do not declare switching to relative actions a new baseline repair.

A baseline run can be correctly executed and still fail the task. Never turn an input incompatibility, loading error or unsupported task into an asserted zero-shot policy-quality failure.

## 4. A1 — Pristine identity and supported input-contract audit

First reuse the existing local pristine model and official source caches. Historical Stage28 points to `/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B`, with original-base tree8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4 and index407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59; those are navigation/provenance, not proof that the current files still match. Verify the actual path, upstream snapshot revision, config and all effective tensors before inference. If a pristine original is genuinely absent, a single fixed-revision official download is allowed within section10; never redownload old campaign Trainers.

Freeze `BASELINE_BINDING.json` before model-quality queries: exact upstream model revision/file hashes, upstream code SHA, dependency/Cosmos snapshot, checkpoint config, dtype and denoising steps, effective modality config, projector row, input/output dimensions, action/state units, frame/TCP, history indices, native action period and wrapper hashes. Use config resolved by the checkpoint, not a table copied from a different version of documentation. Record missing/unexpected keys and reject unexpected fresh/reinitialized learned tensors. Allow only a file-location rebind to a byte-matched offline dependency; no semantic overrides. Hash effective weights before/after the campaign to prove zero updates.

A material current issue: the reviewed source DROID config declares video indices[-15,0], EEF+gripper+joint state and outputs; the prose data guide's generic current-frame recommendation is not sufficient. The actual checkpoint configuration wins. Obtain real historical images with timestamps; do not duplicate the current frame as fictitious history. Initial buffering must follow the official padding rule or a measured stationary pre-roll.

Inspect ALL serialized pretrain routes once, without scoring candidates. Select exactly one source-native route: DROID on the included genuine droid_sample, provided the loaded base confirms support. Identify whether ANY pretrained EEF route can be represented on the target using actual RGB/history, actual EEF/gripper and static documented calibration only. Prefer a genuinely compatible route already validated by the checkpoint, not a posttrain placeholder.

Important constraints:
- Source DROID diagnostics may use real source Panda joints. Actual ARX5 joints are not DROID/Panda joints; padding six angles to seven, invented neutral source angles, filling target XYZ into joint slots, or silently dropping required fields is not allowed.
- A different named camera can be mapped only through an explicit, fixed sensor/frame contract; similar array shape does not establish matching viewpoint distribution. Source-simulation and Go2 observations remain out-of-distribution tests even when units and fields are correct.
- Missing hand/second-arm/joint/visual-history fields cannot be fabricated to satisfy a strict API. No NEW_EMBODIMENT slot is initialized or trained in this goal.
- If no supported route can accept honest target feedback, emit `BLOCKED_TARGET_PRETRAINED_INPUT_CONTRACT` with the exact missing keys/semantics. Still finish nonblocked official inference and WBC reference-replay diagnostics. Do NOT label them a fully closed-loop GR00T+Go2 baseline, and do not automatically train. The next decision may be a declared input adapter or minimal adaptation, but neither is granted here.

This prevents a false dilemma: failure to zero-shot an unsupported embodiment is not proof that the base lacks task knowledge, while mechanically executing its recorded references is not proof of closed-loop transfer.

## 5. A2 — Official native inference, genuinely loaded and generated

Run the stock Gr00tPolicy/official standalone inference on genuine `demo_data/droid_sample`, episodes1 and2. Resolve real episode IDs from pinned metadata; if either is absent, record a source mismatch rather than substitute a convenient task. Select16 evenly spaced valid observation anchors per episode, preserving required history, before examining outputs. No future action labels enter the inference API.

For these32 observations compare the stock reference call and the new thin serving wrapper under the same stored actual initial noise and identical declared runtime. This is64 actual GR00T calls, not64 tasks. Keep native dtype, model horizon and denoising steps (expected4, confirmed from the original package). Record raw network output, stock decoded physical EEF/gripper/joint outputs, valid dimensions, actual language/history/state, timings and model identity. Joint predictions stay diagnostic and are never sent to the Go2 robot.

Report original open-loop error definitions as descriptive source metrics; there is no invented universal MSE gate and no claim of source task completion from open-loop data. Model load/finite-output/decoder parity is an execution gate, not a manipulation-success gate. Require wrapper parity at the stock runtime's declared reproducibility precision; unexplained mismatch is an interface issue, not a reason to tune weights. Record actual noise rather than rely only on same-seed wording.

Do not reuse historical BASE0 predictions as outputs from the genuine base. Do not count cached examples or teacher-conditioned head outputs as fresh autoregressive/free-running predictions.

## 6. A3 — Untrained source-physics bridge, bounded and subordinate to the WBC goal

Once source-native inference works, run the same frozen official model in existing ManiSkill StackCube source-Panda simulation, at most the four previously developed initial states10000..10003, with the fixed original task instruction. This is **zero-update source-domain transfer**, not an in-distribution official benchmark and not Go2/WBC execution. It provides a physical reference-generation comparison, not another SFT campaign.

Only if the source Panda can supply the actual required native state/joints/TCP/gripper/history semantics, map real observations to that source-native route. Reuse raw original128x128 rendered RGB, not downsample96 then enlarge as in the old custom route; retain official preprocessing and disclose the camera-domain difference. Camera or tool extrinsics are physical calibration, frozen before policy scores. If TCP/gripper semantic equivalence cannot be established, mark that lane blocked rather than fabricate values.

Extract only stock-decoded EEF+gripper outputs into canonical SE(3) references; the returned source-joint action is logged but not executed. Decode once: do not apply a second relative transform. Native history/action frequency comes from the bound source metadata, NOT an assumption that H40 always means1.95s. Retain official prediction length and execute the equivalent first8 native timestamps before replanning. At a20Hz source controller, timestamped translation interpolation, SO(3) interpolation and documented gripper sample/hold are allowed solely for time-grid conversion; do not rescale velocities, extrapolate beyond validity or tune resampling based on scores. Maximum80 model calls per32s episode; if native cadence requires more, stop at the bound budget rather than change semantics.

Keep original source dynamics/controller/gate and original StackCube scoring. Record first and28..30s persistent success, release, actual contacts, raw/installed/achieved/reference timing. No scripted finish, oracle state, teacher takeover, success stop, or test-based prompt change. All4 denominators remain even if one reference is rejected. Source-native joints are confined to this diagnostic. The official-data lane and this lane must be labelled separately.

No pass requirement on these4 source scores prevents the independent frozen-WBC fixtures below. Poor predictions are a result; they are not permission for training.

## 7. B — Real frozen-WBC simulation integration, not another ideal/Panda stand-in

### B0: bind the actual controller

Recover the senior-supplied Go2+ARX5 WBC through existing manifests/runtime references. The old repository adapter is reference-only and does not itself instantiate a WBC. Bind actual arm/body/dog actor files, normalizers, configuration, robot/gripper/collision assets, simulator/version, gains, safety termination, dt/decimation and __file__/hashes. Prefer the senior's upgraded release only if a complete runnable manifest is already delivered and verified before outcomes. Otherwise freeze the previously delivered version and label `DELIVERED_OLD_WBC`, not upgraded-WBC performance. Do not wait indefinitely for an unreceived version or silently replace a controller mid-panel.

Historical Stage40/41 had4 model calls/245 native ticks and recorded joint-limit violations; it is a reuse starting point, not new safety or task qualification. Do not inherit its old model/processor. This goal explicitly authorizes the bounded **actual frozen-WBC simulation** below, superseding previous zero-WBC caps only for PI0280. No motors/hardware are authorized.

### B1: native qualification and reference-only controls

Run2 fixed10s source-native WBC fixtures in one bound simulation: (i) hold the measured initial TCP with open gripper; (ii) one smooth5cm vertical translation and return, with known nonidentity-rotation CPU tests outside dynamics. Initial configuration comes from the delivered fixture, not a search for an easy seed. Respect existing controller hard stops; log all inherited limit margins and violations. If the fixture cannot run safely under its original termination rules, report an executor/interface blocker. Never widen a limit.

Use the existing reference spine: canonical TrajectoryChunk -> original SafetyGate -> actual reference manager/native preview -> frozen WBC -> physical Go2/ARX5. The adapter's convenience `from_rot6d` and row/column conventions must be explicitly checked. Construct canonical quaternion chunks with documented matrix conversion where needed. Frame/TCP rigid transforms, timestamps, source gripper physical opening and target actuator polarity/width must be explicit; an arbitrary similarity transform or per-task tuned offset is forbidden.

The delivered WBC uses spatial SE(3) preview (nine points in the reviewed code), not a raw H40 timestep player. Freeze native progress/replacement/endpoint behavior and use actual achieved feedback. Schedule high-level replans from original trajectory timestamps; preserve native WBC dt, do not force it to a Panda clock. Consume gripper commands throughout the installed reference, not just once per chunk. A separate gripper actuator is part of the executable system.

Create one minimal target StackCube scene by reusing the existing task asset/geometry and scripted EEF source references, not by building a broad new benchmark. Use the original four task conditions10000..10003 and a SINGLE fixed task-to-WBC rigid transform, table/support placement and declared initial robot pose. Choose that scene/transform from the supplied robot/tool geometry and nominal script before viewing pretrained outcomes; freeze it. No per-condition/object resizing, target attraction, goal-dependent output correction, post-failure scene changes or result-based seed selection. Real target gripper morphology/contact may make this scene infeasible; report it as such, not as base-model failure. Native task/object checks must exist; a kinematic object attachment is not allowed.

Execute4 existing expert EEF reference replays through this same actual WBC and target scene. This tests task/actuator feasibility; it is not new high-level training data. Compare against the retained source-expert results without treating them as target successes. No replanning with privileged goals is added to learned-policy runs. Expert failures stay in their denominator.

Execute up to4 complete recorded-pretrained EEF streams from A3 through the actual WBC in the paired scene, with their original raw outputs/timestamps and the same gate. This is `PRETRAINED_REFERENCE_REPLAY_WITH_WBC`, explicitly OPEN LOOP with respect to the high-level model. Missing/rejected/truncated A3 streams remain missing/rejected; do not replace them with expert trajectories or regenerate until success. Even if the target feedback input contract is unsupported, these controls can establish which references the real WBC executes. They do not complete B2.

### B2: genuine closed-loop GR00T + Go2/ARX5, if and only if honest target inputs are representable

This is the primary requested end-to-end milestone. It must instantiate both original GR00T and actual frozen WBC, use fresh target-body/wrist observations and actual target TCP/gripper feedback, decode and install a new reference, advance full robot dynamics, then query again from the resulting target state. No source images, source joint states or previously recorded high-level suffix may stand in for live target feedback.

Use the one target-compatible pretrain route fixed by A1, without changing weights/statistics/projector slots. A static calibration and explicit field naming are permitted; inherited task processor or invented source-robot states are not. If none exists, B2 is `BLOCKED_TARGET_PRETRAINED_INPUT_CONTRACT`; B0/B1 and A2 results must still be delivered, without claiming the requested closed-loop baseline completed. Do not evade this boundary via a simulated Panda shadow joint state, neutral padding or an undocumented state-mask assumption.

First use16 previously saved real simulated target observations from B1 as shadow inputs, split by fixed chronological anchors, with no actuation. Then run the same4 target conditions, once each, up to32s and80 model queries. The four conditions are developed, narrow baseline examples, not blind generalization. For severe safety termination stop that attempt; retain gate rejection, collision, fall, joint/torque limits, actuator saturation and physical state. Do not re-seed, retune WBC, or replace a failed valid trial. The script may buffer observation history before time0 under native stationary control; count all pre-roll dynamics and use real buffered frames.

Reference correctness, tracking and object outcome are separate. `CLOSED_LOOP_EXECUTED` requires fresh input provenance and at least3 genuine model queries with at least2 installed reference replacements responding to changed achieved state, not playback. Full task result additionally requires the entire32s interface-valid execution and the unchanged object predicate all41 samples28..30s. Initial integration is not guaranteed merely because arrays fit.

The engineering target is a complete4-condition paired baseline table and a locally runnable frozen model+WBC system; desirable task performance is>=3/4 persistent success but is not promised. Report exact fractions and small-sample uncertainty; even4/4 is not multi-task or hardware evidence. ZERO-success full executions are valid baseline failures, not a reason to run SFT automatically. A structural input/physics blocker means the corresponding milestone was not executed, not four policy-quality failures.

## 8. Speed and reproducibility priorities

H200 is used for native inference, not for training. Prefer one resident pristine model with isolated deterministic request handling and local IPC to the existing simulator; avoid repeated loads and copying9GB per milestone. A2 reference/wrapper parity happens on the same backend. Reuse local model assets and native processed-input caching where identical; never cache stale images, state or future random noise as new observations.

Start with the stock PyTorch path/configuration. No TensorRT/compile/precision/batch/denoising/kernel sweep is part of baseline science. Record actual end-to-end latency and data/model/WBC/physics time. Simulation may pause for inference, labelled NON_REALTIME; historical0.4s or official GPU timing is not a new realtime qualification.

Do not run HF public archive/readback, full old retrospective reports or repeated shard transfers on the critical path. Verify original identity once on loading and immutability at close; keep compact raw outputs/provenance, task/control traces and real saved videos. Optimizer checkpoints are not produced by this zero-update goal. Retain all paused old recoveries.

## 9. Result taxonomy and next decision

Write separate outcomes for:
1. `PRISTINE_MODEL_VERIFIED` — genuine base plus original processor/statistics/config actually loaded.
2. `NATIVE_SOURCE_INFERENCE_REPRODUCED` — official supported-source calls and wrapper parity; not a task.
3. `SOURCE_PHYSICS_BASELINE` — untrained base in source-Panda physics, with exact scores/domain shift.
4. `WBC_REFERENCE_REPLAY_EXECUTED` — actual frozen controller drives full target-body dynamics from saved references; not high-level closed loop.
5. `GROOT_WBC_CLOSED_LOOP_EXECUTED` — genuine fresh target feedback and new predictions through actual WBC.
6. `GROOT_WBC_OBJECT_TASK_RESULT` — original object completion/retention and safety, not trajectory RMSE.

Never collapse those into a single PASS. If input contracts block5, report exactly what minimal non-learning or learned adapter would be needed, without implementing a new scientific workaround here. If expert+WBC fails, investigate the task scene, frame/TCP, gripper/timing or executor first. If source predictions are useful but target feedback fails, isolate transfer/observation/embodiment differences. If official source reproduction fails, first fix loading or processing. If all interfaces are correct but base task quality is poor, this supports considering task adaptation; it does not prove that full VLM fine-tuning, our previous custom losses, or many rounds are necessary.

Only after this evidence is reviewed may the user authorize multi-task/scene generalization or minimal original-base fine-tuning. No automatic return to PI0279. The research claim remains frozen-WBC task-space reuse, with demonstrated scope reported honestly. `selected/deployment=NONE` throughout; even a source or target simulation success does not enable hardware.

## 10. New finite authority and resource budget

New gradient computations, optimizer constructions/steps, learned parameter updates, training canaries, statistics refits, training data production: **0**. Old in-flight update completion, if any, is exclusively a safe-stop predecessor event and must be separately recorded.

Nominal GR00T requests:64 native paired calls +320 source physics +16 target shadow +320 target closed loop =720. Hard cap1024 attempted policy calls and4096 action-head evaluations when the bound native configuration is4-step; all warmup, parity, retry and unsuccessful requests count. If the actual original checkpoint needs another number of steps, preserve it and remain inside the same head cap; do not alter model science to fit a counter.

Nominal physical starts:4 source-Panda +2 WBC fixtures +4 WBC expert-reference +4 WBC recorded-pretrained +4 WBC fresh-policy =18. Hard cap22 starts including at most4 same-condition engineering replacements; valid task/gate failure has0 retries. Source-Panda ticks nominal12800. Native WBC control ticks nominal20200 at the historical50Hz interface; actual native dt/decimation is bound before running. Hard caps:30000 WBC control ticks,120000 actual WBC actor forward calls,250000 total per-environment native simulator integration steps,60000 gripper/control dispatches,30000 RGB frame pairs. Every reset/pre-roll/partial step counts; controllers with different rates must not be retimed to evade caps. No vectorized hidden extra environments.

One legally allocated H200 for frozen inference plus existing legally allocated target simulator resources, maximum12 total GPU-hours for new execution and48 CPU-core-hours. This is a ceiling, not an estimate or deadline; stop early when a blocker is clear. No new cloud/GPU purchase, login-node neural work or duplicate PBS submission. Pristine base/dependency missing-byte downloads <=30GiB combined; new outputs/cache <=60GiB; no upload-first dependency. Preserve personal-quota atomic-write headroom; do not infer it from shared df.

Full WBC SIMULATION is explicitly permitted as above; all real hardware/motor commands remain0. Fresh broad task/scene datasets, teacher-policy corrections, goal-conditioned autonomous task expansion and held-out confirmation30032..30063 are not opened. Developed source examples and narrow target controls are used only for the baseline.

## 11. Implementation, delivery and source basis

Implement isolated code under `scripts/pi0280_execution/`; receipts under `reports/PI0280/`. The driver checks PI0280 authority and zero-training guard at startup; do not modify a running predecessor worker. Version implementation and resolved manifests before neural/physics execution. Entry points should support `pause-adopt`, `bind-pristine`, `native-reproduce`, `source-physics`, `wbc-bind`, `wbc-reference-controls`, `target-shadow`, `target-closed-loop`, and `report`. Data/implementation checks that preserve meaning do not require a second GO; a new input representation, model, controller or learning intervention does.

Deliver PAUSE_ACK; pristine model and weight-loading report; supported-input matrix including joint/history requirements; one fixed complete model/WBC/adapter identity; native-source traces; source and target reference/control/achieved/object tables; actual inference command; short real success/failure recordings; execution/quality/blocker taxonomy; new and preserved-old counters; Issue #11 result with fixed commit and readback. A manifest-only, cached inference or source-Panda result must not be described as a completed GR00T+Go2 WBC baseline.

Sources actually reviewed for this decision: current branch and incremental Issue #11; PI0279 preparation commit86a0d44f; previous pretrained-use audit supplied in conversation; official source51d4c89f `gr00t/configs/data/embodiment_configs.py` (DROID EEF/joint/history); earlier same-SHA model/processor/policy documentation; project `README.md`, `src/go2arx5_vla/adapters/groot_to_wbc.py`, `scripts/stage40_b0_model_control.py`, `reports/CORELAB_STAGE41_M2_BASELINE_PROFILE.md`, and coordination meeting manifest guide. Official Hugging Face base metadata/model card was checked, but this reviewer did not obtain its raw serialized processor JSON or inspect all original weight payloads; A1 must verify those locally. Documentation does not guarantee a native zero-shot Go2 route.

Reviewer performed GitHub read/write only, not SSH, remote stop, inference, training, physics or full checkpoint download. The immediate pause comment and this release are authorization changes; actual shutdown/run completion requires owner receipts. No unverified elapsed-time or quality promise is made.

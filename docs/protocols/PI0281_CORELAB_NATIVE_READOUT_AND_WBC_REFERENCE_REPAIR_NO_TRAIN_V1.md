# PI0281 — Corelab native-readout baseline and frozen-WBC reference repair

Authority: **PI0281_CORELAB_NATIVE_READOUT_WBC_REPAIR_AND_ADAPTATION_GAP_GO_V1**.
Date: 2026-10-07, Asia/Singapore. Reviewed project SHA: `d71082f9b878cf3c13fd4d9372680cdb604c0669`.
Official GR00T source: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
Predecessor terminal: Issue #11 comment **6040035950**, 2026-10-07 22:24:31 SGT; execution/evidence commit `2adbdae100ec8673f0cce1e6870a1b9833a8c112`.
The new canonical binds this document and `reports/PI0281/design/REVIEW_AND_LIMITS.json` at one immutable release.

## 1. Decision, objective and precedence

The user requests a careful failure review, corrective action, and a new Corelab native closed-loop baseline before deciding the fine-tuning gap. This goal authorizes actual source-Panda closed-loop inference, controlled action-readout comparisons, and bounded delivered-WBC simulation/reference repairs. It is NOT another StackCube SFT campaign. New training, backward, optimizer updates, statistics fitting and adaptation dataset production remain ZERO. PI0279 and all old automatic fine-tuning remain paused. PI0280 is sealed; preserve its failures, blocked conditions, counters and archive. No budget reset, retrospective score change or replacement of its effective model.

There are two native baselines, not one interchangeable PASS:
1. The genuine pretrained model consuming honest source-Panda observations and executing its source-native joint/gripper outputs in source simulation; paired EEF diagnostics establish which task-space readout is actually useful.
2. The actual delivered frozen Go2+ARX5 WBC executing native references with a verified coordinate/progress/observation chain.

The raw model's eight serialized routes have already been audited and none accepts honest target EEF/gripper-only input. A new goal cannot make unavailable joints/hands exist. Fresh raw-GR00T+Go2 feedback stays **BLOCKED_TARGET_PRETRAINED_INPUT_CONTRACT** in this goal; target policy calls=0. Source-native success, EEF replay and WBC reference success must not be labelled a completed target VLA closed loop. The result will specify the minimum declared interface adaptation needed next, without executing it. Baseline execution may be correct with zero task success; useful policy quality is separately measured and never promised.

The task-space research charter remains unchanged. Source joint observations/actions are permitted ONLY for this source-native diagnostic. They are not target-robot labels, a new proposed high-level ABI, or an embodiment-unaware result. The desired delivered policy continues to output EEF+gripper; task data must not be generated/admitted according to target-WBC success.

## 2. Evidence and bounded hypotheses

PI0280 verified pristine GR00T revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`, Cosmos `9ce19a195e423419c349abfc86fd07178b230561`, unchanged effective weights, and 32/32 stock/wrapper same-runtime comparisons (64 calls). These are loading/inference results, not 32 successful tasks.

All four source StackCube conditions ran 32 task seconds plus one second of actual history, 60 calls each, with first/persistent success 0/4. Gate239 ACCEPT+1 WARN, no REJECT. IK holds9/191/0/122 and command/achieved XYZ RMS7.765/20.489/3.629/19.541mm are separate diagnostics. The inspected condition10000 has zero grasp samples and no rise from its initial cube height, so the pristine result must not inherit the old tuned policy's terminal-only failure story. Recompute the full four-condition grasp/lift/place/release funnel from saved arrays before assigning causes.

A material newly read source fact: official `examples/DROID/main_gr00t.py` constructs `RobotEnv(action_space="joint_position", gripper_action_space="position")` and executes decoded `joint_position` plus `gripper_position`. PI0280 source.py deliberately discards joint predictions and executes predicted EEF via pd_ee_pose. The official RoboLab guide also executes joint+gripper and uses a separately DROID-finetuned checkpoint; its published scores are NOT raw-base EEF scores. This goal does not download that checkpoint or switch environments. A supported predicted EEF modality does not guarantee equivalence to FK of the separately predicted joint modality.

WBC hold completed10s but raw/actual orientation errors final/max1.074/1.210rad were large; the hold did not save installed orientation. The vertical return stopped at approximately0.5s with spatial progress0.100000m versus time-law0.001224m, exceeding the unchanged0.070000m tracking-abort threshold. Installed versus raw quaternion in the instrumented replacement matched within2.98e-8rad, but actual orientation error was0.988597rad. It was TRACKING_ABORT, not a proven fall or hard-joint-limit event.

The prescribed z(t)=0.025*(1-cos(2*pi*t/10)) path retraces itself and has coincident endpoints, total arclength0.10m. Projection/branch ambiguity, progress initialization/clock handling and the actor's orientation observation chain are working hypotheses. They are NOT proven defects until the exact bound private runtime is traced. One must not diagnose from similarly named historical repository helpers instead of the actual __file__/hash. Copying a correct installed quaternion does not establish correct actor conditioning or physical orientation tracking.

## 3. A0 — Adoption, shared artifacts and three workers

One current main owner adopts Corelab processes, ports, GPU memory and locks, verifies the old pause remains active, and records actual resource availability. No H200/PBS submission is needed. Do not re-download or rerun the64 successful native parity calls. Reuse the sealed model-loading/weight hashes and verify no file modification; check effective tensors at launch and close. Retain all old counters and local recoveries. Use isolated outputs under `scripts/pi0281_execution/` and `reports/PI0281/`.

Up to three disjoint subagents may continue the existing configuration: SOURCE handles source readout/controller diagnostics; WBC handles the bound private reference/observation chain; AUDIT independently recomputes saved arrays and maintains the gap matrix. Owner owns canonical, resource/attempt reservations and integration. Workers write disjoint directories and cannot change another lane's files or grant experiments. Independent GPUs may run concurrently only after actual allocation/memory checks; otherwise serialize native simulation and one resident GR00T server. No duplicate model servers, login-node model work, competing goals or new cloud spend.

Before new physical work, publish an implementation manifest and frozen condition card. CPU diagnostics can run while the other lane initializes. Local original arrays take precedence over HF transfers. Existing HF archive readback is sufficient for predecessor archival status; new bulk archive is deferred.

## 4. A1 — Saved-input dual-output and ABI audit, no new physics

Use all240 saved A3 output packages and the32 unique A2 source outputs. At identical observations, decode the existing EEF modality and compute exact source-Panda FK of the separately decoded joint targets. Apply the already verified DROID/link8/tool/base convention once. Compare H40 and actual first8 by position, geodesic rotation, gripper and the unchanged pose gate. Report mismatch distributions and time/phase dependence; do not assume these two outputs were trained to be perfectly consistent. Do not use achieved positions instead of command targets.

Replay the source input construction on saved RGB/history and actual states: frame order/stride,15Hz source timestamps versus20Hz controls, fixed TCP transform, gripper closure versus actual gap, original statistics/projector and native inverse. PI0280's actual real history and calibrated static transform remain the reference; no fake history or fitted output offset. The data conversion self-consistency already shown by677 DROID frames is preserved, but it does not validate new physical outcome.

Fix only demonstrable semantic bugs against the frozen upstream convention. Record exact old/new inputs and outputs, proof and affected evidence; no prompt search, rotation convention tournament, threshold change or object-dependent correction. Square versus real DROID camera layout and tool morphology are domain shifts, not automatically implementation bugs. Keep original128x128 raw simulator images for the readout comparison, native preprocessing, S4, native BF16 inference and FP32 decoding. Do not simultaneously change resolution/FOV/history/cadence to improve quality.

At most16 fixed existing observations x2 thin-wrapper calls=32 new neural requests for the necessary new native readout wrapper, with stored identical actual noise. Reuse old reference output whenever identical; no repetition of the original64-call test. The native model stays untouched. Prove literal routing of stock decoded absolute q and gripper closure; original relative action must not be decoded twice.

## 5. A2 — Corelab source-native closed loop and task-space comparison

Fixed task/instruction/initial states remain the original StackCube10000..10003. Conditions are developed diagnostic cases, not blind generalization. Every full episode has the original32 task seconds plus1s genuine stationary history;20Hz native simulator control,100Hz integration, and15Hz timestamped model action semantics with H40/first8 and the original absolute scheduling. Maximum60 nominal calls per episode,80 hard; original original images/weights/statistics/gate retained. Store raw model output, installed actuator/reference, actual state and object predicate separately.

### A2a: two short controller semantics fixtures

At most2 source fixtures of2s each, one native joint-position and one EEF controller, verify measured joint/FK state, nonidentity tool rotations and gripper open/close polarity at safe poses. They are not object-task successes. Use existing control gains/physical limits; if an initial pose is unsuitable, report a concrete blocker rather than search poses. Count reset and pre-roll work.

### A2b: ORIGINAL_BASE_JOINT — four genuine closed-loop source episodes

The same original frozen GR00T receives actual source Panda EEF, seven joints, measured closure and true history. Execute its stock-decoded joint_position+gripper_position through the existing native unnormalized pd_joint_pos controller, following the official DROID output choice. This is an OOD simulator port of native output semantics, NOT a claim that we reproduced DROID hardware or official published task scores.

Use the saved PI0280 noise tensor for each matching seed/query index when shape/runtime identical, and document its immutable identity. Policies then see their own actual observations. If a route terminates early, retain that condition and mark unavailable downstream comparisons; never fill a missing suffix with an expert. Joint hard limits, finite values and documented source controller safety apply. No silent q clipping, joint reordering guess, target attraction or smoothing. Report differences between joint safety and EEF gate as part of this controller/readout diagnostic, not a pure identical-executor causal comparison.

Every received output also yields the official predicted EEF (for audit only); the two predictions are NOT blended and the task scorer never selects the better action.

### A2c: FK_OF_JOINT_COMMAND_REPLAY — four physical reference replays

From each A2b condition, preserve the actual commanded joint trajectory and gripper timing. Convert commanded (not achieved or post-IK) source joints by FK and fixed calibration into EEF references, then replay once from the paired source initial state with pd_ee_pose. Reacquire physical observations and score unchanged task outcomes. These are open-loop command replays with respect to GR00T, explicitly NOT new EEF-only learned-policy successes. Retain unavailable/early-terminated source trajectories without extending them.

This separates: native q readout usefulness; direct EEF versus FK(q) prediction consistency; and the source EEF controller's ability to preserve the generated plan. A successful FK replay may suggest a future task-space distillation/interface route, but does not eliminate source-joint conditioning or grant training.

### A2d: expert round-trip controls — four source episodes

Reuse the four existing successful scripted source command sequences, not a new planner or teacher call. Pass their known commanded EEF/gripper through the DROID encode/decode/time-grid bridge and the actual EEF executor, without any neural model. Explicitly quantify re-timing/resampling error before physical execution; protect grasp/release events and preserve actual command semantics. These controls validate this bridge and cannot be counted as model quality. If the15Hz grid materially changes the supplied teacher behavior, report a time-grid/bridge gap; do not repair trajectories or tune durations against success.

### A2e: direct EEF conditional rerun — at most four conditions

The original PI0280 DIRECT_EEF0/4 is the reference, not a failed attempt to delete. Do not rerun it merely hoping for different outcomes. A new four-condition EEF panel is allowed only after one documented input/decode/time-grid semantic defect is proven and corrected in A1, using the same original noise and fixed conditions. Otherwise reuse the sealed results. The correction is applied consistently to the compared route where relevant; old and corrected policy identities/results stay separate. Physical quality failures have no retries.

## 6. B1 — Delivered WBC native-reference failure isolation

Use the exact older delivered WBC identity from PI0280 for diagnosis. Do not replace it by an unreceived upgraded release; an actual new delivery is inventoried for a separate decision rather than mixed into this panel. Read the actual native __file__/hash and config for path canonicalization, nearest projection, progress reset/update, s_ref(t), actor observations, control gains, termination and end-effector link. The project Stage21 file is navigation; the runtime copy and manifest are authoritative.

First reproduce the saved progress jump on CPU with the saved path/poses/times through the actual projection/progress functions. Include deterministic mathematical checks at repeated XYZ/SE3 positions on outbound/return branches, start/end equality, reference replacement and dwell. Check:
- Is progress initialized twice or incorrectly at installation/reset?
- Are absolute episode time, relative reference time and dt/decimation confused?
- Does an endpoint or global nearest-candidate tie jump to another visit of the same pose?
- Are both s_ref and s_progress valid continuous/unwrapped path coordinates?
- Does the native delivered test entry behave differently from the integration wrapper?

Never repair this by setting s_progress=s_ref, zeroing traj_timing_err, disabling TRACKING_ABORT, widening0.07m, or removing the return segment. Keeping only a monotonic path is a diagnostic, not a repaired return result.

For orientation, trace raw quaternion -> canonical curve -> anchored installed quaternion -> preview matrix/6D -> actor-local target error -> actual link and TCP orientations. Bind wxyz/xyzw, active/passive rotations, shoulder/base/world transforms and exact link/tool offsets from assets. The delivered PI0280 binding explicitly recorded a native column-packing shim; do not apply a historical generic row-packed statement without checking the actual actor input. Test nonidentity rotations numerically. Compare native rigid-body pose with independent URDF FK for the SAME physical link/TCP, not two differently defined frames. No empirical rotation bias subtraction or conversion choice based on reward.

An installed quaternion matching raw does not clear the actor input chain. Conversely, a native actor that still tracks orientation poorly with correct inputs is an executor limitation; do not secretly train/retune it.

## 7. B2 — One bounded semantic repair and physical verification

Allow ONE versioned nonlearning repair package covering only proven reference-lifecycle/time/coordinate/observation interpretation defects from B1. Prefer restoring the delivered native semantics where the wrapper diverges. For a demonstrated returning-path branch ambiguity in the delivered projection, a continuity-aware branch/tie correction is authorized, using previous valid segment/progress and native physical search bounds. Any reachable window must be derived from already bound velocity/dt/measurement semantics before scoring, not tuned on successful rollouts. Publish its deterministic rule and CPU regression tests before physical evaluation. If no mechanically justified rule exists, stop that repair as an executor capability gap.

This permission changes reference-management software when necessary, NOT weights, normalization, gains, low-level action scaling, history features, friction, mass, collision geometry, hard stops, robot poses or task thresholds. Record frozen-actor bytes separately from the modified adapter/executor-software identity; do not claim that the entire low-level system is unchanged merely because weights are frozen. Same-semantics error handling may capture termination before auto-reset; it cannot continue the trial after a native stop.

Two versions (native unmodified control path and the one justified corrected path), each with three predeclared10s fixtures, provide six maximum starts:
1. Hold the measured reset TCP pose with open gripper.
2. Monotonic5cm world-Z rise using z=0.025*(1-cos(pi*t/5)) for0..5s, then hold to10s.
3. The unchanged PI0280 retracing return z=0.025*(1-cos(2*pi*t/10)).

Use the delivered fixed seed4242 and robot reset, not outcome-selected initial states. The second fixture locates a closed-path-specific problem, not an easier replacement for fixture3. Reuse an existing exact attempt when every bound byte and recorded observable is sufficient; a newly instrumented control attempt is identified as such, not a quality retry. No third repair branch or parameter sweep.

A usable reference-integration gate requires all three final fixtures complete10s under unchanged native hard stops, with all relevant coordinates/actor inputs accounted for. Additional prospective engineering checks over the final2s: position RMS<=0.020m and max<=0.040m; SO3 RMS<=0.15rad and max<=0.30rad relative to the intended SAME TCP, and no hard URDF limit penetration beyond1e-6rad/m numerical reporting tolerance. Retain all negative soft margins, contact and native tracking criteria. These new requirements are conservative integration gates for this experiment, not a retrospective change to PI0280 nor a hardware qualification. Native stricter criteria win. If they fail, record the real failure; do not tune until pass.

## 8. B3 — Task-reference controls after WBC qualification

Only if B2 passes, instantiate the already frozen PI0280 target task scene and its same four geometry/initial conditions, then execute four existing expert EEF streams and four ORIGINAL_BASE_DIRECT_EEF recorded streams once each (up to32s). No new target scene/scale/seed, no easier replacement, no tuning path or task-to-world transform using outcomes. Preserve original time/progress/independent gripper scheduling and whole-body dynamics. Reuse the physical object scorer and all original safety criteria.

If fewer than3/4 expert replays satisfy the original persistent task predicate, retain that task-executor gap and do not expose the controller to the model-reference panel; those model slots remain unexecuted, not0-quality failures. Otherwise proceed automatically to the four recorded base streams. Incomplete source streams remain incomplete; no teacher-generated continuation. All high-level reference replays are OPEN LOOP with respect to VLA, even though WBC has low-level feedback.

Fresh original-model target feedback remains unsupported and is not retried. No fabricated Panda joints, cloned hands, shadow robot, slot substitution, state-mask assumption or source images are permitted. B3 success cannot be reported as GR00T+Go2 closed-loop manipulation. This baseline establishes the reference executor and the size of the genuine input-adaptation gap without pretending to remove it.

## 9. Evaluation and decision outputs

Keep original source/target task metrics separately: first on-top, grasp/lift/place/release events; native first success; on-top AND static AND not-grasped at all41 samples28..30s; full32s interface-valid execution; raw/install/actual and safety. Gates/source hard stops are not automatically transfer success. Four developed cases do not prove generalization. Report exact per-condition results and null for unexecuted slots.

Deliver a gap matrix with separate evidence and next decisions:
- Expert round-trip fails: source time/TCP/actuator/bridge issue, not grounds for neural training.
- Native joint policy succeeds but direct EEF fails on matched conditions, with substantial same-input EEF-vs-FK(q) mismatch: action-readout/representation gap; joint-conditioned success is not EEF-only success.
- Joint and EEF both fail despite functioning positive controls: pretrained task/visual/state distribution gap is supported; capacity shortage and required training size are still unknown.
- WBC native controls fail: low-level reference/observation/controller capability gap, unrelated to raw-model task quality.
- Honest target input unavailable: a declared shared EEF/gripper adapter or native fine-tuning is needed for the intended target interface; the baseline does not prove projector-only is sufficient.

Propose, but do not execute, the smallest subsequent adaptation: pristine initialization, native flow-matching objective first, preserve VLM initially, compare limited interface tuning against standard projector+DiT only if warranted. Do not automatically reopen old customized losses, recoveries or multi-task data collection. The next user/PI decision follows the actual baseline report.

## 10. Execution, accounting and delivery

Use one resident original model on Corelab with the existing isolated source and WBC environments. Validate actual GPU allocation; do not assume three agents imply three available GPUs. No model reload per episode, duplicate server or H200 queue. Stock inference dtype/S4 and measured15Hz model clock remain; simulator may wait for inference and must be labelled NON_REALTIME. Identical raw outputs can be reused but never called fresh inference.

Nominal maximum new source starts18: JOINT4 + FK(q) replay4 + expert bridge4 + conditional corrected EEF4 +2 short fixtures. Sixteen33s source tasks at100Hz plus two2s fixtures=53200 integrations,10640 control steps. Native new learned requests240 +conditional240 +up to32 wrapper calls=512,2048 four-step heads. Some comparison calls are reusable and actual totals may be smaller.

Maximum WBC starts14: six10s fixtures +eight32s task replays. At50Hz native control/200Hz integration:15800 controls,63200 integrations, normally31600 arm/dog actor calls. Actual initialization/reset/pre-roll and any extra actor calls are charged, not hidden. Combined nominal32 starts,116400 integrations,26440 control steps.

Hard caps:36 physical starts (at most4 same-condition technical replacements, zero valid-quality retries);640 GR00T requests/2560 heads;20000 WBC control ticks/80000 actual actor calls;150000 total per-environment integrations;100000 control/gripper dispatches;20000 RGB frame pairs;8 new GPU-hours total;64 CPU-core-hours;40GiB new local outputs/cache. Use existing model/assets; at most5GiB missing non-model dependency/asset bytes, no new model checkpoint/platform download. Count failed and interrupted attempts and reservations versus completed rows separately. Training/backward/optimizer/statistics fits/new adaptation data production/target VLA queries/real hardware=0.

Preserve PI0280 actual304queries/1216heads/7starts/source13200integrations/WBC2212integrations and all older ledgers. Unused old allowances do not transfer. HF bulk upload/readback, long retrospectives and full duplicated videos are deferred; local checksums, reloadable model identity, actual scripts and representative successful/failed recordings are required. Never block a scientific decision on HF completion.

Final report must give actual native-source closed-loop results, EEF readout/command replay differences, exact WBC code defect or remaining capability gap, before/after fixture receipts, target-replay outcomes if allowed, and the unresolved input-contract boundary. `NATIVE_CLOSED_LOOP_EXECUTED` is not `TASK_POLICY_USABLE`; scripted replay is not a learned controller; target closed-loop remains explicitly unexecuted. selected/deployment stay NONE. No automatic follow-on goal or training.

State machine: adopt -> saved_evidence_and_dual_readout -> CPU_ABI_tests -> source_native_JOINT_and_fixed_replays / WBC_native_isolation_in_parallel -> one_proven_WBC_semantic_repair_and_same_fixtures -> conditional_target_reference_panel -> gap_report_and_canonical_readback. A gate success grants the next listed stage without another signature. End at completion, a concrete irreducible lane blocker, or finite caps; continue independent unblocked work, never manufacture a pass.

## 11. Audit basis and limits

Actually read: PI0280 final report and terminal comment; supported input matrix; source.py; wbc.py; SOURCE_TABLE and the first condition's detailed SOURCE_SUMMARY; repository Stage21 reference install/configuration helper (navigation, not proof of bound private runtime); upstream DROID README and main_gr00t.py including decoded joint execution; upstream RoboLab README. Refreshed branch and comments after terminal, no newer ledger comment at review. No SSH, new neural/physics execution, full private native module trace, full video review or full source-array replay was performed by the reviewer. The branch/projection hypothesis and prospective repairs remain to be tested. Source-derived facts are not silently replaced by outside benchmarks.

Startup: preserve all predecessor pauses, use genuine original GR00T on Corelab, first compare source-native joint execution with predicted EEF and FK(command) replay under real feedback; isolate the returning-path progress and actor-orientation chain in the actual delivered WBC, apply only one proven nonlearning semantic repair, and verify the same original fixtures before target reference controls. No training, fabricated target state, new task platform or hidden success takeover. Deliver separate execution/quality/gap evidence and stop for the minimum necessary adaptation decision.
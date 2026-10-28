# PI0275 — Physical object-transfer teacher and GR00T task-data pilot

Date: 2026-10-06 SGT.
Authority: **PI0275_OBJECT_TRANSFER_PHYSICS_AND_DATA_PILOT_GO_V1**.
Research charter: `GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1`, fixed commit `ed398fee17151cfec47864fed9d48ce149b28d61`, Issue #11 direction canonical `5994405991`.
Reviewed implementation: `882968039afa1123ee3004f68b0fe29b50a71237`; final review refresh `d4f010869cd7a215d381bebadf3114b7fc5aee52` adds only PI0274's saved main32 observation. This protocol and `reports/PI0275/design/REVIEW_AND_BUDGET.json` acquire their immutable release SHA through the new Issue #11 canonical.

## 0. Scientific decision and charter alignment

GO for the **first bounded, actually simulated object-transfer and training-data experiment**, not another geometric GR00T fine-tune. Implement an independent actuated tool environment, execute a fixed scripted task teacher using actual contact dynamics, record object results and synchronized observations/commands, and validate the new gripper/data contract. The goal must end with an evidence-backed DATA_READY or a specific bounded failure, not merely source files or CPU fixtures.

**No learned-model construction/forward/backward, optimizer update, WBC run or real-robot actuation is authorized.** A successful teacher/data pilot prepares a separate GR00T learning decision; it does not automatically launch it.

| Required charter field | Binding answer |
|---|---|
| object_task | Select the language-named red/blue rigid object, grasp and lift it, move it into the left/right tray, release and withdraw; actual object/contact evidence defines success. |
| high_level_contract | Future learner inputs remain body/wrist RGB, measured task-space EEF/gripper and language; labels are commanded XYZ + row-rot6d + parallel gripper opening, H40 at 20 Hz. No robot joints, teacher phase, object truth or success signal in policy inputs. |
| frozen_wbc_identity | **NO_WBC_RUN** in this goal. The senior-supplied WBC remains frozen and unused. A later composition experiment must bind its actual version; absence of a newer release is not a pilot blocker. |
| task_data_provenance | Independent physics-driven tool and privileged offline task teacher; no target-robot teleoperation and no filtering by target-WBC success. |
| single_intervention | Establish physical object interaction and its truthful task supervision. This is a feasibility/data-source pilot, not an ablation proving algorithmic superiority. No simultaneous learning-method search. |
| task_evidence_and_controls | 64 fixed teacher slots, physical engine checks, four fixed open-gripper control replays, raw object/contact/task outcomes, native codec and label/gate checks. |
| budget_and_stop | Section 10 fixes starts, simulator ticks, teacher updates and renders. Stop at the first applicable native/DEV/data failure boundary or complete all registered lanes; no outcome-driven replacement. |
| claim_boundary | At most: a limited synthetic tool-domain task-data source is viable without the target WBC. NOT learned manipulation, mobile operation, real transfer, multi-task autonomy or cross-embodiment success. |

## 1. Predecessor and ownership: do not disrupt PI0274

PI0274 retains its original canonical `5990552027` and release `495b17078cb57fdbd47d8257d33202c57beb058e`. Its 5120 training, exact-resume canary and two complete Trainer readbacks are complete. The latest read saved main observation is **2026-10-06 00:54:48 SGT**, 32/64 episodes: target-pose 16/16 first completions, contour 8/16; reliable hold and full CPU trajectory reconstruction remain pending. This is repository evidence, not reviewer SSH/PBS telemetry.

The same Codex main owner first adopts the actual existing logs/locks, continues PI0274's remaining finite evaluation and closeout, and preserves its result regardless of PASS/FAIL. Isolated PI0275 CPU implementation can proceed without editing live sources. Start PI0275 native dynamics/rendering only after the relevant PI0274 GPU workers/locks have finished and resources are available. Do not wait for a favorable PI0274 quality score or a helper signature; neither is a pilot prerequisite. Durable archival may overlap if old evidence and unique recovery sources stay protected. Do not cancel healthy jobs, introduce a competing automated goal, duplicate qsub, reset budgets, add a permanent branch, restore PR77 pointers or merge unrelated source.

Use isolated `scripts/pi0275_execution/`, `reports/PI0275/`, and PI0275 runtime/data roots. Preserve the prepared source/history; import exact copies or pin dependencies by actual `__file__` and SHA256, rather than loading a moving same-name helper. Publication is not a runtime-start receipt.

## 2. What the review actually found

1. PI0274's complete fixed local-field comparison has useful but mixed results. In M1, center XYZ goes 1.902 to 1.215 mm and pure-rotation corrective directions 466/576 to 576/576, while center SO3 rises .004246 to .005070 rad. M3 center XYZ goes 2.352 to .791 mm, direction 538/576 to 576/576, but SO3 rises .002365 to .004172 rad and 63/96 finite-scale matrices still have spectral radius above one. These are finite-sample diagnostics, not stability certificates or object-task results. They do not justify another automatic geometry-only goal.
2. Object-task preparation has a callable teacher, URDF generator, scene proposal, scorer and episode-window code, but **zero actual object-task physics/teacher episodes** and no admitted dynamic environment at the reviewed SHA. Synthetic fixtures cannot establish actual grasping.
3. The actual old StateActionProcessor probe gives normalized gripper values -1000001, -500001, -1, 499999, 999999 for raw values 0, .25, .5, .75, 1. Old q01=q99=.5; an inverse roundtrip error of zero does not make this a suitable learning scale. The new task must have its own TRAIN-derived gripper contract; PI0274 statistics remain untouched.
4. Current task labels/state use row-rot6d; `TrajectoryChunkV1.from_rot6d` and adapter convenience input decode columns. Enter the canonical representation via row -> SO(3) -> xyzw quaternion, then let the existing preview emit rows. Do not pass row-packed values straight into the column helper. This is a new-route binding requirement, not a claim that PI0274 used that wrong route.
5. The existing adapter selects one gripper command at a given timestamp. A new consumer must continue consuming the separate gripper sequence with the control clock. A one-time value at trajectory installation is insufficient.
6. The prepared teacher's 27 mm TCP offset fixes a previously impossible table clearance: static clearance is 2 mm, overlap with the 30 mm object's height is 28 mm. Four-centimeter tray walls now contain the 30 mm object. These are necessary geometry checks, not physics success.
7. Existing logging/NPZ acceleration patches are preparation only. They do not establish training throughput and are not a reason to load a model in this pilot.

Actual source paths, blobs, arithmetic and read limits are recorded in the companion review JSON. Historical handovers are explanatory context only; older qualifications or proposed WBC-dependent data filtering do not override the current charter.

## 3. A0 — freeze the existing proposal, not a new task search

Use the exact 16 layouts and 64 ordered slots generated by `scripts/object_transfer_proposals.py` at the reviewed SHA and its saved `SCENE_AND_SPLIT_PROPOSAL_DRAFT.json`. Verify the saved file against that generator; do not substitute another layout after seeing outcomes. Freeze a PI0275 scene card containing all copied values before any scored episode:

- Families 001-004 DEV (16 slots); 005-012 TRAIN (32); 013-016 TASK_VAL (16). All four color/tray commands of a family stay in that split.
- Each scene has both red and blue objects, dimensions .04 x .03 x .03 m and mass .05 kg; red/blue positions alternate across layouts. Unique instance IDs are not different shape categories.
- Table bounds [-.6,-.6,.60] to [.6,.6,.65] m; trays have .16 x .16 m interior, .04 m wall height and .008 m wall thickness, with the exact per-tray 3D bounds from the proposal. Use the tabletop as the common tray support floor; do not add overlapping coplanar collision floors.
- Initial TCP [0,-.38,.85] m, rotation diag(1,-1,-1), actual inner gripper width .08 m. New tool TCP/frame identity is `PI0275_TOOL_PALM_FACE` / `PI0275_OBJECT_TASK_WORLD`, with explicit policy position anchor [0,0,0] m. It is not the physical X5 tool identity.
- Body camera: eye [0,-.50,1.0], look-at [0,0,.70], horizontal FOV55 deg. Wrist camera: tool-local eye [.06,0,-.03], look-at [0,0,.027], horizontal FOV65 deg. Both RGB uint8 96x96, square aspect. Body camera is fixed in this first source-domain pilot; wrist follows actual tool pose. Do not claim moving-camera or mobile-robot validation.
- Exact four instructions are the existing `Move the {red|blue} object into the {left|right} bin, release it, and withdraw.` strings. Left/right refers to the frozen body view/task frame; no target highlighting dependent on the command.
- Seed for each slot is first8 SHA256 bytes of `PI0275:OBJECT_TRANSFER_V1:{episode_id}`, interpreted big-endian modulo2**31. The scenario itself is fixed; any engine randomness is recorded. No noise/seed search.

Freeze source, scene, tool, controller, clock, contact mapping, guard, scorer and dataset rules into an execution manifest before native starts. Native load/API-binding work is authorized; it must not be disguised as uncounted simulation. Resolve existing private/runtime dependencies by manifests, not a full download of historical model weights. Do not migrate the WBC or GR00T stack to a different simulator for this pilot.

## 4. A1 — actual finite-actuation physics and two native checks

Implement the new environment in the existing IsaacGym/PhysX stack. Use CPU physics and CPU state/contact access (`use_gpu_pipeline=False`, `physx.use_gpu=False`) with the available GPU only for graphics. Single environment, fixed dt=1/120 s, two PhysX substeps, solver type1, eight position/two velocity iterations, gravity [0,0,-9.81]. Freeze actual engine version, settings and unsupported fields; do not silently replace unavailable pair-contact sensing by aggregate net-force guesses. CPU physics is chosen to make body-pair contact evidence inspectable, not as a new claimed control method.

Use `assets/object_transfer/draft_tool.urdf` generated by `object_transfer_tool_asset.py`: fixed anchor, six root actuation DOFs, palm and two finger DOFs; explicit masses/inertias/collision geometry. Native inventory must match its nine links/eight DOFs, TCP and measured-opening relation. Use DOF effort mode, not unconstrained position servo gains hidden in an SDK default. Controller sees only its command and measured tool state; it never reads the task target or teacher phase.

Fixed virtual-tool controller design for this first pilot:
- Translational task-space PD Kp=600 N/m, Kd=40 N s/m. Gravity compensation is computed from the known tool link masses/Jacobian only, never a privileged attached-object flag. Clip the **total** root translational generalized/world force after compensation to norm20 N; per-axis caps also respect the URDF.
- Angular left-error PD Kp=4 N m/rad, Kd=.3 N m s/rad. Map bounded world torque (norm <=2 N m) through the actual rotational Jacobian, respecting each DOF's limit. Bind the initial downward orientation with a consistent -pi roll branch; do not turn a pi/-pi representation wrap into a full revolution.
- Finger PD Kp=2000 N/m, Kd=5 N s/m per finger, force bounded by10 N per finger. Desired q_left=q_right=commanded_width/2; commanded full-width speed <=.04 m/s. Root desired translation speed <=.10 m/s, angular speed <=.5 rad/s.
- Command rate limits are not claims that actual dynamics never overshoot; record actual speed/forces and saturation. Do not clamp actual state by calling reset/state setters. Gains, limits and solver settings above are fixed before outcome observations; no within-goal sweep.
- Tool/object/support friction .8, restitution0; per-shape contact_offset .001 m and rest_offset0. Record their actual native values. Contact skins do not change the geometric OBB dimensions.

Initial placement/reset may set states once per declared start. During simulation, no object/tool pose assignment, welding, parenting, magnetic attachment or scripted object force is allowed. Runtime setter guards must log and reject state-setting APIs after t=0; legitimate native dynamics updates are not setters. The learned policy will not receive virtual root/finger joint labels: these DOFs are only this data-source actuator.

Contact evidence must identify actual body pairs and whether the left and right finger contact the **selected object**, not merely have nonzero net force. Bind native contact field names, body-ID mapping, signs and units with a fixture. Preserve raw pairs/normal information; do not invent tangential force measurements. Prefer actual pair records plus a force/impulse-derived normal-force floor1e-3 N, with actual substep conversion documented. Reject a missing or unverifiable channel as NATIVE_EVIDENCE_INCOMPLETE. Penetration is native separation when available; otherwise use disclosed normalized-axis OBB intersection depth for the same primitive boxes, validated on CPU positive/negative cases. Do not call proximity alone physical grasp.

Allowed contacts: intended finger-target grasp, object-support/tray contact, and modeled internal gripper contact. Dangerous contacts include tool-table/tray collisions, tool-nontarget contacts, unintended object collisions, or any measured penetration >.002 m. All recorded unsafe events remain; no smoothing removes them.

Two native fixtures, each <=10 s, are separate from all task slots:
1. Free-fall/contact fixture: a declared cube starts .10 m above its source resting position; gravity, pair-specific support contact and settling are observable without any pose setter or attachment. Preserve the whole trace.
2. Tool-drive fixture: in clear space test measured opening .08 -> 0 -> .08 m, a20 mm translation and a small .1 rad orientation command, with the same bounded controller. Check native TCP/FK, angular branch, applied efforts, and sensor/graphics clock. This is not task success or WBC qualification.

Native checks must establish the declared pipeline and usable evidence, not a secretly tuned high-performing controller. Concrete implementation/sign/unit defects may be narrowly repaired with failed source/attempts preserved; physical gains/geometry/friction/teacher/timeouts cannot be changed after native outcome inspection under an 'engineering' label. Failure to establish the source is a legitimate stop, not permission for a replacement simulator/configuration.

## 5. A2 — fixed teacher episodes, actual clocks and guards

Reuse the reviewed `object_transfer_teacher.py` semantic logic: PREGRASP -> APPROACH -> CLOSE -> LIFT -> TRANSFER -> LOWER -> RELEASE -> RETREAT, measured-state transitions and timeouts4/4/2/4/6/4/2/4 s; GRASP_TCP_OFFSET_M=.027. Keep the source privately versioned when binding this new environment. No regrasp/search loop, no target/phase input to a future learner, and no task completion by merely reaching teacher DONE.

Each normal slot has a30 s task horizon and2 s additional recorded tail. Execute 3840 physics ticks to t=32 s, retaining3841 physics states including t=0 and32. Teacher updates occur before each tick at t=k/120, k=0..3839, from the measured state at that time. Capture synchronized RGB/RGB/state/command every sixth tick, k=0,6,...,3834: **640 records at0..31.95 s**, not641. This provides601 possible H40 anchors at0..30 s without padding. Actual teacher commands at120 Hz are retained as well as their20 Hz samples.

A task teacher cannot know future contact-triggered stages before they happen. Therefore:
- Online execution uses an explicitly versioned **PI0275_TASK_COMMAND_GUARD_V1** on the current command: finite pose/SO(3), declared commanded opening[0,.08], monotonic clock, the above command speed/width bounds, current TCP workspace x/y[-.55,.55], z[.65,1.10] m, and tracking gap <=.05 m/.20 rad. Native contact/penetration checks apply at every tick. A guard reject is a real stop/failure, never a command to install anyway.
- The original **full-H40 TrajectorySafetyGateV1** is run OFFLINE on recorded future command windows with the measured anchor. Keep its original numerical thresholds, via canonical SO(3)/quaternion construction and correct new frame/TCP. Its result is training-data compatibility, not a claim the online teacher foresaw the future.
- Do not pad a current setpoint40 times, run that fake chunk through the old gate, and call it proof of the future teacher trajectory. Do not label future achieved poses as commands.

Measured full opening comes from native finger inner surfaces. Permit at most1e-5 m numerical boundary tolerance around[0,.08] in the **new** measurement parser, retaining unmodified values and documenting them; larger violation is a valid limit failure. Commanded labels remain strictly[0,1]. This expressly replaces the preparation parser's exact-boundary assumption only for native measurements, not old records or PI0274.

After teacher failure, no resume into a new attempt or automatic reset. Mark future teacher labels invalid. Where safe, retain bounded last-safe-tool/opening fallback and continue passive recording to32 s; record fallback explicitly, not as valid expert commands. Unsafe/numerically broken simulation may stop immediately; missing tail and actual consumption stay visible. Teacher DONE is not early stop: continue actual terminal commands and physics through32 s; score remains the fixed interval below.

At t=0, verify the four commands in a family see the same physical scene/initial state with no target-dependent rendering. Keep camera geometry and segmentation diagnostics out of the policy observation. Initial visibility/occlusion issues are data-validity findings, not permission to replace a family or recolor the targets. The task is only this small, fixed tool/cube/view domain; static-camera data do not establish mobile-view robustness.

Run the16 DEV slots first, once each, in the existing order. Continue to the frozen32 TRAIN plus16 TASK_VAL slots automatically if: native integrity passes, at least12/16 DEV tasks succeed, at least3/4 per instruction, and no dangerous contact/penetration. These are predeclared pilot-readiness gates, not population guarantees. Otherwise close with DEV_FEASIBILITY_FAIL or an integrity-specific stop; preserve all observed cases and mark the48 unstarted slots DEFERRED, not48 quality failures. No new PI/helper signature is required for a passing continuation.

## 6. A3 — object scoring and four anti-attachment controls

Retain the actual task requirements of `object_transfer_task_score.py`:
- target bottom lifts at least80 mm from its original support and simultaneous bilateral target contact is retained at that lift for>=.3 s;
- the commanded tray contains the **whole** target OBB, not only its center;
- measured opening>=70 mm and no target-tool contact throughout28..30 s;
- measured TCP-target distance>=100 mm and target linear speed<.020 m/s at every final tick;
- nontarget displacement<10 mm, no nontarget lift, and no dangerous contact/penetration or online guard rejection in **all recorded data**, including the tail.
The final window has241 physics samples at120 Hz. A first arrival or favorable middle interval cannot replace it.

Explicit pre-physics scorer refinement (this is a new protocol, not a post-result repair): physical resting contact is allowed up to the already stated2 mm penetration bound **only at the bottom support face when actual contact with the commanded tray's floor/tabletop is present**. Its top and lateral containment faces retain1e-5 m numerical tolerance, not2 mm expansion. Require valid actual support contact for this floor exception. Preserve the preparation's strict whole-OBB result as a separate diagnostic column. This avoids conflating tiny compliant floor penetration with being outside a tray, without allowing wall/rim escape or removing the existing penetration failure. Use an isolated PI0275 scorer wrapper and CPU cases for unsupported-below-floor, supported tolerance, excessive depth, and wall/top escape; do not alter the old draft's historical source or any PI0274 scorer.

Truth for task scoring remains evidence-side. A future actor cannot read teacher stage, body-pair ground truth, target poses or scorer success as control. This pilot does use a privileged teacher for data generation and says so.

Four fixed negative physical replays: use the four primary family001 recorded root-command schedules, same initial scenes and same controller, but keep gripper command fully open throughout. No new teacher calls. Replay recorded safe fallback holds too if a parent teacher failed; do not choose a different successful parent after looking at results. All four traces must retain their identities; they must not show a successful sustained grasp/transfer. Report whether each parent was a complete positive example, so a trivial replay of a failed parent is not advertised as a matched causal success/failure comparison. A control that moves a target through ordinary contact is reported, not forced to remain static. These controls supplement setter guards and native evidence; they do not substitute for them.

Negative replay data never enter the positive BC bank or normalization. All64 primary slots, all four controls, two fixtures, exclusions, stops and technical attempts have separate manifest rows. Valid task FAIL is not a technical retry and is never dropped from its split denominator.

## 7. A4 — complete task episodes -> H40 -> a new TRAIN-only gripper contract

Keep raw120 Hz state/contact/command evidence; use640 actual synchronized20 Hz records per full episode. Split before slicing. A full successful episode supplies at most601 windows; potential maxima are38464 across64 primary episodes and19232 across32 TRAIN episodes. These are overlapping windows, not independent tasks. No cross-episode splicing, predicted teacher phase, repeated-tail padding or copied nominal images.

Policy allowlist: `body_rgb`, `wrist_rgb`, `state_v1`, `language`. State is measured XYZ relative to the explicit anchor, row-rot6d orientation, and measured opening/.08. Label is the recorded commanded pose/gripper at the fixed future times. Metadata, tool joints, object geometry/poses, contact status, phase, split and success are evidence only. Record each window's phase spans for diagnosis without passing them to the learner. Report potential ambiguous identical-input/different-label examples rather than silently adding phase/history features.

Use `object_transfer_preparation.episode_window`/`check_window` through source-bound, narrowly adapted copies (real source name, native numerical measurement tolerance and contact metadata separated). In particular row -> matrix -> xyzw -> canonical chunk -> row WBC preview must pass nonidentity rotation tests. Parallel gripper consumption is tested at control times using saved data, with no WBC calls. This does not claim the future native WBC uses this tool's time-based controller.

All completed physics and label-valid windows are archived. The positive BC bank is explicitly the successful, safe TRAIN teacher episodes whose windows have complete future commands and pass the unmodified full-H40 compatibility gate. Failed episodes remain a failure/recovery evidence bank, not automatically successful BC targets. This declared independent-tool task-quality selection is allowed by the charter; target-WBC outcome is absent and never a filter. DEV/TASK_VAL and the four negative controls cannot supply statistics or training membership.

New scale rule, fixed before seeing learned performance:
1. Copy the qualified pose/rotation processor and effective ROT statistics (old statistics SHA256 `0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9`). Do not edit old files or apply a new processor to old weights and call it an old-policy result.
2. For the gripper **state and action separately**, use measured/commanded raw minima and maxima from the actual positive TRAIN20 Hz records, counted once per source record, as the new affine low/high (`q01/q99` storage fields). This is an explicitly declared TRAIN min/max calibration, not an assertion those extrema are empirical1%/99% quantiles. Save true quantiles too. Keep no-clamp semantics. The observed range must be>=.10 in normalized-opening units for each channel; action must actually contain opening and closing commands. A failed range/coverage gate stops DATA_READY, not a hidden switch to synthetic endpoints.
3. Never fit pose/rotation statistics, normalizers, clipping or pixel settings from DEV/TASK_VAL. The other pose fields stay unchanged in this pilot; report their normalized ranges. Any remaining pose-scale incompatibility is a named source-contract result for the next decision, not permission to refit everything.
4. With CUDA hidden and no `nn.Module.forward`/optimizer calls, run the actual StateActionProcessor/codec on unique processed inputs and their complete windows, cache deterministically by source+language+processor/statistics identities, and verify physical roundtrip: XYZ max<=1e-5 m, SO3 max<=1e-4 rad, normalized gripper max<=1e-6; valid masks for all10 task channels and unchanged padded-mask semantics. Exact unchanged inputs may reuse proofs. Report all source/decoded gate disagreements; no smoothing or threshold change.

The resulting version is a **new task-data processor candidate**, not an automatically approved training recipe or a trained model. An active gripper mask and exact inverse map alone are insufficient; actual nonconstant task exposure, magnitude and physical meaning must all be shown. Do not build GR00T/vision towers to claim CPU preprocessing is a full neural check.

## 8. Readiness, interpretation and the next decision

`PILOT_DATA_READY` requires all applicable source/contact/scorer/codec checks, completed64 primary slots with honest outcomes, >=12/16 DEV, >=24/32 TRAIN and>=12/16 TASK_VAL task successes, respectively>=3/4, >=6/8 and>=3/4 for each instruction, and zero primary dangerous-contact/penetration failures. The four open-gripper controls must not produce sustained grasp/task success. All declared success and failure traces must be durable; all BC-admitted windows must preserve code/gate meaning and the new TRAIN-only gripper range must pass. These are finite pilot gates, not a claim that75% is a general manipulation standard or that every future dataset needs these thresholds.

Report execution status, teacher task success, contact integrity, data validity, codec qualification, and archive status as different fields. Meeting a count cannot compensate for a falsified contact source or invalid label. A completed but poor teacher pilot is `VALID_TASK_SOURCE_QUALITY_FAIL`; missing/corrupt evidence is `INCOMPLETE`, not an invented failure of a learned policy.

Successful stop: **PI0275_OBJECT_TRANSFER_PHYSICS_AND_DATA_PILOT_COMPLETE — STOP_FOR_GROOT_TASK_LEARNING_DECISION**. Deliver the exact positive TRAIN bank, full failure/VAL banks, frozen new gripper candidate and an evidence-based next-learning recommendation. The next learning decision must choose a fixed initialization, data schedule, loss/precision and finite update/evaluation budgets; this protocol does not spend a single update in advance.

Failed stop: preserve the exact blocked contact/teacher/visibility/codec stage and measured reason; identify one needed correction. Do not replace the task with geometric reaching, do not add more random trajectories, and do not make an automatic new source/seed/gain loop. Local physical success is not a learned policy; a free tool is not Go2 locomotion. The senior's WBC training is not our new contribution.

## 9. Efficient execution and full closeout

Reuse already verified CPU definitions and hashes; do not repeat old model archives or PI0274 predictions. Allowed engineering work includes deterministic array caching, incremental append-only journals, buffered physical/RGB output and bounded CPU preprocessing workers with ordered results. These must preserve raw input/label bytes and source identity. Do not apply proposed training acceleration patches to a healthy old trainer; there is no learner here.

At each actual start, persist slot/seed/source/config IDs and counters BEFORE execution. Save actual ticks, contact records, commands, teacher decisions and image/state timestamps in append-only or atomic shards. A recording failure cannot erase consumed physics/render/teacher work. Same-slot technical reconstruction may use at most four replacement starts and the hard remaining budget; all partial attempts remain. Never restart a valid failure, and do not demand impossible bitwise native-physics resume when a fresh technical replay is explicitly logged. Same semantic source repair is allowed; no gain/shape/scorer/teacher-outcome retuning under the technical allowance.

The owner continues automatically through native checks -> DEV gate -> fixed TRAIN/TASK_VAL -> fixed controls -> offline windows/processor -> independent CPU object-score reconstruction -> videos -> durable archive -> Issue #11 closeout/readback. No optional helper becomes a mandatory approver. Do not stop at CODE_READY, a positive fixture, or the first successful lift.

Required artifacts: frozen source/scene/tool/control cards; START/attempt ledger; raw physics-rate trajectories and pair contacts; all synchronized RGB/state/command episode files; primary/control task table with actual versus commanded/gripper quantities; window membership and rejected-window table; TRAIN-only statistics candidate and actual codec/gate receipt; source-language contrast/visibility/exposure summaries; independent CPU score reconstruction using raw arrays (not merely reloading verdict JSON); complete all64 task outcomes or explicit deferred rows; final budget table. Use actual object rotations, corners and velocities. Shared math utilities or scorer components must be disclosed; do not call a second invocation of the same scorer fully independent.

Generate videos from saved frames only: family001's four positive-teacher slots and four control counterparts, including failures, as applicable. No extra rerender/model call for a better video. Preserve raw data, code/cards/statistics, tables and videos in the existing project HF archive location under a new PI0275 prefix; fixed-revision whole-file AND member hashes/readback are required for new artifacts. Old readbacks are reused. Keep source files until durable copies exist; no sweeping cleanup.

## 10. Budget — all physical preparation is counted

| New work | Planned maximum | Hard limit |
|---|---:|---:|
| Primary teacher slots |64|68 starts including up to4 same-slot technical restarts|
| Native fixtures |2 x10 s|2 scientific conditions|
| Open-gripper controls |4 x32 s|4 scientific conditions|
| Total starts |70|74 including technical restarts|
| Primary physics ticks at120 Hz |245760|included in total below|
| Control physics ticks |15360|included below|
| Native-fixture physics ticks |2400|included below|
| Total simulator ticks |**263520**|**278880**|
| PhysX substeps (2/tick) |527040|557760|
| Live task-teacher updates |245760|261120|
| RGB pairs (two images per pair) |**43920**|**48000**|
| Learned model/vision forwards, gradients, optimizer updates |**0**|**0**|
| Target-WBC / dynamic robot / hardware runs |**0**|**0**|

Physical tool simulation is expressly authorized within the limits; 'no dynamic robot' does not mean no object/contact physics. At120 Hz one32 s episode has3840 ticks and3841 states, not3841 calls. At20 Hz observations cover0..31.95 s and number640. The planned maxima include two native fixtures and four negative replays, unlike the earlier64-only preparation budget. The15360-tick technical reserve corresponds to at most four32 s starts, not new task conditions. Extra RGB allowance is for bounded actual technical/native acquisition, not a camera search. All counters are cumulative, including failure/in-flight starts. CPU replay of saved teacher/score arrays is separately counted as replay computation, not new task production or hidden neural calls.

No periodic monitors, mail changes, new cloud services or competing owners are requested. Resource availability is checked from actual runtime, not inferred from GitHub. A health/resource stop preserves completed work and reports what remains; it does not authorize duplication or budget reset.

## 11. Final boundaries and startup

Existing geometric failures, collisions, old WBC PARTIAL and all original qualification rules remain unchanged. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE` throughout. The positive task-data source may later be used to train a VLA, but this experiment cannot nominate a deployment policy or claim the frozen WBC already handles the task.

**Startup prompt:** Read this PI0275 canonical, its immutable release and charter5994405991. Adopt and finish healthy PI0274 without changing its recipe. Reuse the fixed object-transfer preparation; implement and execute the bounded independent-tool physics/teacher/data pilot, with truthful paired contacts, 64 fixed primary slots, two native checks, four open-gripper controls and TRAIN-only gripper/codec validation. Do not stop at CPU preparation; close all actual task outcomes, records, CPU/video/HF and Issue #11. No GR00T learning or WBC/hardware calls, no failed-case replacement, no budget reset; selected/deployment remain NONE.

# Full-body Teacher pilot failure audit: approach does not progress

Date: 2026-10-15. Fixed evidence/code baseline: `7a05ad279349e59e89e51920beec187ac08cdc88`. Status: **SOURCE_AND_RECEIPT_AUDIT_NOT_NEW_EXECUTION_GO**.

User asks whether four unsuccessful fixed pilots mean Teacher-data production is blocked. **Yes: successful full-body EEF13 demonstration production is blocked at the Teacher/task-reference/execution stage, not at GR00T optimization or file storage.** The V2 worker can record complete authentic task episodes; it has not produced successful task demonstrations. Existing fixed-base datasets and M2048 are not invalidated. Do not change WBC gains/costs/weights/solver, reopen R4B control diagnostics, or reinterpret unsuccessful records as expert successes.

## 1. Evidence and limits of this audit

Read at the fixed commit: FIXED_PILOT_RESULTS.md; three per-case CPU/video receipts; task_native_worker.py; task_plan.py; task_plant.py; task_state.py; base_intent.py; scripts/pi0291_arx5_data/teacher.py and config.py; scripts/pi0294/teacher.py and scene.py; original fixed-base scene initialization.

The analysis is from GitHub code and published receipts. No SSH, new physics, model inference, training, native replay, or private full-array recomputation occurred here. The supplied local Windows MP4 path is not accessible to this reviewer; no claim of visually watching it. The receipts identify actual saved-RGB playback, which is different from this reviewer opening the video.

Original per-case source versions remain: W0/T2 `5e361f6fa1f6226c6646ba47f3d8baa9e1455dc5`, W1/T1 `5e706da87d400672779a7f69f5f77aff5188a27f`, W1/T2 `12cf868ca44e81f9dbd7840cd6770efeaaaab718`. Before executable diagnostics, owner must use the actual case source/plant and compare relevant source members with this integration snapshot.

## 2. What actually failed

| Case | Actual result | Correct interpretation |
|---|---|---|
| W0/T1 V1 | 33 steps / 0.066s, gripper startup native safety FAIL, no task intent | Startup failure of V1; not a full T1 Teacher behavior trial |
| W0/T2 V2 | 32s, 641 actual image/state pairs, 640 installed transactions, no stage transition, no actual robot-object contact/push | Complete recorded failure stuck before interaction |
| W1/T1 V2 | 32s, same full capture, no stage transition, no sustained grasp/lift/carry | Complete recorded failure stuck before grasp sequence |
| W1/T2 V2 | 32s, same full capture, no stage transition, no actual push | Complete recorded failure stuck before interaction |

All three V2 cases have `teacher_events=[]`, `final_stage=approach`, zero object horizontal displacement and lift. Original source-time, raw state/label and installed-reference audits passed; physics safety passed for these V2 cases. The immutable mixed-version pilot denominator remains0/4. Do not call this four identical Teacher-policy failures or a GR00T model success-rate measurement.

Recorded BASE maximum forward motion is approximately0.807cm for W0/T2,4.368cm for W1/T1,3.874cm for W1/T2. The moving cases require at least10cm. BASE soft intention was supplied; this is not evidence that the requested body motion was actually realized.

These failed records are valid failure-bank data, not successful behavior-cloning samples. Full-body successful TRAIN remains0; v4/FB13 remain not ready/not run. Native validation of installed14D targets proves reference transfer, not task reachability, solver consumption of each new reference, or execution success.

## 3. Source-proven task-layer mismatches

### 3.1 Old Teacher started from a nearly prepared approach state

The full-body worker directly imports the old fixed-base PickTeacher and PushTeacher. In `scripts/pi0294/scene.py::make_engine`, the old fixed-base initial TCP is set by IK to `[source_x, source_y, table_top + 0.10]`, downward-facing, unless overridden by the card. The PickTeacher approach target is initial cube center (table_top+0.02) plus0.08m, i.e. the same nominal height and XY. Thus the default old pick Teacher's approach can already be satisfied by its reset; its historical success did not demonstrate a large-distance native whole-body approach from an arbitrary home posture.

The native worker instead uses the actual delivered startup,5s warmup and actual TCP, then instantiates exactly those old teachers. It does not include a separately qualified task-level positioning phase before approach. This is a confirmed migration-assumption difference, not evidence that teleporting the actual robot to the old reset is acceptable. Full-body must retain actual free-root dynamics and physically approach.

### 3.2 Internal stage gates are inherited from the fixed-base executor

`PARAMETERS['position_stage_tolerance_m']` in the imported PI0291 config is0.004m.
- Pick: approach->align only when TCP position distance to its target <=4mm. It does NOT include PushTeacher's rotation condition.
- Push: approach->close requires position distance<=4mm AND geodesic orientation error<=0.04rad, approximately2.29degrees.

These are Teacher FSM transition predicates, not the final object-task success criteria and not native safety limits. Both teachers set stage_since but contain no approach no-progress classification, alternative task-level approach, or persistent task-station recovery. Remaining outside a predicate can consume the full32s with repeated approach plans.

Removing the old global WBC hold RMS prerequisite did NOT remove these inherited Teacher-level prerequisites. However, their runtime causal contributions must be recomputed on saved states. In particular, the old native0.07rad hold bias cannot be imported as if it were the observed orientation residual of the new push pilots.

Do not fix this by merely increasing4mm to a larger hand-tuned value or forcing the FSM to close by time. The approach/align handoff should be justified by actual pad-object clearance, safely reachable next segments and task geometry; the original final task and native safety predicates remain unchanged.

### 3.3 Observed failure is much larger than a marginal stage-tolerance miss

W1/T1 source object is configured at XY `[0.60,-0.08]`; it did not move. Its nominal approach target is approximately `[0.60,-0.08,0.54]` (actual settled cube z must be used for exact reconstruction). The recorded actual final TCP is `[0.20423658,0.00689849,0.49519848]`.

The X difference alone is about0.39576m. Therefore this case did not simply hover5mm away from a4mm gate. It remained far from the intended pregrasp target. Loosening a stage gate would not explain or safely repair that lack of approach progress.

The other common observation, approximately0.117m between the last installed horizon endpoint and final actual TCP, must not be called a measured same-time WBC tracking error. The old teacher plans at0.06m/s for offsets0..1.95s, so0.06*1.95=0.117m by construction when the target is farther away. These endpoints are moving local previews, not the fixed final task target, and refer to future times. The numerical coincidence is a useful code/measurement consistency observation, not causal proof.

### 3.4 The native execution cadence changed while the Teacher was reused

Each plan starts at `initial = actual_tcp`; its offset0 pose is exactly the measured pose. The native worker replans, publishes and obtains an installation ACK on every20Hz observation, then increments `next_step +=25` at dt0.002: only0.05s elapses before a fresh actual-pose-anchored H40 is installed. There are640 transactions in32s.

The old config still says H40/prefix8, i.e.0.4s per high-level prefix. The full-body worker is not actually executing one persistent eight-point prefix before replanning. At0.06m/s, a0.05s interval offers3mm of nominal local reference progression, compared with the entire0.117m preview.

This makes repeated recentering of the reference, inadequate persistent task progress, and alignment with asynchronous MPC/preview consumption **high-priority hypotheses**. It does NOT prove 'the first action is current pose so nothing can ever move': the WBC receives all future points and can use their preview. Installation ACK is not necessarily proof of a newly solved/consumed MPC policy for that reference. Read the saved request/installed/actual source times and any existing policy-generation identifiers; mark missing consumer identity UNKNOWN. Do not start another internal WBC redesign.

Two task-side execution designs can be evaluated as explicit alternatives later: persistent absolute trajectory progress with bounded lead and same-clock slicing; or a validated H40/prefix8 high-level transaction retaining20Hz actual observations and500Hz safety. Switching between them must update action/label timing consistently; it is not a silent configuration cleanup and neither is claimed fixed here.

### 3.5 BASE preference is not a station-aware whole-body Teacher

The worker chooses a station as `initial_mount + [0 or0.15,0,0]`, forms a yaw-frame proportional direction capped at0.06m/s, sets heading rate0, slews within the envelope and integrates it into native BASE soft poses. It does this independently of the Teacher's approach stage, grasp geometry or actual task progress.

This is a valid declared soft preference generator, but not yet a qualified task-aware station/path planner. It cannot assume WBC will realize0.15m translation merely because that number appears in the card. Recorded4cm-scale motion is below the10cm moving-task requirement and leaves W1 objects far from the arm's observed position.

Future task-level Teacher should retain a world/task-frame station objective, actual progress and scene-clearance constraints, then coordinate positioning, approach and contact stages through the SAME high-level native interface. Do not publish directly to leg velocity or edit WBC costs to force motion. Persistent references need bounded lead so an unachievable target cannot drift indefinitely.

## 4. Explanations not supported by current evidence

- Not GR00T training/model-capacity failure: no model ran in these attempts.
- Not a file archive failure: complete V2 actual states/images/labels and private readback exist.
- Not a late grip-strength/slip/release failure: no Teacher transition reached those stages.
- Not evidence that0.04rad orientation threshold alone explains T1: Pick has no such gate and W1/T1 is far from the position goal.
- Not proof that the whole-body WBC cannot perform any manipulation: this exact Teacher/reference scheme failed; the broader task remains untested with an adapted Teacher.
- No table-height mismatch was found in the inspected code: both current task_plant and inherited config use0.44m. Do not list this as an observed bug.
- Initial/target collision-free reachability and completed solver consumption for every target are not proved by a successful14D packet comparison.
- No conclusion from the local Windows video has been drawn by this audit.

## 5. Next useful work: one focused Teacher progress audit, zero new WBC study

Reuse the existing samples.jsonl, committed_intents.jsonl and planner.jsonl under the actual owner/private runmap. Do not rerun the four original fixed pilots. In the available read-only scope produce one small `TEACHER_APPROACH_PROGRESS.json`/plot bundle containing:

1. Every sample's actual TCP and actual object pose; exact stage target; position and angle residuals; each boolean transition predicate; stage age; windowed distance-to-goal decrease; number and duration of samples passing each predicate separately and jointly.
2. Distinguish ultimate stage target, H40 first point, H40 future endpoints and same-time actually installed target. Reconstruct physical command age and new-reference installation/consumption coverage without arbitrary time shifting. Quantify resetting the projected path vs retained task progress.
3. Actual BASE station residual, declared velocity preference, installed BASE poses, observed body motion and remaining arm-object distance. Mark absent MPC-consumer identity as UNKNOWN rather than attributing blame to the controller.
4. Using existing correct geometry in a private state copy, inspect task target reachability, tool/pad frames and swept access to the cube/table. This is task-path validity, not a WBC hold-RMS or controller-tuning campaign. A failed local IK solve alone is not proof of physical impossibility.
5. Source-bound CPU Teacher tests: approach initialization near and far from the cube; arbitrary actual observation perturbations; stalled inputs produce explicit diagnostics; progress persists across replans; task-relative geometry transitions cannot close when physically far away; source clock/endpoint/label accounting and both instructions remain correct.

From those results classify: `STAGE_PREDICATE_NONATTAINMENT`, `REFERENCE_PROGRESS_OR_CADENCE_MISMATCH`, `TASK_STATION_OR_GEOMETRY_INVALID`, `REFERENCE_CONSUMPTION_UNOBSERVED`, or a supported combination. The current source review establishes the above migration differences but does not establish a unique dynamic root cause.

## 6. Teacher-only development direction, not another frozen-pilot retry

Build a dedicated full-body Teacher/task adapter rather than copying the fixed-base FSM unchanged. Keep WBC frozen. Retain explicit task goals, bounded world-frame trajectory progress and BASE station objectives; observe actual position/contact; define safe geometry-based handoff from approach to align/contact; record a no-progress termination/replanning reason instead of spending an entire32s doing an unchanging approach. Do not add joint-level control, fit the historic0.07rad bias, weaken native safety or change final task success metrics.

New design must first show small, causal milestones in a separately versioned development run: getting closer to the declared approach region; safely reaching precontact geometry; actual contact; short real push or bilateral grasp; then complete task. A local contact or1cm push is a development milestone, NOT a successful full demonstration. Never force stage transitions simply because a timer expired.

Changing task-side reference cadence, planning progress, scene geometry or Teacher predicates is an explicit versioned intervention. Original V1/V2 results and0/4 fixed denominator remain untouched. A new approved Teacher-development/qualification series is different from retrying the old fixed protocol to search for a success. Use only current valid authority and actual remaining resource limits; this audit supplies no additional GO, restart, PBS, physical allowance, deletion or timeframe.

No new large TRAIN bank and no full-body multiworker production before one actual single-worker Teacher can complete each skill. Current multiworker acceleration would only create more approach-only failures. Existing independent M2048 diagnostics can proceed only under their already valid task/window; lack of new EEF13 successes is not a reason to fabricate BASE supervision or train FB13 on these failures as successful demonstrations.

The target remains a complete free-root Go2+ARX5 with real tasks, frozen WBC, honest physical evidence and learned EEF13. The corrective work is in OUR Teacher, task geometry and high-level temporal adapter, not in WBC servo design.

## 7. Closure

Evidence preserved: original four pilots0/4; three complete V2 failure trajectories;6131 private original files and separate closeout receipts; original model/data archives; selected/deployment NONE.

This publication only documents the source/receipt audit. It does not assert that a new Teacher implementation is complete, that any parameter change will work, that raw arrays were recomputed, or that a physical trial was performed. The first next scientific deliverable should be a time-aligned explanation of why approach never progressed, followed by a new-version task Teacher that demonstrates actual contact and complete behavior.

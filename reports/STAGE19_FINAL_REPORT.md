# Stage19 Final Report — Robot-Specific Dataset Materialization and Pilot Readiness

Date: 2026-09-01 (Asia/Singapore)

## A. Classification

`PASS WITH DEVIATION`

The reusable data contracts, actual on-disk LeRobot/GR00T fixture, official
GR00T dataloader/processor qualification, synchronization stress harness,
RecorderV1 replay, and pilot protocols passed. The retained deviation is the
D3 20 Hz metadata versus 25 Hz source-timestamp discrepancy, classified as
`FRAME_SUBSAMPLING_METADATA_DEFECT`. No physical collection was performed.

## B. Stage16 promotion

`NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`

Stage16 remains frozen and was not rerun. Its 28 native trajectories had 27
`REFERENCE_COMPLETE`, one expected boundary `TRACKING_ABORT`, zero falls,
invalid resets, timeouts, or true joint-limit terminations. All eight selected
progressive cases were `BASE_RELOCATION_LIKE`.

## C. 40-point representation

`40-POINT REPRESENTATION ADEQUATE ON TESTED STAGE16 DISTRIBUTION`

The existing Stage16 review remains the authority: case-level median/P95/max
of reconstruction position means were `2.43964e-5 / 1.98620e-4 /
2.07631e-4 m`, rotation means `5.976e-9 / 8.65439e-5 / 9.98223e-5 rad`, and
end-to-end position/rotation delta means had maxima `1.84976e-5 m` and
`2.15568e-4 rad`. This is not arbitrary high-curvature equivalence.

## D. Git

- base: merged Stage18 commit `237be6fc37c20cf44ccccf5f0cfbd3cce629760f`;
- branch: `stage19/robot-dataset-materialization`;
- origin: private `git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git`;
- reusable source/docs/tests only are staged for the Stage19 commit;
- test result: `49 passed in 0.65s` with `/usr/bin/pytest` on
  `corelab-5080-2`; local shell lacked pytest;
- local `compileall` and `git diff --check`: passed;
- final branch tip is recorded by `git log -1` and the Stage19 PR;
- D3 source, WBC archive/source/checkpoints, MP4/parquet datasets, model
  outputs, and credentials are not Git artifacts.

## E. HF auth

`AUTHENTICATED`

On `corelab-5080-2`, Stage10 `hf auth whoami` returned `leoyuer`. The token
was not requested, printed, copied, inspected, or stored.

## F. Actual GR00T

`SKIPPED_BY_STAGE19_SCOPE`

Stage19 does not rerun model inference. Stage17's existing custom N1.7 runtime
evidence remains separate: finite output and processor/runtime behavior are
not task performance evidence.

## G. GR00T/LeRobot materialization and dataloader

`PASS`

Six D3 V0 episodes produced 366 synchronized rows, six body videos, six wrist
videos, explicit source timestamps, desired action labels, calibration and
label sidecars, and official standard/relative statistics. The official
`LeRobotEpisodeLoader` and `ShardedSingleStepDataset` loaded all six episodes
and 132 valid windows. The processor path verified physical `[40,10]`,
universal `[1,40,132]`, decoded `[1,40,10]`, finite values, action mask, and
round-trip maximum absolute error `3.162260880928969e-08`. The official GR00T
loader is present even though the standalone `lerobot` package is absent.

## H. Synchronization and recorder

`PASS`

Jitter, dropped frame, duplicate/non-monotonic timestamp, offset, drift, and
body/wrist asynchronous cases produced the documented WARN/REJECT outcomes
without correcting source timestamps. Recorder replay preserved 121 native
states, 121 desired commands, 61 body frames, 61 wrist frames, and two
lifecycle events; source fingerprints were unchanged.

## I. Calibration and pilot readiness

`PROTOCOL_DEFINED; PHYSICAL_CALIBRATION_NOT_PERFORMED`

The real calibration protocol defines frame/TCP, camera, joint-zero, gripper,
and clock measurements with explicit `REAL_TBD` states and no invented
numeric thresholds. Therefore the Stage19 gate is:

`READY_FOR_REAL_5_10_EPISODE_PILOT`

This means ready to request/execute a separately authorized P0/P1 pilot; it
does not mean calibration is already measured, hardware is authorized, or
that a real episode exists. P0 must validate the bundle before P1 labels are
retained.

## J. Architecture status

`VLA_WBC_INTERFACE_RUNTIME_READY`

The Stage17 EEF-only thin waist remains frozen. Stage19 adds dataset
materialization, synchronization, recording, label, and calibration
contracts; it adds no network transport or hardware command path.

## K. Task-performance status

`ROBOT_SPECIFIC_GR00T_ADAPTATION_NOT_YET_PERFORMED`

No claim is made that the base GR00T checkpoint performs Go2+ARX5 tasks.

## L. Next action

`START REAL GO2_ARX5 PILOT DATA COLLECTION`

Only after explicit hardware authorization, with P0 calibration/clock
qualification first, followed by the limited 5–10 episode P1 pilot. Stage19
itself performed neither collection nor actuation.

## Final integrity

No P1M, ROS, ZMQ, PolicyServer, hardware, training, finetuning, or model-to-WBC
closed loop was run. Stage19 node-local outputs remain outside Git. The final
process, listener, Git status, staged diff, and private-artifact scan are
recorded at handoff.

# Stage19 Real-Pilot Protocol

## Readiness status

The software/data path is ready to begin a small pilot only after an explicit
hardware authorization and completion of P0 calibration checks. Stage19 did
not collect real data or send commands.

## P0 — calibration and frame/clock qualification (~5 episodes)

- record the measured `CalibrationBundleV1` and calibration diagnostics;
- verify body/wrist frame IDs, TCP identity, camera image geometry, and clock
  offset/drift;
- record desired EEF/gripper labels and achieved EEF/measured gripper as
  separate streams;
- verify lifecycle start/end, reset, timestamp monotonicity, and recorder
  durability;
- reject the bundle or episode if any required field is unknown, stale,
  non-monotonic, or mismatched.

No P0 episode is used for training until the bundle is reviewed and marked
`REAL_VALIDATED`.

## P1 — simple manipulation (5–10 episodes)

After P0 review, record a small set of simple reach/grasp/place tasks with
body RGB, wrist RGB where available, language, StateV1, future desired EEF and
gripper labels, achieved audit streams, and all raw rich state. Run the
quality/synchronization gate and reload a materialized dataset before any
training decision.

## P2 — progressive whole-body tasks

Only after P0/P1 quality is established should progressive base-motion tasks
be considered. These remain task-space labels to General WBC; raw base and
whole-body signals remain available for audit and analysis.

This protocol is a readiness plan, not permission to operate hardware. It does
not introduce ROS/ZMQ/PolicyServer or a real-robot command path.

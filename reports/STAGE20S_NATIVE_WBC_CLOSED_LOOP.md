# Stage20S Native WBC Closed-Loop Evidence

## Known-good orchestration control

The existing Stage17 orchestration path was reused with `E1_W0_00`, not with
actual GR00T output. The chain

```text
BenchmarkTrajectoryProvider
 -> TrajectoryChunkV1
 -> TrajectorySafetyGateV1
 -> EpisodeReferenceBindingV1
 -> GR00TToWBCAdapter
 -> WBCPreviewV1
 -> delivered native WBC
```

passed. The native rollout completed its reference in 98/100 requested
control steps. Position mean/P95 was `0.0129341/0.0267713 m`; rotation
mean/P95 was `0.136018/0.455563 rad`. No GR00T model was called in this
control. Chunk lifecycle also exercised fresh, aging, stale, and out-of-order
states without network transport.

## Actual model candidate

The seeded overfit model candidate received `TrajectorySafetyGateV1=ACCEPT`
with no corrective action. The Stage20S gated wrapper then forwarded that
unchanged candidate to the native simulation. It recorded:

```text
process status: NATIVE_LAUNCHER_COMPLETED
rollout status: TRACKING_ABORT
control samples: 25 / 110 requested
reference complete: false
```

At the pre-reset terminal row:

```text
traj_timing_err = 0.16725899 m
0.70 * traj_L   = 0.16311033 m
traj_d_lat      = 0.02887013 m < 1.0 m threshold
termination_cause = [false, true]  # timing
```

The native source maps the two booleans to `d_lat` and `timing`. The result is
therefore a native tracking/timing abort, not reference completion. The run
had no fall, invalid-state, timeout, or true joint-limit termination. The
negative gripper-DOF margin telemetry is retained as a diagnostic and is not
reclassified as a joint-limit termination because the delivered termination
flag was false.

The model candidate caused meaningful simulated base motion before the abort:
base displacement `0.06508 m`, base path `0.23732 m`, and base yaw change about
`0.30064 rad`. All nine internal body/gait plan channels were finite and
changed over the 25 samples. This is evidence of native controller execution,
not evidence that the visual model learned a general whole-body policy.

## Qualification consequence

The model-to-WBC chain is executable in simulation, but the easy visual-reach
closed loop did not complete its reference. The Stage20S full qualification
gate remains closed. Native WBC semantics and checkpoints were not modified.

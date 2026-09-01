# Stage17 Actual-Model Offline Interface Seal

## Result

`ACTUAL_GR00T_TRAJECTORY_REJECTED_BY_GATE`

The one actual N1.7 output was processed through the safe interface boundary:

```text
actual GR00T [1,40,132]
  -> decoded physical [1,40,10]
  -> GR00TTrajectoryProvider
  -> TrajectoryChunkV1 [40 points]
  -> TrajectorySafetyGateV1 = REJECT
  -> STOP
```

Because the gate rejected the trajectory, `EpisodeReferenceBindingV1` and the
WBC preview were intentionally skipped for this actual output. This is the
required stop behavior; no unvalidated actual-model trajectory entered WBC.

## Shape/frame/rotation checks

- raw universal shape: `[1,40,132]`
- decoded physical shape: `[1,40,10]`
- provider EEF shape: `[40,9]`
- provider gripper shape: `[40]`
- chunk timestamp shape: `[40]`
- orientation conversion: `GR00T_FIRST_TWO_ROWS -> canonical SO(3)`
- source frame: `STAGE10_SYNTHETIC_EEF_FRAME`
- TCP: `STAGE10_SYNTHETIC_TCP_UNRESOLVED`
- timestamp source: `0.05 s`,
  `TEST_ONLY_STAGE10_SYNTHETIC_DATASET`
- decoded hash matched the inference manifest:
  `a361e4fea21c5ad817167e10226fe46a936cdd630b4f509baf2c9301b23865dc`
- all provider arrays and the SO(3) representation were finite

## Descriptive trajectory audit

The trajectory duration was `1.95 s`, translation path length `1.824632 m`,
SE(3) arc length `1.838658 m`, maximum linear speed `2.114013 m/s`, maximum
angular speed `2.279299 rad/s`, and orientation-from-start range `0.071475
rad`. Initial position and rotation discontinuities against the supplied
Stage10 synthetic state were both `0`.

The action timing and TCP/frame labels remain test/synthetic provenance. They
are not production robot calibration or an assertion that the base checkpoint
understands Go2+ARX5 tasks.

## Boundary integrity

The node-local seal is
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage17/actual_groot/actual_model_interface_seal.json`.
It records `wbc_executed_from_actual_model: false` and
`stop_boundary: WBC_PREVIEW_ONLY`. The actual output NPZ and model files are
not committed.

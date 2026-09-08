# Stage19 Real Calibration Protocol

## Status

`PROTOCOL_DEFINED; PHYSICAL_MEASUREMENT_NOT_PERFORMED`

Stage19 defines `CalibrationBundleV1` measurement fields and review states but
does not invent physical transforms, camera intrinsics, joint zeros, clock
offsets, or safety limits. D3's `SIMULATION_ONLY` bundle cannot qualify a
real episode.

## Required P0 measurements

Before retaining P1 labels, a future authorized P0 pilot must:

1. freeze the frame graph and TCP semantic IDs;
2. measure Go2-body to ARX5-mount and flange-to-TCP transforms with a
   documented fiducial/robot-calibration procedure;
3. calibrate body and wrist camera intrinsics/extrinsics at the actual
   recording resolution;
4. record the ARX5 joint-zero/firmware version and measurement procedure;
5. measure raw-to-normalized gripper mapping and invalid/saturation behavior;
6. measure clock-domain offset, drift, and uncertainty using a shared
   timestampable event or synchronization signal;
7. store method, operator, time, samples, diagnostics, and source artifacts in
   a versioned calibration bundle;
8. review the bundle before it is marked `REAL_VALIDATED`.

Numeric acceptance limits are intentionally `REAL_ROBOT_TBD` until the actual
robot, cameras, calibration target, and deployment clock architecture are
selected. No simulation transform is promoted to a real transform.

## Bundle state policy

`REAL_TBD` and `MEASUREMENT_IN_PROGRESS` cannot qualify an episode.
`MEASURED_PENDING_REVIEW` requires a reviewer. Only `REAL_VALIDATED` may set
`real_robot_calibration=true`; any frame graph, sensor, mount, TCP, joint-zero,
or clock change invalidates the bundle.

The executable contract and field definitions are in
`docs/contracts/CALIBRATION_BUNDLE_V1_REAL_PROTOCOL.md`. Stage19 performed no
physical calibration and no actuation.

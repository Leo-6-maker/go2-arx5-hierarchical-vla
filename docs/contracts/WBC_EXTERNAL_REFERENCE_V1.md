# Delivered WBC external reference V1

The project-side source object is a timestamped EEF trajectory represented by
XYZ plus canonical SO(3). A native episode binding turns a relative
`TrajectoryShapeV1` plus a `PlacementProfileV1` into the absolute reference
used by the delivered runtime. The canonical adapter stores poses in
quaternion `xyzw`, then emits the native WBC rot6d wire form.

The historical WBC transforms world/anchor poses into its current
gravity-aligned, yaw-aligned heading frame while constructing the policy
observation. The adapter therefore requires explicit frame and TCP metadata;
it does not silently treat a GR00T-local frame as WBC world.

The WBC preview is spatial rather than a fixed number of raw time samples. It
uses the nine offsets defined in
`docs/contracts/TRAJECTORY_INTERFACE_V1.md`. Tangent, speed, reachability, and
progress features that the historical runtime derives internally are not
invented as GR00T output by this repository.

The native implementation constructs arc length with the Euclidean composite
increment `sqrt(||dp||^2 + (0.15*dtheta)^2)`. The canonical adapter mirrors this
metric; it does not use an additive translation-plus-rotation shortcut.

Reference modes for native evaluation are:

1. `NATIVE_GENERATOR`;
2. `BENCHMARK_CONTINUOUS`;
3. `BENCHMARK_40POINT_RECONSTRUCTED`.

All three must terminate at the same `WBCPreviewV1` boundary.

Stage15 qualification uses `MODE_NATIVE_CONTINUOUS` only. The native anchor is
reset-dependent and frozen at reset; the preview's heading frame is then
recomputed from the current base yaw during observation construction. The
controlled pose is the `x5_link6` orientation with the `gripper_center`
position offset `[0.1424, 0, 0.0001057] m`.

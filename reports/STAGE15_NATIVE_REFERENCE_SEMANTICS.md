# Stage15 native reference semantics

## Result

`VERIFIED_WITH_NATIVE_SELF_CONSISTENCY`.

The delivered external object is a variable-length trajectory of XYZ plus
rotation. In the native runtime, trajectory points are converted to the
canonical Gamma representation, then the reset-dependent native anchor is
added to position only. The policy does not consume that raw world pose
directly: it samples a nine-point SE(3) preview and expresses the preview in a
current gravity-aligned, yaw-only heading frame.

## Source flow

```text
native generator or external timestamped trajectory
  -> modules.trajectory.to_canonical
  -> TrajectoryBatch.load
  -> WBCEnv._load_trajectory_for
       Gamma position += reset anchor (translation only)
  -> TrajectoryBatch.sample_preview
       Gamma arc-length fractions and quaternion SLERP
  -> WBCEnv._arm_traj_obs_terms
       current base yaw / gravity heading transform
  -> 201D arm observation
  -> 60-frame arm history
```

Source anchors:

- trajectory canonicalization and preview: `WBCTrackingShare/modules/trajectory.py:406-414`;
- external/native load and anchor: `.../scripts/wbc_env.py:1302-1338`;
- shoulder frame: `.../scripts/wbc_env.py:325-332`;
- policy preview/frame construction: `.../scripts/wbc_env.py:2390-2442`.

## Preview contract

The preview uses exact offsets, in metres of the composite SE(3) coordinate:

```text
[0.010, 0.025, 0.050, 0.090, 0.150, 0.225, 0.325, 0.425, 0.500]
```

The native metric is `sqrt(||dp||^2 + (0.15*dtheta)^2)`. Position is linearly
interpolated on the Gamma grid and orientation uses quaternion SLERP. At the
end of a trajectory, samples are clamped to the final pose. A rotation-only
trajectory has zero translational motion but still has non-zero SE(3) progress
when its orientation changes.

The archived native rot6d wire layout is the first two rotation rows flattened
by the bundled PyTorch3D compatibility implementation. The repository keeps
this separate from the canonical/GR00T first-two-columns representation and
converts only at the WBC boundary.

## External versus internal coordination

The external reference is EEF-only. The arm policy emits six arm channels plus
the structured nine-channel internal body/gait plan. The latter is generated
after reference observation construction and is sent to the dog policy; it is
not part of `TrajectoryShapeV1`, is not a GR00T label, and is not required from
the external provider.

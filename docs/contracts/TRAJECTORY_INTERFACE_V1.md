# Canonical trajectory interface V1

## `TrajectoryChunkV1`

`TrajectoryChunkV1` is simulator/controller independent. It contains:

- `chunk_id`, `source`, and `provenance`;
- absolute `observation_timestamp` and monotonic `point_timestamps`;
- `eef_position[N,3]` in metres;
- canonical `eef_orientation[N,4]` as unit quaternion `xyzw`;
- optional derived `eef_rot6d[N,6]`, packed as the first two rotation-matrix
  columns, column-wise;
- optional `gripper[N,1]` kept outside the WBC action;
- explicit `source_frame`, `tcp_semantic_id`, and validity interval.

`N` is not fixed. GR00T currently supplies `N=40`; benchmark and replay
providers may use another count.

## `WBCPreviewV1`

The delivered WBC external preview contains nine `[XYZ, rot6d]` points. Its
offsets in the composite SE(3) arc-length coordinate are:

```text
[0.010, 0.025, 0.050, 0.090, 0.150, 0.225, 0.325, 0.425, 0.500] m
```

The implementation uses the delivered native metric
`sqrt(translation_m**2 + (0.15 m/rad * rotation_rad)**2)` and quaternion SLERP
for pose interpolation. Endpoint points are clamped and a fully stationary
path is held. A rotation-only path remains finite because rotational progress
contributes to the SE(3) coordinate.

## Shape, placement, and episode binding

`TrajectoryShapeV1` is a derived, embodiment-neutral view of relative SE(3)
motion and timing. `PlacementProfileV1` stores the intended placement class
without claiming that the class transfers between robots. Before native WBC
consumption, `EpisodeReferenceBindingV1` binds the shape to a reset state,
native shoulder/reach anchor, TCP semantic ID, and native policy frame. The
native binding is explicitly translation-only for Gamma positions; it is not
silently treated as an arbitrary rigid transform.

## `GripperCommandV1`

The normalized gripper command has its own timestamp, source convention, and
stale flag. The delivered 18D WBC action has 12 Go2 + 6 ARX5 dimensions and
does not own the gripper.

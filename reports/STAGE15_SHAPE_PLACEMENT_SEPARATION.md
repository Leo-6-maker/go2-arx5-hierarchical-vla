# Stage15 shape/placement separation

## Contracts

`TrajectoryShapeV1` preserves the derived relative SE(3) motion and timing:
trajectory ID, parent E1/E2 ID, point timestamps, task poses, family, path
parameter, and original file/hash provenance. For ordinary E1/E2 entries,
positions and orientations are expressed relative to the first EEF pose; the
original absolute file remains outside Git and is not overwritten.

`PlacementProfileV1` stores intent such as `CENTERED`, `BOUNDARY`,
`STATIC_BASE_RELOCATION_TARGET`, `CONTINUOUS_BASE_COORDINATION_TARGET`, or
`EXPECTED_INFEASIBLE`. It is a request to instantiate a challenge, not a
portable label.

`EpisodeReferenceBindingV1` adds the native reset state, shoulder/reach anchor,
TCP semantic, task placement transform, native frame, and binding semantics.

## Why the separation matters

The same relative path can be easy, boundary, or unreachable after changing
the robot geometry, initial pose, mount, or TCP. Conversely, a path that was a
continuous-base candidate in the P1M setup need not be a continuous-base
candidate in the native setup. This is a property of:

```text
trajectory shape + placement + embodiment
```

not of the path shape alone.

## Derived artifact

The 114 E1/E2 shapes were re-instantiated under
`/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage15/native_benchmark/shapes`.
Each entry retains the original path, original SHA256, derived shape SHA256,
placement intent, native classification, and a position-only reachability
record in `native_benchmark_manifest.json`.

The aligned seven-run set uses native-bound continuous trajectories. The
`E2_cont_0_0` item is retained as a negative/infeasible reference after native
reclassification; it is not counted as a native continuous-base success case.

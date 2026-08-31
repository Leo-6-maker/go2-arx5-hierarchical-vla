# STAGE14 FINAL REPORT

## A. Classification

`PASS WITH DEVIATION`

Repository consolidation and CPU contract validation passed. Native runtime
execution and mechanical external-reference injection passed. The scientific
native E1/E2 baseline remains gated by one specific reference contract issue.

## B. Repository

- canonical path: `/home/liuyu-nfs/Projects/go2-arx5-hierarchical-vla`;
- synchronized local path:
  `/mnt/LocalHDD1/HomeNFS/liuyu-nfs/Projects/go2-arx5-hierarchical-vla`;
- consolidation commit used by native runs:
  `b9b58f857a21f6d5b7426ce43ca4007bc0e6e224`;
- GitHub remote: none; GitHub CLI unavailable, so private remote creation was
  not attempted;
- private/large artifacts: outside Git and bound through five manifests.

## C. Canonical interface

The repository now owns `TrajectoryChunkV1`, `WBCPreviewV1`, and
`GripperCommandV1`. A chunk has variable point count, timestamps, canonical
SO(3) orientation, optional derived rot6d, explicit source frame/TCP identity,
validity interval, provenance, and a separate optional gripper channel.

The delivered WBC preview is nine XYZ+rot6d points at exact SE(3) arc offsets
`[0.010, 0.025, 0.050, 0.090, 0.150, 0.225, 0.325, 0.425, 0.500] m`, with
lambda `0.15 m/rad`. Rotations use quaternion SLERP; rot6d is not linearly
interpolated.

## D. Native runtime and checkpoints

The native IsaacGym runner was reconstructed with the existing arm/dog network
definitions and both checkpoints loaded using `weights_only=True`; strict key
and shape binding passed. `NATIVE_GENERATOR` executed finite tensors and
natural teardown, but its deterministic smoke ended at 25/100 steps under the
native timing termination rule.

## E. External native baseline

Seven requested diagnostic trajectories were injected in
`BENCHMARK_CONTINUOUS`: six completed 100 steps and the orientation-only case
ended at 25 steps under native timing termination. No NaN, crash, hardware, or
process-lifecycle fault occurred. The detailed table is in
`STAGE14_NATIVE_BASELINE.md`.

These results are explicitly **preliminary unaligned injection evidence**,
not `NATIVE_GENERAL_WBC_TRACKING`. E1/E2 files declare `SIM_REFERENCE_ONLY`,
while the native loader adds a reset-dependent shoulder/reach anchor. No
source-backed transform from the benchmark frame/TCP convention into native
world/episode coordinates is currently bound. The adapter therefore has a
`REFERENCE_INJECTION_BLOCKER`.

## F. Metrics and ablations

The repository keeps separate APIs for:

- `REFERENCE_RECONSTRUCTION_ERROR`;
- `CONTROLLER_TRACKING_ERROR`;
- `END_TO_END_TASK_SPACE_ERROR`.

The CPU 40-point adapter semantics pass, but the native 40-point paired
ablation was not run because the continuous reference contract is not yet
qualified. No 24--36 trajectory expansion was launched.

## G. P1M boundary

P1M remains a later target-embodiment transfer case. Stage13C's
`93/100` clipping and `0.527/0.864 m`, `1.761/3.018 rad` errors remain labeled
`UNQUALIFIED_BRIDGE_TRACKING`; they are not included as native WBC results.

## H. CPU validation

- `17 passed` in the Git-tracked pytest suite;
- `CPU_CONTRACT_SMOKE_PASS`;
- five artifact manifests validated.

## I. Final scientific gate

`REFERENCE_INJECTION_BLOCKER`

This is the precise remaining issue: obtain or define the explicit
`SIM_REFERENCE_ONLY -> native WBC world/episode` frame, TCP, and anchor
contract, then rerun the seven continuous trajectories. Do not use the current
numbers to claim EEF-only success/failure, E2 whole-body resolution, or native
generalization.

## J. Integrity

The delivered WBC archive and P1M SHA256 anchors are unchanged. The shared
GR00T and SONIC repositories were not modified. No Stage14 process, ROS, ZMQ,
PolicyServer, or project-owned listener on 5550/5556/5557 remained at exit.

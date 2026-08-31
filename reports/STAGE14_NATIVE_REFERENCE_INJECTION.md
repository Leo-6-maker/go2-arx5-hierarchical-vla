# Stage14 native external-reference injection

Status: `MECHANICAL PASS; SEMANTIC FRAME/ANCHOR BLOCKER`

## What was changed

The maintained launcher in
`scripts/native_wbc_benchmark.py` is a clean-room integration layer. It does
not copy the private lab source into Git and does not alter policy weights,
history construction, normalization, action post-processing, physics, or the
native robot asset. It supports:

- `NATIVE_GENERATOR`;
- `BENCHMARK_CONTINUOUS`;
- `BENCHMARK_40POINT_RECONSTRUCTED`.

For external modes, only trajectory loading is overridden. The source is
converted through the delivered runtime's existing `to_canonical` and loaded
into the same `TrajectoryBatch`, after which the native heading-frame preview,
reachability field, timing state, arm policy, internal body plan, dog policy,
and 18D actuation path are unchanged.

## Mechanical evidence

The following continuous-reference runs loaded external NPZ trajectories,
loaded both checkpoints with `weights_only=True`, produced finite tensors, and
completed the requested 100 control steps unless noted:

| trajectory | result | steps | note |
|---|---:|---:|---|
| `E1_W0_00` | full | 100/100 | natural completion |
| `E1_W0_01` | full | 100/100 | natural completion |
| `E1_ang_0_00` | partial | 25/100 | native timing termination |
| `E1_curvature_1_00` | full | 100/100 | natural completion |
| `E1_coupling_1_00` | full | 100/100 | natural completion |
| `E2_static_0_0` | full | 100/100 | natural completion |
| `E2_cont_0_0` | full | 100/100 | natural completion |

The orientation-only `no_early_term` run is diagnostic only and is not part of
the baseline. It disables two trajectory termination gates and therefore does
not have the baseline configuration.

## Why this is not yet a qualified interface

The benchmark files identify their frame as `SIM_REFERENCE_ONLY`. Their first
E1 point is approximately
`[0.45435, 0.00100, 0.13208]` with the reference-only quaternion
`[0.60816, -0.60815, 0.36076, -0.36075]`. The native runtime's normal loader,
in contrast, constructs an origin-centered trajectory and applies a
reset-dependent shoulder/reach anchor in native world coordinates. The current
hook loads the benchmark values and then applies that native anchor; it does
not possess a source-backed `SIM_REFERENCE_ONLY -> native_world` transform or
a declared TCP/mount calibration.

Consequently the first E1 target in the direct run is approximately
`[0.95622, 0.04943, 0.69034]`, while a same-seed native-generator run begins
near `[0.55964, 0.01241, 0.63440]`. This is evidence of different anchor
semantics, not a measurement of policy failure. The comparison is not claimed
to be a paired trajectory comparison; it is an injection audit.

The correct unresolved question is therefore not whether the tensors can be
plumbed, but which explicit transform/anchor convention makes the project
benchmark comparable to the native training/reference convention. Applying a
start-pose alignment without a declared contract would be an untracked
experimental assumption, so it was not silently introduced.

## Gate

`REFERENCE_INJECTION_BLOCKER` remains the Stage14 gate. The current runs are
retained as preliminary, unaligned diagnostic evidence. No broad 24--36
trajectory benchmark and no native 40-point rollout ablation was launched.

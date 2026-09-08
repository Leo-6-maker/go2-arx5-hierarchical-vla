# Stage20T Comparison with the Stage16 Qualified Timing Envelope

## Scope

Stage20T re-ran the known-good Stage16 `E1_W0_00` reference at seeds 13, 14,
and 15. This is a repeatability control, not a replacement for the frozen
28-trajectory Stage16 benchmark.

The Stage16 seed-13 baseline was 27 `REFERENCE_COMPLETE` and one retained
boundary `TRACKING_ABORT` over 28 cases. Stage20T uses the delivered native
termination semantics without changing the `0.70 * traj_L` rule.

## Stage20T control envelope

| Control seed | Result | max absolute timing fraction | minimum timing margin (m) | first runtime jump (m/rad) |
|---:|---|---:|---:|---:|
| 13 | `REFERENCE_COMPLETE` | 0.184231 | +0.025950 | 0.0382 / 0.0543 |
| 14 | `REFERENCE_COMPLETE` | 0.251655 | +0.022557 | 0.2153 / 0.7145 |
| 15 | `TRACKING_ABORT` at step 0 | 0.999912 | -0.015089 | 0.2005 / 1.9555 |

For the two completed controls, the observed max timing fraction was
`0.184231–0.251655` and the minimum margin was `0.022557–0.025950 m`.
The aggregate analyzer envelope is therefore explicitly scoped to the two
completed controls; it is not a universal native bound.

## Matched reference comparison

At `1.00x`:

| Run | Result | max timing fraction | minimum margin (m) |
|---|---|---:|---:|
| A model, seed 13 | abort | 0.733565 | -0.007821 |
| A model, seed 14 | abort | 0.753924 | -0.012565 |
| A model, seed 15 | shared reset boundary | 0.977684 | -0.064704 |
| B GT, seed 13 | complete | 0.186645 | +0.031014 |
| B GT, seed 14 | complete | 0.252483 | +0.027037 |
| B GT, seed 15 | shared reset boundary | 0.998449 | -0.018031 |

The model seed-13/14 traces leave the Stage16 completed-control envelope
while the same-seed GT traces remain within it. This is the key attribution
evidence for a model trajectory profile limitation. The seed-15 control and GT
failure is outside that comparison because the reset starts with an immediate
forward-projection/timing violation.

## Spatial and temporal descriptors

The model candidate has `0.233015 m` translation path length, `1.95 s`
duration, and `0.310131 m/s` maximum linear speed at original timing. The GT
label has `0.060415 m`, `1.95 s`, and `0.046463 m/s`. The Stage16 control has
`0.050312 m`, `2.00 s`, and `0.047164 m/s`.

This supports the interpretation that the accepted model output has a much
larger spatial/speed profile than the same-observation desired label and the
qualified control. It is not evidence for a production action frequency or a
real-robot speed threshold.

## Decision

```text
NATIVE_TIMING_BOUNDARY_CHARACTERIZED
```

The boundary is characterized with a matched GT control and three fixed
seeds. The model-specific result is additionally recorded as
`MODEL_TRAJECTORY_PROFILE_LIMITATION_IDENTIFIED`.

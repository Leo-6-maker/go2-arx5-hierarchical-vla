# Stage34 WBC-Aligned Motion API Qualification

Stage34 is a zero-model qualification of a side-by-side `MotionIntentV2`
contract.  It does not modify the dense H40 relative-EEF GR00T ABI, the
Stage23 action contract, the delivered WBC, or any historical artifact.

## Frozen protocol

The geometry candidates are `K = 4, 6, 9, 12` control poses selected at
uniform normalized arc-length knots under
`sqrt(||dp||^2 + (0.15*dtheta)^2)`.  Reconstruction is linear XYZ plus
quaternion SLERP, with exact endpoint and duration preservation.  The preview
comparison uses the existing nine native offsets:

```text
[0.010, 0.025, 0.050, 0.090, 0.150, 0.225, 0.325, 0.425, 0.500] m
```

Timing candidates are `T1`, `T3`, `T5`, and `T9`, represented by monotonic
piecewise-linear progress laws.  No smoothing, spline fitting, optimization,
manual knot placement, outcome-dependent retiming, model construction, or
simulator execution is part of this harness.

The promotion thresholds are the preregistered Stage16 controller-floor
fractions: preview position p95 `< 0.006767 m`, rotation p95 `< 0.038215 rad`,
endpoint-clamp agreement `>= 0.99`, exact start/end pose, and exact duration.

## Provenance and stop rule

The primary promotion cohort requires the complete source-backed Stage16
native reference authority: 28 cases with the 27 nominal-complete cases as
primary, the retained boundary case as secondary diagnostic, and the exact
delivered WBC inputs.  Already-unlocked Stage22 train snapshots are sentinel
diagnostics only.  Validation `e` and holdout `f` are never read.

If the primary authority cannot be reproduced exactly, the harness still
seals its CPU diagnostics but reports `INSUFFICIENT_REFERENCE_PROVENANCE`,
does not promote `K_STAR` or `T_STAR`, and does not execute the conditional
oracle WBC campaign.  Stage34-E0 only audits or designs a future
support-aware continuous-base cohort; it never fabricates or rolls out one.

## MotionIntentV2 boundary

`MotionIntentV2` is intentionally additive.  Its timestamped control poses,
explicit frame/TCP identity, validity interval, timing-law knots, separate
gripper channel, runtime-anchor semantics, and provenance are validated
fail-closed.  Conversion back to `TrajectoryChunkV1` is guarded by exact
frame/TCP expectations.  Native row-packed rot6d remains an adapter-boundary
wire detail and is not a V2 model target.

The final machine-readable return is written only after the CPU evidence is
sealed and stops at
`GPT_PI_REVIEW_AFTER_STAGE34_WBC_ALIGNED_MOTION_API_RESULT`.  No PR is merged
automatically.

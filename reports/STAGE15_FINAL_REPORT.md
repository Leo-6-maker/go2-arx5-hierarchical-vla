# STAGE 15 FINAL REPORT

## A. Classification

`PASS WITH DEVIATION`

The native reference-binding contract is closed and independently reproduced.
The derived shape/placement benchmark is available, and the aligned native
diagnostic runs execute coherently for centered/easy cases. The stage does not
qualify a broad native benchmark because the reused E1/E2 set contains no
native continuous-base cohort and its selected native static-relocation case
terminates under the native timing rule.

## B. Native anchor

- reset-dependent: **YES**;
- temporal behavior: **frozen at reset/trajectory creation**;
- shoulder: `base_position + R(base_quaternion) [0.05, 0, 0.10]`;
- direction: `R(base_quaternion) normalize([0.36, 0, 0.10])`;
- radius: native reach-table query;
- scale: `rho_star=0.60`;
- position operation: world-aligned translation-only Gamma anchoring;
- policy preview frame: current gravity-aligned, yaw-only heading.

The 12-reset independent audit reproduced anchor position with maximum error
`7.2759e-08 m`.

## C. TCP

The native controlled semantic is `x5_link6_gripper_center`: orientation is
the `x5_link6` orientation and position applies the fixed link-to-TCP offset
`[0.1424, 0, 0.0001057] m`. P1M TCP and real calibration are not used.

## D. Binding self-consistency

`EpisodeReferenceBindingV1` is ready. Across 12 reset traces and 12 generated
trajectories:

| comparison | maximum error |
|---|---:|
| anchor position | `7.2759e-08 m` |
| bound position | `0.0 m` |
| bound orientation | `4.2147e-08 rad` |
| nine-point preview position | `4.4178e-08 m` |
| nine-point preview orientation | `8.9605e-07 rad` |
| policy reference tensor | `2.4875e-07` |

The archived native row-packed rot6d convention is now explicitly separated
from the canonical/GR00T column-packed convention.

## E. Shape / placement contracts

- `TrajectoryShapeV1`: **READY**;
- `PlacementProfileV1`: **READY**;
- `EpisodeReferenceBindingV1`: **READY**.

All 114 E1/E2 references were converted to derived initial-EEF-relative shape
objects with original IDs, source paths, and source hashes retained as
provenance. Placement intent is separate and is re-evaluated for the native
embodiment.

## F. Native reclassification

| class | count |
|---|---:|
| `NATIVE_ARM_EASY` | 70 |
| `NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT` | 17 |
| `NATIVE_CONTINUOUS_BASE_MOTION_REQUIRED` | 0 |
| `NATIVE_BOUNDARY` | 3 |
| `NATIVE_GEOMETRICALLY_INFEASIBLE` | 24 |

All 16 original E2 static cases remained native static-relocation candidates.
All 16 original E2 continuous cases became native geometrically infeasible in
the bounded native position-only reachability analysis. This is a
trajectory-plus-placement-plus-embodiment classification change, not a policy
failure claim.

## G. Orientation-only

The Stage14 early termination was caused by native timing progress:
`timing_err=0.0397052 m` exceeded the configured `0.0352306 m` threshold;
distance-lateral termination was false. After native reset/TCP orientation
binding, `E1_ang_0_00` completed 100/100 steps. Classification:
`SUPPORTED_WITH_EXPLICIT_DURATION`.

## H. Aligned qualification

The seven-run native-continuous diagnostic set was:

- five centered/easy or nominal references: 100/100 natural completion;
- one native static-relocation candidate: 25/100, native timing termination;
- one native geometrically-infeasible negative control: 100/100 bounded run,
  but not a success case.

The five nominal cases had position mean errors from `0.0122` to `0.0235 m`
and rotation mean errors from `0.1320` to `0.1804 rad`. The static case had
`0.2063 m` position mean and `0.5766 rad` rotation mean before termination.
All controller/state tensors were finite; no crash, NaN, ROS, hardware, or
lifecycle failure occurred.

The native internal 9D plan remained finite and was logged together with base
motion, arm action, dog action, and reference progress. The runs demonstrate
that the internal body/gait channel is active, but the seven runs are too small
and placement-incomplete to establish whole-body redundancy resolution over a
native benchmark.

## I. Architecture hypothesis

`EEF_ONLY_EXTERNAL_INTERFACE_STILL_SUPPORTED`

`EpisodeReferenceBindingV1` is the required native adapter. The 9D body/gait
plan remains internal to the delivered WBC and is not a required GR00T output.
The GR00T gripper scalar remains a separate parallel channel.

## J. Stage14 reinterpretation

Stage14 results remain `UNALIGNED_REFERENCE_INJECTION_SMOKE`. They are not
used as native General-WBC performance evidence. The Stage13C P1M numbers
remain `UNQUALIFIED_BRIDGE_TRACKING` and are not native performance evidence.

## K. Next gate

`ONE_SPECIFIC_NATIVE_REFERENCE_CLASS_BLOCKER_REMAINS`

The remaining blocker is not frame binding. It is the native qualification
set: the selected static-relocation placement does not yet yield a valid
native diagnostic result, and no reused E1/E2 shape instantiates as a native
continuous-base-required case. Before a broad benchmark, either resolve that
native placement/timing qualification or deliberately create a native-valid
shape/placement cohort and classify it with the same native analysis.

No P1M tracking, GR00T, GR00T40 ablation, training, or hardware control was
run in Stage15.

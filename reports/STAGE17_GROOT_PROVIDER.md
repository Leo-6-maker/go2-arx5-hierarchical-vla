# Stage17 Canonical GR00T Trajectory Provider

## Contract

`GR00TTrajectoryProvider` implements the repository's `TrajectoryProvider`
contract and exposes one `TrajectoryChunkV1` through `next_chunk`. It accepts
decoded physical model output, not a Torch tensor, `BatchFeature`, or raw
universal action. Model-specific representation details stop at this
boundary.

The Stage10 physical layout is:

```text
[H,10] = [eef_9d: XYZ + GR00T first-two-rows rot6d, gripper_position]
```

The provider emits:

```text
TrajectoryChunkV1: EEF XYZ + canonical quaternion / first-two-column rot6d
GripperCommandV1: parallel channel, never part of WBC preview input
```

The 40-point split is explicit: EEF `[40,9]`, gripper `[40,1]`. Batched
`[B,40,10]` output is supported with an explicit batch index.

## Orientation and time semantics

The validated conversion is `GR00T_FIRST_TWO_ROWS` rot6d to SO(3), then to the
repository's canonical quaternion and column-packed rot6d. The native WBC
preview conversion remains a separate row-packed operation in the existing
adapter.

The provider never assumes 20 Hz. It requires explicit point timestamps or a
configured positive `action_sample_period_s` with a non-TBD provenance label.
The production value is `TBD_BY_ADAPTATION_DATASET`; the Stage17 actual-model
smoke labels `0.05 s` as `TEST_ONLY_STAGE10_SYNTHETIC_DATASET`.

## Provenance and safety

Each chunk carries source frame, TCP semantic ID, timestamp provenance,
physical layout, slices, source, and model-output provenance. No re-anchoring,
clipping, re-timing, or silent correction is performed by the provider.

The provider is covered by CPU tests for `[40,10]` splitting, batched output,
explicit timestamps, gripper separation, and rejection of implicit production
timing. The Stage17 CPU suite passed after the final lifecycle test addition.

# ADR-013: Provider Thin Waist

## Status

Accepted for Stage17.

## Context

GR00T universal tensors and processor internals are model-specific. WBC and
future transports need an embodiment-facing, timestamped EEF contract with
explicit frame/TCP semantics and a separate gripper channel.

## Decision

`GR00TTrajectoryProvider` is the canonical thin waist. It accepts decoded
physical `[H,10]` or `[B,H,10]`, converts the validated GR00T rot6d layout to
canonical SO(3), emits `TrajectoryChunkV1`, and separates gripper into
`GripperCommandV1`. It requires explicit timestamps or a labelled test period;
it does not assume 20 Hz and does not expose raw universal tensors downstream.

## Consequences

The adapter and WBC remain independent of GR00T tensor layout. Dataset timing,
frame calibration, and TCP semantics become explicit integration contracts.
The provider does not imply that the base model learned the target robot.

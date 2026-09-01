# ADR-024: Same-Scene Expert Supervision at Replanning Points

- Status: Accepted for Stage22
- Date: 2026-09-01

## Decision

Generate robot-specific adaptation labels at the actual native-simulator
replanning observations. Each example contains native body/wrist RGB, the
current `StateV1`, and a deterministic future desired EEF trajectory plus
gripper. The target actor may generate labels and evaluation metadata but is
excluded from the model input.

The first expert is a smoothstep task-space position path from the current EEF
to the same-scene target with constant current orientation and a hold. Prefix
execution and fresh observations occur without physics reset, teleport, or
policy-history reset.

## Rationale

Stage21 model inference encounters post-prefix achieved states. A dataset that
contains only reset-state examples cannot test that distribution. The simple
visual-reach expert isolates action-profile and runtime attribution before
contact, grasp, or object dynamics are introduced.

Snapshot-expanded 40-point files are permitted as a materialization fixture,
but are explicitly not represented as continuous native camera recordings.

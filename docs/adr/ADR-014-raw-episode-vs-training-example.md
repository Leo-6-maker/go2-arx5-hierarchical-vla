# ADR-014: Keep raw episodes separate from derived training examples

## Status

Accepted — Stage18

## Decision

Record every future robot episode first as immutable `RawEpisodeV1`. Derive
`GR00TTrainingExampleV1` offline through explicit calibration and
synchronization contracts. LeRobot/official GR00T data is a derived consumer,
not the sole raw archive.

## Rationale

The raw record must preserve native timestamps, achieved state, whole-body
diagnostics, lifecycle events, and calibration provenance even when a model
only consumes compact EEF state and desired EEF/gripper labels. This keeps
label choices reversible and makes timing/calibration failures auditable.

## Consequences

The D3 fixture can validate the conversion path without claiming real-robot
calibration or task performance. Future real pilot promotion requires a
calibration bundle and quality result that are not simulation-only.

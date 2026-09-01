# ADR-012: Trajectory Safety Gate

## Status

Accepted for Stage17.

## Context

An external EEF trajectory can be finite and syntactically decodable while
still having discontinuities, invalid timestamps, excessive speed, or an
unsupported workspace/path burden. A downstream controller must not silently
repair such output.

## Decision

Use `TrajectorySafetyGateV1` as a non-corrective `ACCEPT`/`WARN`/`REJECT`
boundary. It validates finite values, timestamps, SO(3), continuity, speed,
workspace/path sanity, arc length, and chunk age. Every threshold carries
explicit provenance; production robot safety values remain
`REAL_ROBOT_TBD`.

## Consequences

Rejected trajectories remain unchanged and cannot enter binding or WBC. The
gate can safely describe an unadapted model output without converting that
observation into a task-performance claim.

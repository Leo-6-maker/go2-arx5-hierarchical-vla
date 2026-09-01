# ADR-010: Native WBC Baseline Promotion

## Status

Accepted for Stage17.

## Context

Stage16 ran 28 native benchmark trajectories with 27 reference completions,
one deliberate boundary `TRACKING_ABORT`, zero falls/invalid resets/timeouts,
and eight progressive cases classified `BASE_RELOCATION_LIKE`. The one abort
was traced to the native timing condition and is retained as a known boundary
limitation.

## Decision

Promote the frozen Stage16 result as
`NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`. Stage17
does not continue WBC research or silently remove the boundary case. Native
WBC evidence remains separate from actual GR00T task evidence.

## Consequences

Stage17 can use a qualified Stage16 trajectory for orchestration validation.
The promotion does not imply universal dynamic coverage, continuous-base
coordination, or robot-specific VLA performance.

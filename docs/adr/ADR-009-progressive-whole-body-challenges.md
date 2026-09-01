# ADR-009: Progressive Whole-Body Challenges

## Status

Accepted for Stage16.

## Context

The native E1/E2 placement library contains static-base-relocation candidates but no valid native cohort in which continuous base motion is required and geometrically feasible. Several parent candidates also begin with a large EEF jump from the reset state. Such a jump confounds initial continuity with whole-body coordination.

## Decision

Whole-body controller evaluation should prioritize references that start continuously from the current reachable state and progressively increase base-motion demand. A derived trajectory may add a smooth SE(3) lead-in, but it must preserve the parent trajectory ID/hash, derived hash, join time, join error, lead-in duration/path length, and unchanged challenge suffix. The derived reference must be marked as derived and must not overwrite the parent shape.

The fixed-base reachability calculation is used to define a burden boundary, not as a unique ground-truth base path. Controller behavior is classified from measured base/EEF/internal-plan traces.

## Consequences

The Stage16 progressive cohort removes artificial initial jumps and makes before/after reach-boundary behavior interpretable. It does not manufacture a continuous-base cohort when geometry does not provide one, and it does not require the WBC to reproduce a single oracle base trajectory.

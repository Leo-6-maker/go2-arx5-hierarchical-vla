# Stage16 Continuous-Base Challenge Decision

## Decision

`NO_VALID_NATIVE_CONTINUOUS_BASE_COHORT_FOUND`

The existing 114 E1/E2 shape-placement combinations produced zero `NATIVE_CONTINUOUS_BASE_MOTION_REQUIRED` cases after native position reachability reclassification. A bounded offline search over long smooth lines, arcs, workspace crossings, and splines did not yield a trajectory that simultaneously started continuously, had bounded speed, was not immediately unreachable, was not covered by one static base pose, and remained geometrically feasible with a moving base.

The initial native benchmark is therefore classified:

`NOT_REQUIRED_FOR_INITIAL_VLA_WBC_INTEGRATION`

This is not a controller failure and does not block the native EEF-interface benchmark. The eight progressive relocation cases are the available clean whole-body stress cohort and are explicitly marked `DERIVED_PROGRESSIVE_RELOCATION`; none is mislabeled as a continuous-base success case.

## Future stress-stage criterion

A future continuous-base stress case should be admitted only if it has a reset-continuous lead-in, bounded linear/angular velocity, a measurable fixed-base reach boundary, no immediate initial infeasibility, and a separate evidence-backed moving-base feasibility analysis. The moving-base path is an oracle feasibility witness, not a unique controller ground truth.

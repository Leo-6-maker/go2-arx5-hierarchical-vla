# ADR-002: Redundancy resolution initially belongs inside WBC

**Status:** Accepted for the initial baseline

**Decision:** The WBC may internally produce and consume its structured 9D
body/gait coordination plan. That plan is not a required GR00T output.

**Rationale:** E2 found multiple feasible whole-body decompositions for the
same EEF path. The delivered historical controller's arm policy produces a
structured body/gait signal internally, which is a concrete mechanism for
resolving that redundancy without widening the VLA output.

**Consequence:** Stage14 logs all nine channels for analysis but does not train
or label GR00T with them. An external body-intent ablation remains future work
and requires native WBC evidence.

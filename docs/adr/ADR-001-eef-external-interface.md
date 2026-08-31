# ADR-001: EEF trajectory is the external VLA interface

**Status:** Accepted for the initial baseline

**Decision:** GR00T supplies a timestamped EEF trajectory. The external
reference is XYZ + SO(3)/rot6d, with explicit frame and TCP metadata. It does
not supply joint targets or a body plan.

**Rationale:** Stage13A/B verified the delivered WBC's external reference as a
variable-length XYZ+rot6d trajectory with a nine-point spatial preview. This is
the smallest auditable interface and preserves the E2 question: can internal
WBC redundancy resolution realize the same EEF request?

**Consequence:** A reference adapter is required where source and WBC frame or
sampling semantics differ. EEF-only is a hypothesis under native evaluation,
not a claim of universal transfer.

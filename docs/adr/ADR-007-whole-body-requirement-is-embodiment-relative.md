# ADR-007: Whole-body requirement is embodiment-relative

**Status:** Accepted for Stage15

**Decision:** Labels such as arm-only, static-base relocation, and
continuous-base coordination are recomputed for each embodiment and placement.

**Rationale:** Stage15 reclassification changed the interpretation of the
original E2 set: the 16 P1M continuous-base cases were geometrically
infeasible under the bounded native reach analysis, while the 16 original
static cases remained native static-relocation candidates. The shape did not
change; the embodiment and placement did.

**Consequence:** Native qualification reports use `NATIVE_*` labels and never
use an original P1M label as a native success/failure ground truth.

# ADR-006: Bind every external reference to the native episode

**Status:** Accepted for Stage15

**Decision:** External WBC references pass through an explicit
`EpisodeReferenceBindingV1` containing reset state, native shoulder/reach
anchor, TCP identity, frame semantics, placement, and preview provenance.

**Rationale:** The delivered WBC constructs a reset-dependent anchor and then
converts the bound trajectory into a dynamic heading-frame policy preview.
Raw benchmark XYZ values are therefore not a sufficient reference contract.

**Consequence:** Native self-consistency against the delivered reference
provider is the release gate for aligned benchmark tracking.

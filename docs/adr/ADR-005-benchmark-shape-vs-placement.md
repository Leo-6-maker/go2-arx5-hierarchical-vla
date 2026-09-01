# ADR-005: Separate trajectory shape from embodiment placement

**Status:** Accepted for Stage15

**Decision:** E1/E2 references are represented as a derived `TrajectoryShapeV1`
plus an embodiment-specific `PlacementProfileV1`. Relative SE(3) motion,
timing, speed, curvature, and orientation are retained as shape. Initial
workspace location, reach margin, and whole-body intent are retained as
placement.

**Rationale:** The original E1/E2 absolute positions were created under prior
simulation assumptions. Whole-body requirement depends on trajectory,
embodiment, and placement, so an original P1M label cannot silently become a
native-WBC label.

**Consequence:** A shape must be instantiated and geometrically reclassified
for each embodiment before it is used as a controller benchmark.

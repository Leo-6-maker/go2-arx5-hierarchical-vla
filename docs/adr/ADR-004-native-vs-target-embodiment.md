# ADR-004: Native embodiment is the WBC baseline

**Status:** Accepted for Stage14

**Decision:** The delivered WBC is first qualified in its native IsaacGym
Go2+ARX5 embodiment. P1M is a separate target-embodiment transfer case.

**Rationale:** Stage13C showed coherent native execution but material P1M
mismatches in ARX5 axes, origins, limits, mount/TCP, and drive semantics.

**Consequence:** P1M tracking numbers remain `UNQUALIFIED_BRIDGE_TRACKING` and
must not be reported as general-WBC performance. Native reference tracking is
the next scientific baseline.

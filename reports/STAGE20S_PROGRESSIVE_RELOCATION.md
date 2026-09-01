# Stage20S Progressive Relocation Stress Cohort

## Construction

Eight deterministic `PROGRESSIVE_BASE_RELOCATION` fixture episodes were
materialized with target deltas increasing from approximately `0.08 m` to
`0.255 m`. Each starts continuously from the fixture reset state and has a
61-row episode with a 40-point future label. The 8-episode dataset passed the
official loader/collator/processor path with 176 valid windows.

## Actual-model offline audit

The overfit easy checkpoint was evaluated on progressive episode 0 only as an
offline stress test. The candidate was finite, valid SO(3), and accepted by
the test gate. Its audit values were:

| Quantity | Value |
|---|---:|
| initial position discontinuity | `0.004391 m` |
| rotation discontinuity | `0 rad` |
| path length | `0.367653 m` |
| maximum declared-timing speed | `0.537517 m/s` |
| model-vs-label position mean/P95/max | `0.047252 / 0.063373 / 0.070148 m` |

No progressive candidate was forwarded to native WBC after the easy
actual-model rollout failed to complete. In addition, the fixture's base is
static and its camera is cross-simulator, so this cohort is not evidence of
learned continuous base coordination. It is retained as a clean future stress
fixture rather than promoted to a whole-body success result.

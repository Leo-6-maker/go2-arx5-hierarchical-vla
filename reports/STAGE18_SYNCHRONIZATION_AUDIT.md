# Stage18 Synchronization Audit

## D3 measured source timing

For all six selected V0 episodes:

| Stream | Samples | Timestamp interval | Effective rate | Declared rate |
|---|---:|---:|---:|---:|
| native state/commands | 121 | 0.02 s | 50 Hz | not declared |
| body camera | 61 | 0.04 s | 25 Hz | 20 Hz |
| wrist camera | 61 | 0.04 s | 25 Hz | 20 Hz |

`SynchronizerV1` uses body camera source timestamps as the fixture observation
times, maps state/desired/achieved streams by nearest source timestamp, and
retains indices. All selected D3 camera samples coincide exactly with native
timestamps, so camera↔state, desired↔state, and achieved↔state skew are 0 s.
There were zero non-monotonic timestamps, dropped frames, or detected clock
discontinuities in this fixture.

## Decision

The effective/declared rate discrepancy is recorded as
`D3_CAMERA_FPS_METADATA_MISMATCH` with `BENCHMARK_DERIVED` provenance. It is a
data-pipeline blocker for real-pilot readiness until the source metadata and
clock/rate contract are reconciled. No 20 Hz assumption, resampling, or
retiming was introduced. Production action timing remains
`TBD_BY_ADAPTATION_DATASET`.

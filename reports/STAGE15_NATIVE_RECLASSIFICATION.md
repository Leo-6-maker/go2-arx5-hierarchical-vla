# Stage15 native reclassification

## Counts by source group

| source group | native arm-easy | native static | native continuous | native boundary | native infeasible |
|---|---:|---:|---:|---:|---:|
| E1 (78) | 70 | 1 | 0 | 3 | 4 |
| E2 static (16) | 0 | 16 | 0 | 0 | 0 |
| E2 continuous (16) | 0 | 0 | 0 | 0 | 16 |
| E2 expected infeasible (4) | 0 | 0 | 0 | 0 | 4 |
| **total (114)** | **70** | **17** | **0** | **3** | **24** |

The E1 row includes the E1 boundary/stress/whole-body families as encoded in
the source manifest. The classification method is the bounded native
reach-table analysis described in `STAGE15_NATIVE_REACHABILITY.md`; dynamic
feasibility and orientation feasibility are explicitly `NOT_EVALUATED`.

## Interpretation

Sixteen original E2 static-base cases remain useful native static-relocation
diagnostics. Sixteen original E2 continuous-base cases do not form a native
continuous-base cohort under the same derived placement; they are retained as
negative controls. The label change demonstrates why the original P1M
whole-body labels cannot be reused silently.

No native `NATIVE_CONTINUOUS_BASE_MOTION_REQUIRED` benchmark was fabricated.
The aligned Stage15 suite therefore contains one static candidate and one
native-infeasible negative control rather than claiming coverage that the
reinstantiated data do not provide.

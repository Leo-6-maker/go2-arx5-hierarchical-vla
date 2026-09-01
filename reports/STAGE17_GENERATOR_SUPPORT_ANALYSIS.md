# Stage17 Actual-Output Generator Support Analysis

## Method

The actual decoded output was compared descriptively with the same 10,000
Stage13B reconstructed generator descriptors used by Stage16. The comparison
uses the existing 12-feature robust median/IQR normalization and nearest
descriptor distance. It is named `GENERATOR_DESCRIPTOR_DISTANCE`; it is not a
training-distribution measurement.

## Actual output descriptor

| Feature | Actual output |
|---|---:|
| duration | `1.950000 s` |
| translation path | `1.824632 m` |
| net displacement | `0.045777 m` |
| translation speed mean / p95 / max | `0.935709 / 1.815093 / 2.114013 m/s` |
| angular speed mean / p95 / max | `0.645301 / 1.699249 / 2.279299 rad/s` |
| orientation range from start | `0.071475 rad` |
| curvature proxy | `44.890160 rad/m` |
| SE(3) path length | `1.838658 m` |

The nearest-descriptor distance was `16.013720`. Historical leave-one-out
support thresholds were `0.516587` (p95) and `0.697260` (p99), so the actual
output is `OUTSIDE_EMPIRICAL_SUPPORT_RANGE` under this calibration.

## Comparison with Stage16 native benchmark

The Stage16 28-case ranges were:

| Feature | Stage16 range | Actual output |
|---|---:|---:|
| duration | `1.481482–5.610000 s` | `1.950000 s` |
| SE(3) path | `0.050312–0.728784 m` | `1.838658 m` |
| translation path | `0–0.695800 m` | `1.824632 m` |
| translation speed mean | `0–0.343516 m/s` | `0.935709 m/s` |
| translation speed p95 | `0–0.513281 m/s` | `1.815093 m/s` |
| angular speed mean | approximately `0–0.694245 rad/s` | `0.645301 rad/s` |
| angular speed p95 | approximately `0–0.977406 rad/s` | `1.699249 rad/s` |
| orientation range from start | `0–0.419883 rad` | `0.071475 rad` |
| curvature proxy | `0–29.235549 rad/m` | `44.890160 rad/m` |

Several motion descriptors exceed the tested native benchmark range. That
supports the gate's descriptive rejection and the need for robot-specific
adaptation; it does not support an assertion of model quality or task intent.

## Interpretation boundary

The descriptor distance is not `TRUE_TRAINING_OOD`: the actual training
visitation frequencies are unknown. It also cannot turn a finite output into
Go2+ARX5 task performance. The trajectory remained outside WBC execution by
the safety gate.

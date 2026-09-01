# Stage21 Actual-Model Full-Chunk Profile Audit

## Result

The six chunks produced during the actual same-scene receding run were
audited in full before any prefix execution. No smoothing, clipping, or
retiming was applied.

| Metric over 6 chunks | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Duration (s) | 1.9500 | 1.9500 | 1.9500 | 1.9500 |
| Translation path length (m) | 0.23177 | 0.23199 | 0.23574 | 0.23582 |
| Initial position jump vs achieved (m) | 0.001170 | 0.001171 | 0.001250 | 0.001261 |
| Initial rotation jump (rad) | 1.55e-8 | 1.05e-8 | 3.91e-8 | 4.21e-8 |
| Maximum linear speed (m/s) | 0.26627 | 0.26659 | 0.27182 | 0.27201 |
| Mean second-difference roughness (m) | 0.009839 | 0.009832 | 0.010043 | 0.010049 |

All six full chunks were finite and passed the test-only SafetyGate with no
correction. A same-scene GT desired label for these exact camera/state
observations was not available, so a Stage21 pred/GT path-length or speed
ratio is not fabricated. Stage20T's separate matched single-chunk result
(model path `0.233015 m` versus GT `0.060415 m`, ratio about `3.86`) remains
prior attribution evidence, not a Stage21 same-scene ratio.

## Chunk-handoff result

| Raw handoff metric over 5 updates | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Previous prediction -> new first position (m) | 0.04509 | 0.02276 | 0.12443 | 0.14829 |
| Previous prediction -> new first rotation (rad) | 0.11623 | 0.13480 | 0.15665 | 0.15702 |
| Achieved EEF -> new first position (m) | 0.001152 | 0.001152 | 0.001210 | 0.001215 |
| Achieved EEF -> new first rotation (rad) | 1.44e-8 | 0 | 3.97e-8 | 4.21e-8 |

The new chunks begin close to the current achieved EEF, but they do not
continue the previous predicted profile. This distinction is the central
Stage21 finding: the runtime can hand off a fresh chunk, while the unmodified
model's 40-point profile is not temporally/spatially coherent across
replanning boundaries.

## Native tracking during the diagnostic run

Over 24 native control rows, the consumed-reference-to-achieved metrics were:

| Metric | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Position error (m) | 0.03636 | 0.01476 | 0.15292 | 0.15445 |
| Rotation error (rad) | 0.07721 | 0.07023 | 0.15573 | 0.15702 |

The joint/base traces were finite. Base displacement/path length/yaw change
were `0.06591 m / 0.23586 m / 0.19119 rad`; arm joint path length was
`2.83690 rad`. The minimum arm-joint margin was `0.027996` in the delivered
native diagnostic. No fall, reset, invalid-state, or explicit native
joint-limit termination occurred.

This is a model-profile diagnostic, not a task-success measurement.

# Stage16 Initial Reference Continuity

## Definition

For every candidate native reference, Stage16 computes:

```text
initial_position_discontinuity = ||p_ref(0) - p_achieved_EEF(0)||
initial_rotation_discontinuity = SO3_geodesic(R_ref(0), R_achieved_EEF(0))
```

The audit covers all 114 seed-13 Stage15-bound candidates before progressive derivation.

## Distribution over 114 candidates

| Metric | Min | P25 | Median | P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| position (m) | 0 | 0 | 0 | 0.3049633 | 1.1173958 | 1.1173958 |
| rotation (rad) | 2.1073424e-08 | 2.1073424e-08 | 2.1073424e-08 | 2.1073424e-08 | 2.1073424e-08 | 2.1073424e-08 |

Position discontinuity was exactly zero for 64 of 114 candidates. Rotation was numerically continuous at the reset for all candidates.

## Native-class distribution

| Native class | Count | Position range / median (m) | Rotation (rad) |
|---|---:|---|---:|
| `NATIVE_ARM_EASY` | 70 | min 0, median 0, max 0.3049633 | 2.1073424e-08 |
| `NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT` | 17 | min 0.4728242, median 0.6002077, max 0.6002077 | 2.1073424e-08 |
| `NATIVE_BOUNDARY` | 3 | 0.4728242 for all | 2.1073424e-08 |
| `NATIVE_GEOMETRICALLY_INFEASIBLE` | 24 | min/median 0.3049633, max 1.1173958 | 2.1073424e-08 |

The 17 static-relocation candidates therefore had large parent-level initial position jumps. They were not directly scored as clean whole-body challenges.

## Progressive cohort effect

The selected eight derived references begin at the reset/current EEF. Their initial position discontinuity is `0 m` (within stored floating-point precision), and their initial rotation discontinuity is `2.1073424e-08 rad`. Each lead-in is SE(3)-continuous, retains the parent trajectory ID/hash, and leaves the challenge portion unchanged. The lead-in provenance and derived hashes are recorded in the node-local progressive manifest.

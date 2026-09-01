# Stage16 Progressive Relocation Cohort

## Selection

The native position-only reclassification supplied 17 `NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT` candidates. No candidate was labeled `NATIVE_CONTINUOUS_BASE_MOTION_REQUIRED`; this is a property of the existing E1/E2 shape-placement library, not a controller failure.

Eight candidates were selected for a clean progressive cohort. The original shape and placement files were not overwritten. Each derived trajectory has a new hash, a parent trajectory/hash, a smooth SE(3) lead-in, a join time, and an unchanged challenge suffix.

## Selected analysis

Values below use the derived timeline for the first reach-boundary time and the parent native position-only reach analysis for fixed-base feasibility. `rho` is the reach-table normalized burden; no static-base yaw oracle was available in the position-only Stage15 classification, so yaw is reported as unavailable rather than invented.

| ID | Lead-in / total duration (s) | First fixed-base-unreachable (s) | Fixed-base feasible fraction | Fixed-base max rho | Min static translation (m) | Best static offset (m) | Best static max rho |
|---|---:|---:|---:|---:|---:|---|---:|
| `E1_boundary_01` | 2.84 / 4.84 | 3.40 | 0.502488 | 1.015765 | 0.700000 | (0.70, 0.00) | 0.409833 |
| `E2_static_0_0` | 3.61 / 5.61 | 2.71 | 0.000000 | 1.120000 | 0.764853 | (0.75, 0.15) | 0.349778 |
| `E2_static_0_1` | 3.61 / 5.61 | 2.71 | 0.000000 | 1.148292 | 0.806226 | (0.80, 0.10) | 0.383273 |
| `E2_static_0_2` | 3.61 / 5.61 | 2.71 | 0.000000 | 1.120000 | 0.764853 | (0.75, 0.15) | 0.349778 |
| `E2_static_0_3` | 3.61 / 5.61 | 2.71 | 0.000000 | 1.120000 | 0.813941 | (0.80, 0.15) | 0.347275 |
| `E2_static_1_0` | 3.61 / 5.61 | 2.71 | 0.000000 | 1.120000 | 0.764853 | (0.75, 0.15) | 0.349778 |
| `E2_static_2_1` | 3.61 / 5.61 | 2.71 | 0.000000 | 1.148292 | 0.806226 | (0.80, 0.10) | 0.383273 |
| `E2_static_3_3` | 3.61 / 5.61 | 2.71 | 0.000000 | 1.120000 | 0.813941 | (0.80, 0.15) | 0.347275 |

The derived references are sampled at 100 Hz, have bounded maximum linear speed of approximately `0.25 m/s` and angular speed of approximately `0.65 rad/s`, and join with zero position error and numerical rotation continuity. The progressive controller rollouts showed a meaningful before/after reach-boundary change in base motion and task-space error; all eight were classified `BASE_RELOCATION_LIKE`. None was promoted to `CONTINUOUS_BASE_COORDINATION`, because the recorded controller behavior did not establish continuous coordination as a stronger claim.

## Provenance

Implementation: `src/go2arx5_vla/benchmark/progressive_relocation.py`.

Node-local manifest: `/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage16/progressive_relocation/progressive_cohort.json`.

Parent shapes remain under the Stage15 native benchmark directory. Derived files are under `/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage16/progressive_relocation/` and are marked `DERIVED_PROGRESSIVE_RELOCATION`.

## Optional continuous-base search

The bounded offline search did not produce a valid native `NATIVE_CONTINUOUS_BASE_CHALLENGE` from the current library. The result is recorded as `NO_VALID_NATIVE_CONTINUOUS_BASE_COHORT_FOUND`. No random or fabricated trajectory was added to fill the category.

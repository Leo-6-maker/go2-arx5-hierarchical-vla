# Stage16 Native Continuous Baseline

## Gate and protocol

All Stage16 baseline cases used `ALIGNED_NATIVE_CONTINUOUS_REFERENCE` with native IsaacGym/PhysX, the delivered URDF, delivered arm/dog checkpoints and config, one headless environment, seed 13, no camera, no writer, no GR00T, no P1M, and no hardware interface.

The first-eight gate completed with seven normal reference completions and one intentional boundary tracking abort. The gate had finite state/policy outputs, no invalid reset, no fall, and no unexplained lifecycle result. The full suite was then run.

## Full native result

| Quantity | Result |
|---|---:|
| trajectories | 28 |
| normal reference completions | 27 |
| tracking aborts | 1 (`E1_boundary_00`, stress control) |
| time-limit / fall / invalid / joint-limit termination | 0 / 0 / 0 / 0 |
| invalid-reset rows | 0 |
| reference mode | `ALIGNED_NATIVE_CONTINUOUS_REFERENCE` |

Case-average tracking metrics over the 28 seed-13 cases:

| Metric | Mean across cases | Minimum | Maximum |
|---|---:|---:|---:|
| position mean (m) | 0.0238690 | 0.0123066 | 0.1352272 |
| position p95 (m) | 0.0676702 | 0.0267713 | 0.3964850 |
| rotation geodesic mean (rad) | 0.1538881 | 0.1091481 | 0.3607483 |
| rotation p95 (rad) | 0.3821522 | 0.2806737 | 0.7300566 |
| time in 0.08 m/rad tolerance | 0.8018387 | 0.2800000 | 0.9103943 |
| path completion | 0.9605886 | 0.1143464 | 0.9999994 |

The boundary stress control is the maximum-error/minimum-completion case and is retained as a negative control, not silently removed. Tracking metrics exclude post-reference idle time.

## First-eight values

| Case | Position mean / p95 (m) | Rotation mean / p95 (rad) | Tolerance | Completion | Lifecycle |
|---|---:|---:|---:|---:|---|
| `E1_W0_00` | .012934 / .026771 | .136018 / .455563 | .7959 | .999986 | complete |
| `E1_W0_01` | .023743 / .045207 | .163689 / .382114 | .7347 | .991971 | complete |
| `E1_ang_0_00` | .012307 / .036553 | .132136 / .379779 | .8673 | .991736 | complete |
| `E1_curvature_1_00` | .022056 / .045709 | .155878 / .378932 | .7041 | .991971 | complete |
| `E1_coupling_1_00` | .017763 / .036481 | .182841 / .462604 | .7347 | .988941 | complete |
| `E1_boundary_01` | .018698 / .052410 | .124865 / .294356 | .9042 | .997530 | complete |
| `E2_static_0_0` | .015238 / .055319 | .121287 / .287206 | .9104 | .999999 | complete |
| `E1_boundary_00` | .135227 / .396485 | .360748 / .730057 | .2800 | .114346 | tracking abort |

## Whole-body outcome

The eight progressive cases had mean base path length `0.85213 m`, mean base displacement `0.42003 m`, and mean base yaw change `0.32792 rad`. Their mean position error was `0.01736 m` and rotation mean was `0.11948 rad`, with mean path completion `0.99944`. Before/after reach-boundary measures were recorded for every selected case; for example, `E2_static_0_0` had base motion `0.61005 m` before and `0.29900 m` after the boundary, while position error changed from `0.02428 m` to `0.00688 m`.

All eight progressive cases were classified `BASE_RELOCATION_LIKE`. This is evidence of base relocation behavior, not proof that the controller follows a unique geometric oracle base path or that it performs continuous base/arm coordination in the stronger sense.

## Robustness representatives

The representative set was also evaluated at reset seeds 14 and 15. No fall or invalid reset occurred. Seed 14 easy and curvature cases ended in native timing aborts; seed 15 completed those cases. The progressive case completed at both additional seeds, while the boundary control remained a tracking-abort stress control. This perturbation sensitivity is why the final Stage16 classification is `PASS WITH DEVIATION`, not an unconditional pass.

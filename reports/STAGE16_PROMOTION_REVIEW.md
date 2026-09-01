# Stage16 Promotion Review

Date: 2026-09-01 (Asia/Singapore)

## Decision

`NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`

This is a promotion of the Stage16 scientific interpretation, not a new
controller run. The Stage16 seed-13 native suite is frozen as the controller
baseline. The qualification is conditional on retaining the explicit boundary
limitation and on not interpreting the benchmark as robot-specific GR00T task
performance.

## Evidence reviewed

The existing Stage16 suite contains 28 native trajectories:

| Cohort | Count | Reference complete | Position mean (m) | Position p95 (m) | Rotation mean (rad) | Rotation p95 (rad) | Lag mean (s) | Base displacement (m) | Minimum arm margin (m) | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `NATIVE_ARM_EASY` | 18 | 18 | .01960 | .04448 | .15549 | .38774 | .03829 | .21592 | .02678 | all complete |
| `NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT` | 8 | 8 | .01736 | .05496 | .11948 | .28795 | .01599 | .42003 | -.07893 | all complete |
| `NATIVE_BOUNDARY` | 2 | 1 | .08834 | .32721 | .27714 | .70866 | .05911 | .25152 | .04472 | 1 complete, 1 `TRACKING_ABORT` |
| **all native cases** | **28** | **27** | **.02387** | **.06767** | **.15389** | **.38215** | — | — | — | **27 complete, 1 abort** |

The suite had zero falls, invalid resets, timeouts, and true joint-limit
terminations. The detailed per-case metrics remain in the node-local Stage16
manifest and metrics JSON; the figures above are a read-only aggregation of
those artifacts.

## Boundary case review

The single abort is `E1_boundary_00`. It is a deliberate native-boundary
stress control, not a nominal easy/curvature representative. Its exact
termination is `TRACKING_ABORT`, with position mean/p95/max
`.135227/.396485/.437015 m`, rotation mean/p95/max
`.360748/.730057/.755007 rad`, reference completion `false`, and path
completion `.114346`.

The Stage16 lifecycle snapshot attributes the abort to the timing condition:
the pre-reset timing error was `0.0445596687`, exceeding
`0.70 * traj_L = 0.0352189329` after grace. Lateral distance was
`.0467929579`, below its threshold. The row was not a reference completion,
fall, invalid state, timeout, or true joint-limit termination. Its initial
parent placement was also a large jump, so it must not be described as a clean
whole-body tracking challenge.

Classification: `EXPECTED_BOUNDARY_LIMITATION`, with
`GENERATOR_SUPPORT_BOUNDARY` retained as secondary context. All 28 benchmark
cases were outside the empirical generator-descriptor support range, so that
support label is not a unique causal explanation. It is not
`NOMINAL_TRACKING_FAILURE` and is not `UNKNOWN`.

## Progressive relocation validity

All eight derived progressive cases were checked using the existing detailed
logs, not base displacement alone. Each has zero initial position jump and
numerical rotation continuity, a bounded smooth lead-in, a meaningful base
motion change across the fixed-base reach boundary, finite internal
`dv_x/dv_y/dv_yaw` responses, coherent EEF tracking before and after the
boundary, and no fall/reset. All eight were classified
`BASE_RELOCATION_LIKE`; none is promoted to `CONTINUOUS_BASE_COORDINATION`.

The current E1/E2 library still has no valid native continuous-base cohort.
That absence is not a controller failure and no trajectory was fabricated to
fill the category.

## 40-point representation review

The matched Stage16 interface ablation remains adequate on the tested
distribution. The following are case-level distributions of per-case means;
the last two rows are paired 40-point-minus-continuous deltas:

| Quantity | Median | P95 | Max |
|---|---:|---:|---:|
| reconstruction position mean (m) | `2.43964e-5` | `1.98620e-4` | `2.07631e-4` |
| reconstruction rotation mean (rad) | `5.976e-9` | `8.65439e-5` | `9.98223e-5` |
| controller position delta mean (m) | `-5.88891e-6` | `2.67372e-6` | `4.11745e-6` |
| controller rotation delta mean (rad) | `2.96751e-5` | `1.88351e-4` | `2.15691e-4` |
| end-to-end position delta mean (m) | `-2.92425e-6` | `1.49714e-5` | `1.84976e-5` |
| end-to-end rotation delta mean (rad) | `2.96913e-5` | `1.88228e-4` | `2.15568e-4` |

The largest per-case reconstruction p95 was `5.54006e-4 m` and
`1.88898e-4 rad`; largest absolute reconstruction errors were `8.52269e-4 m`
and `2.01291e-4 rad`. The paired end-to-end penalty is negligible relative
to controller error in this synthetic reconstruction test. The supported
claim is only:

`40-POINT REPRESENTATION ADEQUATE ON TESTED STAGE16 DISTRIBUTION`

It is not arbitrary high-curvature equivalence and it is not evidence about
actual GR00T visual inference.

## Promotion boundary

Stage16 is therefore frozen as a qualified native general-WBC baseline with a
known boundary/timing limitation. Stage17 does not continue WBC research. It
uses this baseline to validate the provider, safety gate, and offline GR00T
interface. No P1M result, Stage14 unaligned result, or base GR00T tensor is
used as native WBC generalization evidence.

# Stage17 Trajectory Safety Gate

## Decision

`TrajectorySafetyGateV1` is a non-corrective validation boundary. Its result
is one of `ACCEPT`, `WARN`, or `REJECT`; it never clips, smooths, re-times, or
otherwise changes a GR00T trajectory.

Checks include:

- finite arrays and contract shapes;
- strictly increasing timestamps;
- quaternion/rot6d SO(3) validity;
- initial EEF position and rotation discontinuity when achieved EEF is given;
- linear and angular speed;
- path length and SE(3) arc-length degeneracy;
- optional workspace envelopes;
- chunk age, including aging and stale decisions;
- short or empty chunks.

## Threshold provenance

Every threshold is labelled in the result. The current defaults are explicitly
not real-robot limits:

| Check | Stage17 default | Provenance |
|---|---:|---|
| initial position warn/reject | `0.25 / 0.75 m` | `BENCHMARK_DERIVED` |
| initial rotation warn/reject | `0.50 / 1.50 rad` | `BENCHMARK_DERIVED` |
| linear speed warn/reject | `1.0 / 2.0 m/s` | `TEST_ONLY` |
| angular speed warn/reject | `2.0 / 4.0 rad/s` | `TEST_ONLY` |
| max path length | `5.0 m` | `TEST_ONLY` |
| minimum arc length | `1e-6 m` | `CONTROLLER_CONTRACT` |
| fresh/aging/stale | `0.20 / 0.75 s` | `CONTROLLER_CONTRACT` |
| workspace warning | optional | `GENERATOR_SUPPORT` |
| workspace rejection | optional | `TEST_ONLY` |
| production robot limits | not supplied | `REAL_ROBOT_TBD` |

## Actual base-model output

The actual output had zero initial position/rotation discontinuity and valid
finite SO(3) values. The gate returned:

```text
REJECT
  LINEAR_SPEED_REJECT: 2.1140129 m/s > 2.0 m/s (TEST_ONLY)
  ANGULAR_SPEED_WARN:  2.2792987 rad/s > 2.0 rad/s (TEST_ONLY)
```

This is a safe descriptive result for an unadapted base checkpoint, not an
integration failure. The trajectory was not modified and no WBC call was
made.

## Test coverage

CPU tests cover finite acceptance, duplicate timestamps, initial jumps,
staleness, workspace warning/rejection, NaN payloads, SO(3) conversion, and
provenance preservation.

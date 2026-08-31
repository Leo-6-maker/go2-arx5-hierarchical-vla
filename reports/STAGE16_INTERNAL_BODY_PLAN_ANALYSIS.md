# Stage16 Internal Body Plan Analysis

## Contract

The internal plan is the explicit nine-dimensional body/gait command produced inside the native WBC interface. It is not a latent representation and is not an externally required GR00T output.

| Index | Channel |
|---:|---|
| 0 | `dv_x` |
| 1 | `dv_y` |
| 2 | `dv_yaw` |
| 3 | `body_height` |
| 4 | `body_pitch` |
| 5 | `body_roll` |
| 6 | `gait_frequency` |
| 7 | `stance_width` |
| 8 | `stance_length` |

## Finite/bounded check

All 28 baseline cases had finite values for all nine channels. Across the full seed-13 suite, normalized channel extrema were:

| Channel | Min | Max |
|---|---:|---:|
| `dv_x` | -0.987861 | 0.999971 |
| `dv_y` | -0.758183 | 0.823825 |
| `dv_yaw` | -0.540546 | 0.614102 |
| `body_height` | -0.873037 | 0.971228 |
| `body_pitch` | -0.999952 | 0.930270 |
| `body_roll` | -0.976882 | 0.767111 |
| `gait_frequency` | -0.694975 | 0.986953 |
| `stance_width` | -0.972137 | 0.620523 |
| `stance_length` | -0.264920 | 0.910323 |

## Progressive before/after response

Mean normalized values across the eight progressive cases, split at the first fixed-base reach boundary:

| Channel | Before | After |
|---|---:|---:|
| `dv_x` | 0.404314 | 0.264818 |
| `dv_y` | -0.094860 | 0.177143 |
| `dv_yaw` | 0.084209 | 0.019945 |
| `body_height` | 0.437682 | 0.658230 |
| `body_pitch` | 0.162410 | -0.136387 |
| `body_roll` | -0.169117 | 0.401171 |
| `gait_frequency` | 0.257500 | 0.222791 |
| `stance_width` | -0.155562 | 0.005290 |
| `stance_length` | -0.005543 | 0.016869 |

Descriptive per-case correlations between the recorded reach burden and channel output, averaged over progressive cases, were: `dv_x -0.2878`, `dv_y 0.7341`, `dv_yaw -0.4427`, body height `0.4406`, body pitch `-0.7189`, body roll `0.8689`, gait frequency `-0.3429`, stance width `0.7250`, and stance length `-0.2058`.

These correlations are observational summaries of controller output, not causal claims. They show that body/base channels changed while reach burden changed, but do not establish a unique internal planning policy.

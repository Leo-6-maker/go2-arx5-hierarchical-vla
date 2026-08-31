# Stage16 Static Termination Root Cause

## Case and evidence

The Stage15 `E2_static_0_0` result was rerun from the original Stage15 bound input, without changing native controller semantics:

`/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage16/lifecycle/e2_static_0_0_original_bound`

The launcher captured a pre-reset termination snapshot. The old Stage15 row 25 was post-reset and had already lost the final pre-reset values; it is not used as the root-cause evidence.

The “step 25” below means the 25th logged control sample (zero-based source step 24):

| Signal | Pre-reset value |
|---|---:|
| `sim_time` | `0.5200000405 s` |
| `reference_duration` | `2.0 s` |
| `traj_L` | `0.0503127612` |
| `traj_s` | `0.0503127612` |
| `traj_s_ref` | `0.0057530920` |
| `traj_timing_err` | `0.0445596687` |
| timing threshold (`0.70 * traj_L`) | `0.0352189329` |
| `traj_d_lat` | `0.0467929579` |
| `reset_buf` | `true` |
| `time_out_buf` | `false` |
| `fall_flag` | `false` |
| `invalid_state` | `false` |
| `joint_limit_violation` | `true` diagnostic only |
| `joint_limit_termination` | `false` |
| `reference_complete` | `false` |
| `episode_done` | `true` |
| termination code | `TRACKING_ABORT` |

## Root cause

The timing error exceeded the configured early-termination threshold after the grace period. The lateral-distance value was far below its `1.0 m` threshold. The episode was not reference-complete, not a time limit, not a fall, not an invalid state, and not a joint-limit termination.

`traj_s == traj_L` is the forward projection saturating at the path endpoint; it is not evidence that the time-law reference completed. At the same sample `traj_s_ref` was only `0.00575`, while the reference duration was `2.0 s`.

## Initial jump

For the original bound placement:

- position discontinuity between reset achieved EEF and the first reference point: `0.6002077383 m`;
- rotation discontinuity: `2.1073424e-08 rad`.

This placement therefore began with a large position jump. It was an unsuitable unmodified whole-body challenge, and it likely contributed to the early timing abort. Stage16 does not relabel that parent input as a successful progressive relocation case. The selected derived progressive version adds a provenance-preserving smooth lead-in and is analyzed separately.

## Consequence

The lifecycle implementation and report distinguish normal reference completion from tracking failure. The native early-termination threshold remains part of the delivered runtime; Stage16 does not disable or rewrite it. A storage-capacity adjustment was used only for long derived references so that the delivered `max_tl_points=512` allocation did not silently truncate input samples; it did not alter physics, controller semantics, or termination thresholds.

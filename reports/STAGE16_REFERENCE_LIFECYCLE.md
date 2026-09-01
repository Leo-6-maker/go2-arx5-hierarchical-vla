# Stage16 Reference Lifecycle

## Decision

Native rollout completion is determined by the explicit episode/reference lifecycle, not by a fixed 100-control-step assumption. The benchmark records the reference time-law interval, termination signals, a pre-reset snapshot, and a normalized termination code before the environment reset can erase the final state.

## Source audit

The delivered source was read at:

- `WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py:1060-1092`: advances `traj_sim_time`, reads `s_ref`, computes forward-projected `traj_s`, lateral distance `traj_d_lat`, reference progress `traj_s_ref`, and timing error `traj_timing_err`.
- `WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py:1122-1154`: applies grace, lateral-distance, and timing early-termination checks.
- `WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py:1259-1285`: distinguishes trajectory success from early abort and timeout.
- `WBCTrackingShare/go1_gym/envs/roboduet/legged_robot.py:373-419`: checks termination and immediately calls `reset_idx`, including timeout/fall and the arm termination hook.
- Native config `scripts/config/wbc.py:477-535`: preview/time-law settings, success criteria, `terminate_d_lat=1.00 m`, `terminate_timing_frac=0.70`, and `terminate_grace=0.50 s`.

`traj_s` is a forward projection and can saturate at `traj_L` even while the time-law reference is near its beginning. Therefore `traj_s >= traj_L` is not used as a standalone completion predicate.

## Codes

| Code | Meaning | Rule used in Stage16 |
|---|---|---|
| `REFERENCE_COMPLETE` | reference time interval reached within the control-step tolerance | authoritative reference duration reached and no earlier terminal condition |
| `TIME_LIMIT` | simulator/environment time limit | timeout flag or time-limit signal |
| `FALL` | body-height/fall condition | explicit fall flag |
| `INVALID_STATE` | non-finite or invalid simulator/controller state | explicit invalid-state signal |
| `JOINT_LIMIT` | true joint-limit termination | explicit joint-limit termination signal only |
| `TRACKING_ABORT` | native trajectory tracking early abort | timing/lateral early-abort signal |
| `NATIVE_TASK_TERMINATION` | other explicit native task termination | native task signal not covered above |
| `UNKNOWN` | reset/termination without an identifiable reason | fallback only |

A negative joint-limit margin is retained as a diagnostic metric. The delivered runtime did not set a joint-limit termination flag for these rows; margin violation is therefore not reclassified as `JOINT_LIMIT`.

## Observed lifecycle

The 28-case seed-13 native suite produced 27 `REFERENCE_COMPLETE` cases and one `TRACKING_ABORT` stress control. There were zero timeout, fall, invalid-state, or true joint-limit termination cases. The 40-point matched suite had the same lifecycle breakdown. All records include `episode_done`, `reset_buf`, `time_out_buf`, reference completion, and the termination code.

The first-eight gate had seven normal completions and one expected boundary tracking abort. The abort was not hidden by padding the rollout to 100 steps.

## Perturbation lifecycle

For the four representative cases at reset seeds 14 and 15, no fall or invalid reset occurred. Seed 14 produced native timing aborts on the easy and curvature representatives despite modest task-space error; the progressive case completed and the boundary control aborted. Seed 15 completed easy, curvature, and progressive cases and aborted the boundary control. This is a real reset/initial-condition sensitivity and remains a Stage16 deviation.

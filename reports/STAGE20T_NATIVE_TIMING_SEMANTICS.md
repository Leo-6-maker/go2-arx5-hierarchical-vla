# Stage20T Native WBC Timing Semantics

Date: 2026-09-01 (Asia/Singapore)

## Scope

This report records the read-only source audit used by the Stage20T matched
timing experiment. The native WBC source, checkpoint, timing threshold, and
termination semantics were not modified.

## Source execution order

The delivered native source was audited at:

- `WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py:1023-1045`: the
  post-physics hook refreshes the EEF, advances the reference target, updates
  task-space error and trajectory progress, and returns from the trajectory
  path.
- `WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py:1060-1067`: each native
  control step increments `traj_sim_time` by `dt`, evaluates `s_ref`, and
  writes the current desired pose.
- `WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py:1069-1082`: `traj_s` is
  a forward-window projection of the achieved EEF onto the canonical path;
  `traj_timing_err = traj_s - s_ref`.
- `WBCTrackingShare/go1_gym/envs/roboduet/wbc_env.py:1122-1154`: after a
  `0.50 s` grace period, native early termination checks `traj_d_lat > 1.00 m`
  and `abs(traj_timing_err) > 0.70 * traj_L`.
- `scripts/config/wbc.py:497-535`: the delivered success and early-abort
  thresholds are declared. The timing threshold is a fraction of the
  trajectory arc length.

The project-owned launcher captures a pre-reset termination snapshot. This is
necessary because the base environment resets the state immediately after
setting the reset flag. `REFERENCE_COMPLETE` is assigned from the reference
time interval after no native termination event is present; it is not inferred
from a fixed 100-step rollout or from `traj_s >= traj_L` alone.

## Definitions used in Stage20T

```text
traj_timing_err       = traj_s - traj_s_ref
timing_error_fraction  = abs(traj_timing_err) / traj_L
timing_margin_m        = 0.70 * traj_L - abs(traj_timing_err)
timing_margin_fraction = 0.70 - timing_error_fraction
```

The complete per-step trace retains `traj_s`, `traj_s_ref`, their signed gap,
timing error, normalized fraction, margin, lateral distance, desired/achieved
EEF, EEF velocities, base state, arm/dog state, actions, and all nine internal
body/gait plan channels. The authoritative node-local artifacts are under:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20t/timing_sensitivity_v1_full/
```

## Important native preprocessing caveat

External references are passed through the delivered `to_canonical()` function
before native WBC consumption. At `T=40`, it applies its existing
Butterworth/zero-phase preprocessing when `butter_fc < fs / 2`:

```text
to_canonical(): modules/trajectory.py:428-495
```

With the declared test period of `0.05 s`, scale `0.75` uses `dt=0.0375 s`,
`fs=26.67 Hz`, and therefore enters that existing filter branch. Scales
`1.00`, `1.25`, `1.50`, and `2.00` do not. The harness leaves the candidate
position/orientation bytes unchanged, but the native internal canonical path
can differ at `0.75`:

| Condition | 0.75x native `traj_L` (m) | 1.00x native `traj_L` (m) |
|---|---:|---:|
| A model | 0.180427 | 0.233015 |
| B GT | 0.060421 | 0.060415 |

Consequently, `0.75x` is retained as a declared derived timing diagnostic,
not as a clean pure-time scaling result. The primary model-vs-GT attribution
uses the unfiltered `1.00x`–`2.00x` range.

## Reset sensitivity

The native configuration still contains embodiment/reset randomization in the
delivered path, including mount position/rotation and stage-2 arm property
randomization (`go1_gym/envs/config/wbc.py:231-240, 343-350`). The project
launcher disables the unrelated top-level perturbations used for evaluation,
but it does not rewrite the delivered nested native configuration. Therefore
the three fixed seeds are a matched native-reset sensitivity sample, not three
identical initial states.

Stage20T adds a second continuity measurement from the first post-physics
rollout row. This is deliberately separate from the pre-run candidate audit,
which compares the source observation state to the first candidate point.

## Decision

The native timing envelope is a property of the delivered arc-length progress
and reset lifecycle. Stage20T does not raise the `0.70` threshold, disable the
termination hook, clip/re-time a model trajectory, or change a checkpoint.

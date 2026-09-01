# Stage15 orientation-only semantics

## Stage14 termination diagnosis

The earlier orientation-only injection ended at 25/100 steps. The native
termination record was:

```text
traj_L       = 0.0503294 m (composite SE(3) length)
traj_s       = 0.0503294 m
traj_s_ref   = 0.0106242 m
timing_err   = 0.0397052 m
threshold    = 0.0352306 m
timing cause = true
distance-lateral cause = false
```

This was a progress/timing mismatch in the unaligned placement/injection, not
a simulator crash, NaN, or Isaac lifecycle failure.

## Native semantics

The native arc-length metric includes the rotational term
`0.15*dtheta`, so a changing-orientation path has finite composite progress
even when translation is stationary. Preview positions hold while preview
orientations advance through quaternion SLERP. A trajectory must still have an
explicit duration/time law; disabling the termination rule would hide a
reference contract issue and was not used for the aligned result.

Classification:

```text
SUPPORTED_WITH_EXPLICIT_DURATION
```

## Aligned result

After binding the initial pose and orientation to the native reset TCP, the
orientation case `E1_ang_0_00` completed 100/100 control steps with natural
termination. Position mean/p95 was `0.0122/0.0364 m`; rotation mean/p95 was
`0.1320/0.3753 rad`; no NaN or reset loop occurred. This supports native use of
the class, while not claiming that every orientation profile is within policy
limits.

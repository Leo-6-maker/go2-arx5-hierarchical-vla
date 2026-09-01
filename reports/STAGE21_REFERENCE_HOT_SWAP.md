# Stage21 Reference-Only Receding-Horizon Hot-Swap

## Contract

`RecedingHorizonExecutorV1` and `ReferenceUpdateV1` make the replacement
boundary explicit. An accepted update must preserve:

```text
physics_reset        = false
physics_teleport     = false
policy_history_reset = false
```

The validation set declared `K in {4,6,8,10}` and froze `K=4` for the
reported primary runs. The native control period, rather than the chunk
sample period, determines executed prefix duration:

```text
native dt       = 0.01999999955 s
K=4 duration    = 0.0799999982 s
chunk period    = 0.05 s, TEST_ONLY
```

The lifecycle CPU tests cover `FRESH`, `AGING`, `STALE`, and rejection of
`OUT_OF_ORDER` and invalid prefix/update cases. No transport was introduced.

## GT-first result

Artifact:

```text
/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/receding_horizon/gt_k4_20260901_telemetry/gt_receding_summary.json
```

Result:

```text
GT_RECEDING_HORIZON_INFRASTRUCTURE_PASS
24/24 valid steps
5 reference updates after the initial chunk
0 reset events
```

All updates were `FRESH`; root state, DOF state, episode length, and arm/dog
policy histories remained unchanged at each replacement. The executor and
native loader therefore demonstrated reference-only hot-swap in one physics
episode.

The raw previous-prediction-to-new-first position discontinuities were
`0.000400`, `0.000879`, `0.001479`, `0.002147`, and `0.002839 m`; the
corresponding rotation discontinuity was approximately `2.1e-8 rad`.
Achieved-to-new-first discontinuity was much larger in this GT replay because
the replay chunks were generated from a fixed Stage16 schedule rather than
freshly conditioned on each observed native state. That is an expected
limitation of this infrastructure control, not a reason to call hot-swap
broken.

## Actual model use

The actual-model runner used the same executor and native reference loader.
Every model chunk was gated before binding and WBC execution. Its separate
profile and handoff metrics are in `STAGE21_MODEL_PROFILE_AUDIT.md` and
`STAGE21_NATIVE_RECEDING_HORIZON.md`.

# Stage20T Timing Sensitivity Matrix

## Protocol

The complete matrix contains 33 native simulation runs:

```text
A_MODEL:             5 scales x 3 seeds = 15
B_GT:                5 scales x 3 seeds = 15
C_STAGE16_CONTROL:   1 scale  x 3 seeds = 3
```

The declared original action period is `0.05 s`, provenance
`TEST_ONLY_SIM_VISUAL_REACH_V0`. Scale values are applied only to the
timestamp grid before provider construction. All non-unit rows are labeled
`DERIVED_TIMING_DIAGNOSTIC`.

## Completion matrix

`complete/3` is the number of `REFERENCE_COMPLETE` rollouts; the remainder
are native `TRACKING_ABORT`.

| Condition | 0.75x | 1.00x | 1.25x | 1.50x | 2.00x |
|---|---:|---:|---:|---:|---:|
| A model | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| B GT | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| C Stage16 control | — | 2/3 | — | — | — |

Overall: 14 `REFERENCE_COMPLETE`, 19 `TRACKING_ABORT`, and zero
`TIME_LIMIT`, `FALL`, `INVALID_STATE`, or true `JOINT_LIMIT` terminations.
All 33 native processes launched and produced a trace; all 33 interface
safety-gate decisions were `ACCEPT`.

## Descriptor changes

The source position/orientation geometry hash is invariant across timing
scales within each condition. The declared input descriptors are:

| Condition | path length (m) | duration at 0.75/1.00/1.25/1.50/2.00 (s) | max linear speed at 1.00x (m/s) |
|---|---:|---|---:|
| A model | 0.233015 | 1.4625 / 1.9500 / 2.4375 / 2.9250 / 3.9000 | 0.310131 |
| B GT | 0.060415 | 1.4625 / 1.9500 / 2.4375 / 2.9250 / 3.9000 | 0.046463 |
| C control | 0.050312 | — / 2.0000 / — / — / — | 0.047164 |

The native canonical `traj_L` differs at A/0.75x because the delivered
filter branch is time-step dependent. This is documented in
`STAGE20T_NATIVE_TIMING_SEMANTICS.md` and prevents treating that one point as
a pure temporal ablation.

## Original timing tracking summaries

These are statistics over the three per-seed rollout means, not a pooled
per-step metric.

| Condition | position mean median / p95 / max (m) | rotation mean median / p95 / max (rad) | per-run min timing margin median / p95 / max (m) |
|---|---:|---:|---:|
| A model, 1.00x | 0.053493 / 0.056019 / 0.056300 | 0.405924 / 0.489785 / 0.499103 | -0.012565 / -0.008296 / -0.007821 |
| B GT, 1.00x | 0.022166 / 0.041919 / 0.044114 | 0.212560 / 0.467275 / 0.495577 | 0.027037 / 0.030617 / 0.031014 |
| C control, 1.00x | 0.020845 / 0.041408 / 0.043692 | 0.193560 / 0.456363 / 0.485563 | 0.022557 / 0.025611 / 0.025950 |

The A model original-timing per-run maximum absolute timing fractions were
`0.733565`, `0.753924`, and `0.977684` for seeds 13, 14, and 15. The B GT
values were `0.186645`, `0.252483`, and `0.998449`; C values were `0.184231`,
`0.251655`, and `0.999912`.

## Artifact authority

The machine-readable manifest and analysis are node-local:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20t/timing_sensitivity_v1_full/stage20t_timing_sensitivity_manifest.json
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20t/timing_sensitivity_v1_full/stage20t_timing_analysis.json
```

Every run also retains `native_external_rows.jsonl`,
`native_external_rollout.npz`, the native summary, stdout log, candidate NPZ,
and interface seal. These private/large artifacts are intentionally not Git
tracked.

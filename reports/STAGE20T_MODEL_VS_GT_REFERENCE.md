# Stage20T Model-vs-GT Reference Attribution

## Matched design

The experiment compared three conditions using the same native runner,
checkpoint pair, simulator, asset, provider/binding/adapter chain, timing
grid, and reset seed:

- `A_MODEL`: the exact decoded Stage20S actual-model `[40,10]` output replayed
  without spatial correction;
- `B_GT`: the same observation's future desired EEF label `[40,10]`, using the
  same provider path and timing;
- `C_STAGE16_CONTROL`: the known-good Stage16 `E1_W0_00` native control.

All 33 safety-gate decisions were `ACCEPT`. No model trajectory was silently
modified. The original declared timing is `1.00x`; all other scales are
`DERIVED_TIMING_DIAGNOSTIC`.

## Original 1.00x matched results

The first post-physics discontinuity is the distance/geodesic angle between
the desired and achieved EEF in the first authoritative rollout row. It is
not the same as the pre-run candidate audit.

| Seed | A model result | A first jump (m/rad) | B GT result | B first jump (m/rad) | C control |
|---:|---|---:|---|---:|---|
| 13 | `TRACKING_ABORT`, first crossing step 20, terminal step 24 | 0.0433 / 0.0985 | `REFERENCE_COMPLETE`, 96 rows | 0.0366 / 0.0456 | complete |
| 14 | `TRACKING_ABORT`, first crossing step 15, terminal step 24 | 0.2173 / 0.7022 | `REFERENCE_COMPLETE`, 96 rows | 0.2149 / 0.7122 | complete |
| 15 | `TRACKING_ABORT`, first crossing step 0, terminal step 24 | 0.1992 / 1.9512 | `TRACKING_ABORT`, first crossing step 0 | 0.1991 / 1.9480 | abort step 0 |

For seed 13, A crossed the threshold at `t=0.44 s` with
`traj_d_lat=0.02285 m`, `traj_s=0.19297 m`, and normalized timing error
`0.73356`; the terminal margin was `-7.82 mm`. For seed 14, the first
crossing was earlier at `t=0.34 s`, with `traj_d_lat=0.00805 m` and fraction
`0.74644`. These are timing-progress failures, not lateral-distance failures.

Seed 14 has a large initial reset jump in both A and B, yet B completes and C
completes. Thus that initial jump alone is not sufficient to explain the
model-vs-GT difference. Seed 15 is different: B and C also start with the
whole path projected at/near its end (`traj_s=0.06042 m` and `0.05031 m`,
respectively) while `s_ref` is near zero. Their first-step timing fractions are
`0.99845` and `0.99991`. This is a native reset/initial-condition boundary and
is not attributed to the model.

## Model profile evidence

At the unfiltered original timing, A completed `0/3`; B completed `2/3`; C
completed `2/3`. The same A/B difference holds for seeds 13 and 14 at
`1.25x`, `1.50x`, and `2.00x`: A is `0/3` at each scale, while B is `2/3` at
each scale. The two successful B seeds are the same two seeds that pass the
control; seed 15 remains the reset boundary at every scale.

The model path is materially more demanding than the same-observation GT
label:

| Original 1.00x descriptor | A model | B GT |
|---|---:|---:|
| translation path length (m) | 0.233015 | 0.060415 |
| duration (s) | 1.950 | 1.950 |
| max linear speed (m/s) | 0.310131 | 0.046463 |
| source initial audit position jump (m) | 0.000608 | 0 |
| source initial audit rotation jump (rad) | 0 | 0 |

The native trace therefore supports the narrow conclusion:

```text
MODEL_TRAJECTORY_PROFILE_LIMITATION_IDENTIFIED
```

with a separate:

```text
NATIVE_RESET_INITIAL_CONDITION_SENSITIVITY_CHARACTERIZED
```

It does not support the stronger claim that the WBC cannot track the task, nor
does it support a production timing or real-robot speed limit.

## 0.75x interpretation

A completed 2/3 at `0.75x`, while B also completed 2/3. This is not treated as
evidence that speeding the model path fixes the native timing issue, because
the native `to_canonical()` filter branch is active at that scale and changes
the model path length from `0.233015 m` to `0.180427 m`. Scales `1.00x`–`2.00x`
remain the clean attribution range for the model profile.

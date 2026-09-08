# Stage20T Repeatability and Boundary Trace Review

## Fixed seeds

Seeds `13`, `14`, and `15` were run for A and B at every timing scale and for
the C Stage16 control at the original timing. A seed was passed to Python,
NumPy, PyTorch, and CUDA before native environment construction. The seed
comparison is matched between A/B/C, but it is not a claim that all three
resets have identical physical initial states.

## Original timing by seed

| Seed | A model | B GT | C control | Interpretation |
|---:|---|---|---|---|
| 13 | abort at step 24; first timing crossing step 20 | complete at step 95 | complete at step 97 | cleanest attribution pair |
| 14 | abort at step 24; first timing crossing step 15 | complete at step 95 | complete at step 97 | large reset jump, but GT/control recover |
| 15 | abort; first timing crossing step 0 | abort; first crossing step 0 | abort; first crossing step 0 | shared reset/initial-condition boundary |

The first post-physics position/rotation discontinuities were:

| Seed | A model (m/rad) | B GT (m/rad) | C control (m/rad) |
|---:|---:|---:|---:|
| 13 | 0.0433 / 0.0985 | 0.0366 / 0.0456 | 0.0382 / 0.0543 |
| 14 | 0.2173 / 0.7022 | 0.2149 / 0.7122 | 0.2153 / 0.7145 |
| 15 | 0.1992 / 1.9512 | 0.1991 / 1.9480 | 0.2005 / 1.9555 |

The pre-run candidate audit remained small for the model source observation
(`0.000608 m`, `0 rad`) and zero for the GT label. The much larger seed-14/15
values above arise after the native reset/anchor path and are the values that
must be used for runtime continuity claims.

## Sudden crossing versus accumulated lag

The model seed-13 original-timing trace was not a one-step lateral failure. It
held `traj_d_lat` near `0.023 m` when the timing threshold was first crossed
at `t=0.44 s`; `traj_s=0.19297 m` was already well ahead of
`traj_s_ref=0.02204 m`. The terminal row at `t=0.52 s` had timing margin
`-0.00415 m` and lateral distance `0.02887 m`.

Seed 14 crossed at `t=0.34 s` with lateral distance `0.00805 m`. Seed 15
crossed at the first row: the forward projection was already at the full
candidate path length while the reference time law was near zero. This same
first-row signature appeared in B and C, making it a reset/initial-condition
case rather than a learned-model attribution.

The analyzer records the complete trace and the last ten terminal-aware rows
for every run. It does not discard pre-termination rows or replace a partial
rollout with a fixed-length padded result.

## Timing-scale repeatability

- A model: `0/3` complete at `1.00x`, `1.25x`, `1.50x`, and `2.00x`; `2/3`
  complete at `0.75x`.
- B GT: `2/3` complete at every scale; seed 15 aborts at the initial reset
  boundary.
- C control: `2/3` complete at `1.00x`; seed 15 aborts at the same boundary.

The A/0.75x completions are not used to claim a timing rescue because native
`to_canonical()` filtering is active at that timestep. The unfiltered A
1.00x–2.00x failures remain repeatable for all three seeds, with the seed-15
row separately identified as a shared reset sensitivity.

## Repeatability conclusion

The evidence supports two simultaneous findings:

```text
MODEL_TRAJECTORY_PROFILE_LIMITATION_IDENTIFIED
NATIVE_RESET_INITIAL_CONDITION_SENSITIVITY_CHARACTERIZED
```

It does not justify merging the seed-15 reset boundary into the model failure
rate, and it does not justify relaxing the native timing termination.

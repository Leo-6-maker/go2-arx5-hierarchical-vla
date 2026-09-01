# Stage20T Final Report — Native WBC Timing-Sensitivity Qualification

Date: 2026-09-01 (Asia/Singapore)

## A. Classification

`PASS WITH DEVIATION`

The predeclared 33-run matched timing matrix completed in simulation and
produced authoritative per-step traces. The experiment isolated a
model-trajectory profile limitation from a native reset/initial-condition
boundary. It did not qualify the unchanged actual-model visual-reach rollout
for end-to-end promotion.

Primary result:

```text
MODEL_TRAJECTORY_PROFILE_LIMITATION_IDENTIFIED
```

Secondary result:

```text
NATIVE_TIMING_BOUNDARY_CHARACTERIZED
```

## B. Stage16/Stage20S baseline promotion

The frozen Stage16 result remains:

```text
NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION
```

Stage20T did not rerun or modify the 28-trajectory Stage16 benchmark, the
native controller, or either checkpoint. Stage20S remains a simulation-only
single-chunk model-to-WBC result with a native timing abort; it is not a
general task-performance claim.

## C. Protocol and matrix

| Condition | Scales | Seeds | Runs | Reference complete | Tracking abort |
|---|---|---|---:|---:|---:|
| A actual Stage20S model output | 0.75, 1.00, 1.25, 1.50, 2.00x | 13, 14, 15 | 15 | 2 | 13 |
| B same-observation GT desired label | 0.75, 1.00, 1.25, 1.50, 2.00x | 13, 14, 15 | 15 | 10 | 5 |
| C Stage16 known-good control | 1.00x | 13, 14, 15 | 3 | 2 | 1 |
| **Total** |  |  | **33** | **14** | **19** |

The original test period was `0.05 s` with provenance
`TEST_ONLY_SIM_VISUAL_REACH_V0`. All non-unit scales were labeled
`DERIVED_TIMING_DIAGNOSTIC`. All 33 interface gates returned `ACCEPT`; no
trajectory was clipped, corrected, or silently re-timed.

## D. Reference lifecycle

Native completion uses the explicit reference time interval. Native early
termination uses the delivered `0.70 * traj_L` timing rule and `1.00 m`
lateral-distance rule after the `0.50 s` grace period. The launcher retained
pre-reset terminal snapshots and emitted `REFERENCE_COMPLETE` separately from
`TRACKING_ABORT`.

Across the matrix there were zero `TIME_LIMIT`, `FALL`, `INVALID_STATE`, or
true `JOINT_LIMIT` terminations. The complete source audit and equations are
in `STAGE20T_NATIVE_TIMING_SEMANTICS.md`.

## E. Matched attribution

At original timing, A completed `0/3`, B completed `2/3`, and C completed
`2/3`:

| Seed | A model | B GT | C control |
|---:|---|---|---|
| 13 | abort at step 24; first timing crossing step 20 | complete | complete |
| 14 | abort at step 24; first timing crossing step 15 | complete | complete |
| 15 | shared reset boundary, first crossing step 0 | shared reset boundary, first crossing step 0 | shared reset boundary, first crossing step 0 |

For seeds 13 and 14, the same native WBC and timing complete B while A
aborts. A is materially different from B: path length `0.233015 m` versus
`0.060415 m`, and maximum linear speed `0.310131 m/s` versus `0.046463 m/s`.
This is sufficient for the narrow model-profile attribution.

Seed 15 is not counted as model evidence: A, B, and C all enter the first
post-physics row with the path projection at/near its endpoint and timing
fraction near one. This is a shared native reset/initial-condition boundary.

## F. Timing boundary and Stage16 envelope

The model seed-13 original trace first crossed at `t=0.44 s` with
`traj_d_lat=0.02285 m`, timing fraction `0.733565`, and terminal margin
`-0.007821 m`. Seed 14 crossed at `t=0.34 s` with `traj_d_lat=0.00805 m` and
fraction `0.746443`. Neither is a lateral-distance termination.

The two completed Stage20T controls observed maximum timing fractions
`0.184231` and `0.251655`, with minimum margins `0.025950 m` and `0.022557 m`.
The third control seed crossed at step 0 because of the shared reset boundary.
This control envelope is a small repeatability sample, not a replacement for
the full Stage16 distribution.

The model A failures persist for all three seeds at unfiltered `1.00x`–`2.00x`.
B completes for seeds 13/14 at all those scales. The `0.75x` A completions are
not treated as a pure timing rescue: the delivered native `to_canonical()`
filter branch is active there and changes A's internal `traj_L` from
`0.233015 m` to `0.180427 m`.

## G. Tracking and whole-body telemetry

At original timing, statistics over the three per-seed rollout means were:

| Condition | position mean median / p95 / max (m) | rotation mean median / p95 / max (rad) | min timing margin median / p95 / max (m) |
|---|---:|---:|---:|
| A model | 0.053493 / 0.056019 / 0.056300 | 0.405924 / 0.489785 / 0.499103 | -0.012565 / -0.008296 / -0.007821 |
| B GT | 0.022166 / 0.041919 / 0.044114 | 0.212560 / 0.467275 / 0.495577 | 0.027037 / 0.030617 / 0.031014 |
| C control | 0.020845 / 0.041408 / 0.043692 | 0.193560 / 0.456363 / 0.485563 | 0.022557 / 0.025611 / 0.025950 |

For the Stage20S A seed-13 replay, native base displacement/path/yaw were
`0.06508 m / 0.23732 m / 0.40793 rad`. All nine internal body/gait plan
channels were finite and varied in every inspected original-timing run. This
shows that native WBC body-plan execution was active; it does not show that
GR00T learned whole-body coordination.

## H. Initial continuity and reset sensitivity

The pre-run candidate audits were small for the model source state (`0.608
mm`, `0 rad`) and zero for the GT label. The actual first post-physics
runtime jumps include native reset/anchor effects:

| Seed | A model position/rotation | B GT position/rotation | C control position/rotation |
|---:|---:|---:|---:|
| 13 | 0.0433 m / 0.0985 rad | 0.0366 m / 0.0456 rad | 0.0382 m / 0.0543 rad |
| 14 | 0.2173 m / 0.7022 rad | 0.2149 m / 0.7122 rad | 0.2153 m / 0.7145 rad |
| 15 | 0.1992 m / 1.9512 rad | 0.1991 m / 1.9480 rad | 0.2005 m / 1.9555 rad |

The first-row measurement is now present in the machine-readable analysis and
is not conflated with the source-observation audit.

## I. 40-point representation claim

The Stage16 matched ablation remains the authority:

```text
40-POINT REPRESENTATION ADEQUATE ON TESTED STAGE16 DISTRIBUTION
```

The tested-distribution paired end-to-end additional error was approximately
`1.07e-6 m` mean position and `4.45e-5 rad` mean rotation; the full median,
p95, and max reconstruction/controller/end-to-end values remain in
`reports/STAGE16_40POINT_ABLATION.md`. This is not arbitrary high-curvature
equivalence and Stage20T does not change the GR00T horizon.

## J. Artifacts and reproducibility

The node-local manifest and analysis are:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20t/timing_sensitivity_v1_full/stage20t_timing_sensitivity_manifest.json
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20t/timing_sensitivity_v1_full/stage20t_timing_analysis.json
```

The manifest records the Stage20S model-output SHA256
`0c1ea36801cfc438775550b9f3274edae1ad60b9c6eff180ddfe7c1764034bcc`, the
Stage16 candidate SHA256
`0baaa2ac6f9f0536a98033e26a2f45927b997e6b6503464bff42ed6132106b48`, native
runtime metadata, safety seals, candidate hashes, and per-step trace paths.

## K. Safety and non-goals

No new GR00T inference or training was performed in Stage20T; A replays the
already sealed Stage20S actual-model output. No P1M, hardware, physical
command path, camera inference, ROS, ZMQ, PolicyServer, or checkpoint/source
modification was used. No production action period was selected.

## L. Git and tests

- parent Stage20S commit: `bb0d9cf59401a5418501a34141da574da04aeb56`;
- branch: `stage20t/native-wbc-timing-qualification`;
- origin: private `git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git`;
- Stage20T code and report commit: recorded by the final `git log -1` and
  stacked PR handoff;
- CPU-only test suite: `55 passed in 0.28s` on `corelab-5080-2` with the
  existing environment; no Isaac/GPU/HF dependency is required;
- node-local NPZ/JSON/log artifacts, WBC source/checkpoints/archive, and
  credentials were not staged.

## M. Next gate

The next scientific gate is:

```text
SAME_SCENE_RECEDING_HORIZON_VLA_WBC_GATE
```

It must use native-scene RGB, explicit StateV1/action timestamps, fresh
single-chunk inference after a prefix execution, and a fixed-reset control.
Stage20T does not authorize real hardware, does not promote the model-driven
visual reach, and does not turn retimed diagnostics into task evidence.

## Final integrity

The required immutable input hashes are retained in
`reports/STAGE20T_RESOURCE_GATE.md`. Final process, listener, and Git staged
artifact checks are recorded in the project handoff and PR. No Stage20T
process remains after matrix completion.

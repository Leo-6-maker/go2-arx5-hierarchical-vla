# Stage21 Final Report — Same-Scene Native Vision and Receding Horizon

Date: 2026-09-01 (Asia/Singapore)

## A. Classification

```text
PASS WITH DEVIATION
```

Native same-scene camera and reference-only hot-swap infrastructure passed.
The actual model traversed the same-scene receding execution chain in a
synchronous paused simulation. Full Stage21 promotion is withheld because
the unmodified model's raw predicted chunk profile is discontinuous at
replanning boundaries:

```text
ONE_SPECIFIC_MODEL_PROFILE_BLOCKER_REMAINS
```

This is a model-profile limitation, not a native-camera or hot-swap failure.

## B. Stage20 baseline and scope

The frozen Stage16/Stage20T native WBC baseline was not modified. Stage21
used the delivered native controller and the existing actual N1.7 checkpoint
from the Stage20S simulation adaptation. No new WBC training or checkpoint
change was made. No physical robot, hardware ROS, P1M, ZMQ, PolicyServer, or
production transport was used.

## C. Native camera and visual evidence

The native camera gate passed with a target actor, body/wrist cameras, native
WBC, and same-scene state. The actual receding run captured six observations
from one native simulator instance (`native-isaacgym-pid-2691550`). The
matched-state counterfactual audit produced finite model outputs and small
but nonzero output changes when target images were changed or occluded.

The evidence supports native RGB/state plumbing and output sensitivity only.
It does not prove visual grounding, target localization, or task success.
See `STAGE21_NATIVE_CAMERA.md` and `STAGE21_VISUAL_COUNTERFACTUAL.md`.

## D. Actual GR00T output and interface seal

The Stage10 custom N1.7 path produced finite:

```text
raw universal output:   [1,40,132]
decoded physical output: [1,40,10]
EEF:                    [40,9]
gripper:                [40,1] parallel channel
```

Runtime: Python 3.12.3, PyTorch 2.9.0+cu128, RTX 5080. The three-input
inference audit used seed `2121`, checkpoint index SHA256
`ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7`, and
GR00T source SHA `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.

The offline chain

```text
actual GR00T -> GR00TTrajectoryProvider -> TrajectoryChunkV1
-> TrajectorySafetyGateV1 -> EpisodeReferenceBindingV1
-> GR00TToWBCAdapter -> WBCPreviewV1
```

passed for all three audited inputs: 3/3 `ACCEPT`, 0 warnings, 0 rejects,
no correction. Production action period remains
`TBD_BY_ADAPTATION_DATASET`; `0.05 s` is test-only.

## E. GT-first hot-swap

The fixed seed-13 GT-first run used frozen `K=4` from the declared set
`{4,6,8,10}`. It ran 24/24 valid native steps, 5 updates, and 0 reset
events, ending at `SCHEDULE_END`. Root state, DOF state, episode length, and
arm/dog policy histories were unchanged for every update. The result is
`GT_RECEDING_HORIZON_INFRASTRUCTURE_PASS`.

The executor tests also cover fresh, aging, stale, out-of-order, invalid
timestamp, invalid prefix, and forbidden-reset cases.

## F. Actual-model receding execution

The actual-model run used
`SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON`, seed 13, target
`[0.30,0.10,0.58]`, native control dt `0.01999999955 s`, and prefix `K=4`
(`0.0799999982 s` of native simulation time). It recorded:

```text
6 actual-model chunks
24/24 valid native steps
5 reference updates
6/6 SafetyGate ACCEPT
0 reset/fall/invalid-state events
SCHEDULE_END, not native task termination
```

Actual N1.7 forward latency was mean `0.1492 s`, median `0.0633 s`, P95
`0.4507 s`, max `0.5798 s`; peak allocation was `6089.454 MiB` on RTX
5080. The approximately `29.7257 s` mean mailbox wait is offline NFS
visibility/staging latency and is not model latency or real-time readiness.

## G. Full-chunk profile and handoff metrics

Across the six complete chunks:

| Metric | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Path length (m) | 0.23177 | 0.23199 | 0.23574 | 0.23582 |
| Initial position jump (m) | 0.001170 | 0.001171 | 0.001250 | 0.001261 |
| Maximum linear speed (m/s) | 0.26627 | 0.26659 | 0.27182 | 0.27201 |
| Mean second-difference roughness (m) | 0.009839 | 0.009832 | 0.010043 | 0.010049 |

Across five raw chunk handoffs:

| Metric | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Previous prediction -> new first position (m) | 0.04509 | 0.02276 | 0.12443 | 0.14829 |
| Previous prediction -> new first rotation (rad) | 0.11623 | 0.13480 | 0.15665 | 0.15702 |
| Achieved EEF -> new first position (m) | 0.001152 | 0.001152 | 0.001210 | 0.001215 |

The new chunks are anchored near achieved state but do not continue the
previous model prediction. This is the specific model-profile blocker. No
same-scene GT desired label existed for the exact camera/state observations,
so no fabricated Stage21 pred/GT path or speed ratio is reported.

## H. Native WBC tracking and whole-body behavior

During the 24-row model diagnostic, consumed-reference tracking was:

| Metric | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| Position error (m) | 0.03636 | 0.01476 | 0.15292 | 0.15445 |
| Rotation error (rad) | 0.07721 | 0.07023 | 0.15573 | 0.15702 |

The 0.05 m / 0.20 rad joint tolerance fraction was `0.8333`. Base
displacement, path length, and yaw change were `0.06591 m`, `0.23586 m`, and
`0.19119 rad`; arm joint path length was `2.83690 rad`. The internal 9D body
plan was finite for 24/24 rows. Its presence and base motion show that native
whole-body policy outputs were active; they do not prove that GR00T learned
whole-body coordination.

No fall, reset, invalid-state, timeout, or explicit native joint-limit
termination occurred. A diagnostic all-DOF margin flag was present on the
gripper index, while the arm minimum margin was positive and no native
joint-limit termination was raised; these are not conflated.

The target-marker endpoint distance was `0.10855 m` as a descriptive value.
Task success was not evaluated.

## I. Runtime and async status

```text
GT reference hot-swap:       PASS
actual same-scene execution: EXECUTED, not qualified
async runtime:               ASYNC_RUNTIME_NOT_QUALIFIED
```

No network transport was introduced. The synchronous paused mode proves
re-observation/replanning control flow only; it does not qualify real-time
deployment.

## J. Git and tests

```text
branch: stage21/same-scene-receding-horizon
parent: 58fe006a4986386b923bcef2b7a43d6f6559bbff
origin: git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git (private)
commit: recorded after final staged-artifact audit
CPU tests: 61 passed (remote Stage10/IsaacLab-compatible Python environment)
```

The repository contains reusable contracts, executor, same-scene structures,
native runners, analysis helper, tests, reports, and ADRs only. Node-local
model/WBC artifacts and credentials remain outside Git.

## K. Final gate and next action

```text
FINAL_GATE: ONE_SPECIFIC_MODEL_PROFILE_BLOCKER_REMAINS
```

The next action is to address this one model-profile attribution/fix with a
predeclared action-generation/profile experiment. Do not start async
transport, real hardware, or task-success claims before that gate is closed.

Task-performance status remains:

```text
ROBOT_SPECIFIC_GR00T_TASK_PERFORMANCE_NOT_QUALIFIED
```

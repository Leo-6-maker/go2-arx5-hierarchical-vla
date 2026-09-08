# Stage20S Final Report — Simulation-only GR00T Adaptation and Native WBC Closed Loop

Date: 2026-09-01 (Asia/Singapore)

## A. Classification

`PASS WITH DEVIATION`

The simulation-only data → official GR00T adaptation → actual custom N1.7
inference → provider → safety gate → binding → native WBC simulation chain was
executed. The known-good benchmark orchestration completed. The actual
model-driven visual-reach rollout was accepted by the gate but ended in a
native timing `TRACKING_ABORT` after 25 samples, so the stronger end-to-end
qualification gate remains closed.

## B. Stage16 promotion

`NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`

Stage16 was audited, not rerun. Its frozen 28-trajectory baseline remains 27
`REFERENCE_COMPLETE`, one expected boundary `TRACKING_ABORT`, zero falls,
invalid resets, timeouts, or true joint-limit terminations. Its eight selected
progressive cases remain `BASE_RELOCATION_LIKE`.

## C. 40-point representation

`40-POINT REPRESENTATION ADEQUATE ON TESTED STAGE16 DISTRIBUTION`

The Stage16 matched ablation remains the authority: case-level mean-error
median/P95/max were `2.43964e-5 / 1.98620e-4 / 2.07631e-4 m` for reconstruction
position and `5.976e-9 / 8.65439e-5 / 9.98223e-5 rad` for reconstruction
rotation. The maximum paired end-to-end mean penalties were
`1.84976e-5 m` and `2.15568e-4 rad`. This is not arbitrary high-curvature
equivalence.

## D. Git

- parent: `33760f738029dda225ec91f49fb26fa2e18965db`;
- branch: `stage20s/sim-vla-wbc-closed-loop`;
- origin: private `git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git`;
- final Stage20S commit: recorded by `git log -1` after this report is committed;
- CPU tests: `51 passed in 0.42s` on `corelab-5080-2`;
- local `compileall` with `python3` and `git diff --check`: passed;
- node-local model/data/rollout artifacts were not staged;
- WBC archive/source/checkpoints, ZIPs, videos, datasets, tokens, and machine
  local credentials were not staged.

## E. Data and training

`SIM_VISUAL_REACH_V0` materialized easy 10/25/50/100 episode fixtures and an
8-episode progressive offline fixture. The progressive fixture passed the
official GR00T loader, sharded dataset, collator, and processor with 176 valid
H=40 windows and round-trip maximum absolute error `3.0950172e-08`.

The 10/25/50/100 ladder ran 50 steps each. A 10-episode 500-step overfit
sanity run reduced the logged loss from `1.0204` at step 1 to `0.1179` at step
500. This was a resource-bounded projector/state/action-encoder adaptation;
the large backbone and diffusion transformer stayed frozen. It is not a
general task-performance or full-memory recipe qualification.

## F. Actual GR00T

`PASS`

One reproducible seeded custom N1.7 inference produced finite universal
`[1,40,132]` and decoded physical `[1,40,10]` output on the RTX 5080 in
`0.462736 s`, with peak allocation about `6089.454 MiB`. The EEF `[40,9]` and
gripper `[40,1]` channels were separated. Hugging Face authentication was
`AUTHENTICATED`; no token was exposed. The fixture's constant orientation
means the zero rotation prediction error is not an orientation-generalization
claim.

## G. Provider and safety gate

`PASS`

The actual model output was converted only through
`GR00TTrajectoryProvider` to `TrajectoryChunkV1`, with explicit timestamps,
frame, TCP, SO(3), provenance, and a parallel gripper channel. The easy
seeded candidate had a `0.608 mm` initial position discontinuity and passed
`TrajectorySafetyGateV1=ACCEPT` with no correction. The provider/binding/adapter
seal produced finite native `[9,9]` WBC preview. The progressive offline
candidate also passed its seal; it was not sent to WBC.

## H. Offline interface seal

`PASS`

The actual chain

```text
actual GR00T -> TrajectoryChunkV1 -> SafetyGate -> Binding -> WBC preview
```

was sealed without WBC execution first. No raw GR00T universal tensor was
consumed by WBC code, and no trajectory was silently clipped or re-timed.

## I. Runtime orchestration and actual WBC

Known-good benchmark orchestration was `PASS`: `E1_W0_00` completed in 98/100
steps with position mean/P95 `0.0129341/0.0267713 m` and rotation
mean/P95 `0.136018/0.455563 rad`.

The actual accepted model candidate was then executed by the gated
simulation-only wrapper. The native launcher completed, but the rollout was:

```text
EXECUTED_TRACKING_ABORT
25 / 110 samples
REFERENCE_COMPLETE = false
```

The pre-reset timing error was `0.16725899 m`, just above the delivered
`0.70 * traj_L = 0.16311033 m`; lateral distance was only `0.02887013 m`, and
the termination cause was timing. There was no fall, invalid state, timeout,
or true joint-limit termination. The four-error decomposition is recorded in
`STAGE20S_ERROR_DECOMPOSITION.md`.

## J. Whole-body behavior

The partial model rollout produced meaningful simulated base motion: base
displacement `0.06508 m`, path length `0.23732 m`, and yaw change `0.30064
rad`. All nine internal body/gait plan channels were finite and varied over
the logged prefix. This supports that native WBC execution and its internal
body-plan channel were active; it does not establish learned visual
whole-body coordination.

## K. Progressive/continuous-base decision

`NO_NEW_VALID_NATIVE_CONTINUOUS_BASE_COHORT_QUALIFIED`

The eight Stage20S progressive cases are cross-simulator visual fixtures with
a static recorded base. They are retained as offline stress data. No native
continuous-base cohort was fabricated, and the progressive cases were not
promoted to `CONTINUOUS_BASE_COORDINATION`.

## L. Architecture and task status

`SIMULATION_ONLY_VLA_WBC_INTERFACE_RUNTIME_READY_WITH_NATIVE_TIMING_LIMITATION`

The thin EEF-only interface, provider, gate, binding, adapter, and native WBC
orchestration are reusable in simulation. The model-driven easy rollout did
not meet the stronger end-to-end qualification gate, and the camera path is a
procedural cross-simulator fixture rather than a same-scene native camera.

`SIMULATION_ONLY_ROBOT_SPECIFIC_ADAPTATION_PERFORMED; REAL_ROBOT_ADAPTATION_NOT_PERFORMED`

No claim is made about physical robot readiness or general manipulation.

## M. Next action

`FIX_ONE_SPECIFIC_NATIVE_WBC_TIMING_ROLLOUT_LIMITATION_IN_SIMULATION`

The next experiment should isolate the delivered timing-abort behavior with a
predeclared simulation timing sensitivity protocol. It must preserve the
native controller semantics and keep progressive model-driven WBC execution
behind a qualified easy-task gate.

## Final integrity

Stage20S did not use P1M, hardware, real data, ROS, ZMQ, PolicyServer, or
physical command paths. No Stage20S process remained after the runs. The
private source/checkpoint/data boundaries remain external to Git.

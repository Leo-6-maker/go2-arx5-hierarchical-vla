# Stage17 Final Report

Date: 2026-09-01 (Asia/Singapore)

## A. Classification

`PASS WITH DEVIATION`

Stage17 closed the actual custom GR00T runtime, canonical provider, safety
gate, chunk lifecycle, offline interface boundary, and known-good native-WBC
orchestration. The deviation is intentional: the one unadapted actual-model
trajectory was rejected by the non-corrective gate under test-only speed
thresholds, so it did not reach binding or WBC. No task performance is claimed.

## B. Stage16 promotion

`NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`

The promotion review did not rerun the 28-case suite. It audited the existing
28 native results: 27 `REFERENCE_COMPLETE`, one expected boundary
`TRACKING_ABORT`, zero fall/invalid-reset/timeout/true-joint-limit events, and
eight progressive cases classified `BASE_RELOCATION_LIKE`. The abort is an
`EXPECTED_BOUNDARY_LIMITATION`, with native timing error as the precise cause
and generator-support boundary as secondary context.

## C. 40-point representation

`40-POINT REPRESENTATION ADEQUATE ON TESTED STAGE16 DISTRIBUTION`

Case-level median/P95/max of per-case means:

| Quantity | Median | P95 | Max |
|---|---:|---:|---:|
| reconstruction position mean (m) | `2.43964e-5` | `1.98620e-4` | `2.07631e-4` |
| reconstruction rotation mean (rad) | `5.976e-9` | `8.65439e-5` | `9.98223e-5` |
| controller position delta mean (m) | `-5.88891e-6` | `2.67372e-6` | `4.11745e-6` |
| controller rotation delta mean (rad) | `2.96751e-5` | `1.88351e-4` | `2.15691e-4` |
| end-to-end position delta mean (m) | `-2.92425e-6` | `1.49714e-5` | `1.84976e-5` |
| end-to-end rotation delta mean (rad) | `2.96913e-5` | `1.88228e-4` | `2.15568e-4` |

The largest per-case reconstruction p95 was `5.54006e-4 m` and
`1.88898e-4 rad`; largest absolute errors were `8.52269e-4 m` and
`2.01291e-4 rad`. This is not arbitrary high-curvature equivalence.

## D. Git

- branch: `stage17/groot-runtime-integration`
- source commit before Stage17: `093e92791ed88894cf5a14d103f408f8f3ff4022`
- Stage17 implementation commit: `13858cb` (`feat: seal Stage17 GR00T to WBC interface`)
- final branch tip: the subsequent documentation/audit commit containing this
  recorded handoff; the exact tip is also printed by `git log -1`
- private origin: `git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git`
- test count: `33 passed in 4.84s` in the Stage10 Python environment with
  `CUDA_VISIBLE_DEVICES=""`; local `pytest` was unavailable
- compileall: passed
- `git diff --check`: passed
- node-local actual model NPZ, logs, private checkpoints, WBC source/archive,
  and credentials: not staged

The existing canonical `main`, Stage15, and Stage16 branches were pushed
without rewriting history before Stage17 work began.

## E. HF authentication

`AUTHENTICATED`

On `corelab-5080-2`, Stage10 `hf auth whoami` returned `user: leoyuer` and
the gated Cosmos processor loaded. The token was never requested, printed,
copied, inspected, or stored.

## F. Actual GR00T

Full custom N1.7 inference: `PASS`.

- input: body/wrist Stage10 synthetic first frames, state, and language
- raw output: `[1,40,132]`
- decoded physical output: `[1,40,10]`
- latency: `27.130034 s`; model load `200.957005 s`
- device: CPU; peak GPU allocation/reservation `0 / 0 MiB`
- GR00T source SHA: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`
- raw hash: `c4ecb7459fa5ccfad2be6054f38e64fdf21b3b044c26fdbc2c3d96ebe38f49f6`
- decoded hash: `a361e4fea21c5ad817167e10226fe46a936cdd630b4f509baf2c9301b23865dc`

## G. GR00T provider

`PASS`

`GR00TTrajectoryProvider` implements `TrajectoryProvider`, converts decoded
physical `[H,10]`/`[B,H,10]`, emits `TrajectoryChunkV1`, and separates the
gripper into `GripperCommandV1`. It requires explicit point timestamps or an
explicitly labelled test period and does not expose raw universal tensors to
WBC.

## H. Trajectory safety gate

`PASS`

The actual base-model trajectory was `REJECT`:

```text
LINEAR_SPEED_REJECT: 2.1140129 m/s > 2.0 m/s (TEST_ONLY)
ANGULAR_SPEED_WARN:  2.2792987 rad/s > 2.0 rad/s (TEST_ONLY)
```

Its initial position and rotation discontinuities were both zero and its SO(3)
values were finite. The trajectory was not modified.

## I. Offline end-to-end interface

`SKIPPED AFTER REQUIRED GATE REJECT`

The actual path passed model-output decoding and provider validation, then
stopped at the gate as required. Binding and WBC preview were not invoked for
the rejected actual output. This is a safe gate outcome, not an integration
failure, and `wbc_executed_from_actual_model` is false.

## J. Runtime orchestration

`PASS`

Using the known-good Stage16 `E1_W0_00` trajectory, the provider → chunk →
gate → identity binding → adapter → `[9,9]` WBC preview → native WBC chain
passed on `corelab-4080s`. Native WBC returned `REFERENCE_COMPLETE` after 98
logged steps, with position mean/RMSE/p95/max `.012934/.014983/.026771/.038226
m` and rotation mean/RMSE/p95/max `.136018/.191413/.455563/.565715 rad`.
The abstract chunk lifecycle produced FRESH, AGING, STALE, and OUT_OF_ORDER
events as specified. Actual GR00T was not used in this run.

## K. Architecture status

`VLA_WBC_INTERFACE_RUNTIME_READY`

The transport-independent provider, gate, binding, adapter, lifecycle, and
known-good native-WBC orchestration are sealed. Production action frequency,
physical frame/TCP calibration, and robot safety thresholds remain adaptation
or deployment contracts; Stage17 does not invent them.

## L. Task-performance status

`ROBOT_SPECIFIC_GR00T_ADAPTATION_NOT_YET_PERFORMED`

Correct tensor shapes, finite values, authentication, and runtime latency are
not task evidence. No actual model task success or closed-loop model-to-WBC
result was evaluated.

## M. Continuous-base cohort

`NOT_REQUIRED_FOR_INITIAL_VLA_WBC_INTEGRATION`

Stage16's bounded search found no valid native continuous-base cohort. The
initial benchmark remains interpretable without fabricating one.

## N. Next action

`START_ROBOT_SPECIFIC GR00T DATASET PREPARATION`

The next stage should define synchronized RGB/language/proprioception,
absolute EEF and gripper labels, frame/TCP calibration, timestamps, and
quality/lifecycle filters before training or finetuning.

## Final integrity

The immutable WBC archive, native URDF, native checkpoints, native config, and
P1M canonical asset retained their required hashes. No P1M, hardware, ROS,
ZMQ, PolicyServer, camera inference, training, or finetuning was run. No
Stage17 process remained at handoff; no project-owned listener was left on
ports 5550/5556/5557. Shared Isaac-GR00T pre-existing modifications were not
touched. Final Git status, staged diff, and private-artifact scan are recorded
with the final commit.

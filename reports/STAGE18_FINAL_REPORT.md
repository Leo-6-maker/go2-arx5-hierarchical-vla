# Stage18 Final Report — Robot-Specific Data Readiness

Date: 2026-09-01 (Asia/Singapore)

## A. Classification

`PASS WITH DEVIATION`

The raw/derived contracts, synchronization metrics, non-corrective quality
gate, D3 fixture conversion, logical LeRobot record, and CPU tests are in
place. Real-pilot readiness is withheld for one specific pipeline issue: D3
camera metadata declares 20 Hz while source timestamps measure 25 Hz.

## B. Stage16 promotion

`NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`

Stage16 is frozen. Its 28 native trajectories had 27 reference completions,
one expected boundary `TRACKING_ABORT`, zero falls/invalid resets/timeouts/
true joint-limit terminations, and eight progressive cases classified
`BASE_RELOCATION_LIKE`. No Stage16 benchmark was rerun.

## C. 40-point representation

`40-POINT REPRESENTATION ADEQUATE ON TESTED STAGE16 DISTRIBUTION`

The reviewed case-level median/P95/max values are:

| Quantity | Median | P95 | Max |
|---|---:|---:|---:|
| reconstruction position mean (m) | `2.43964e-5` | `1.98620e-4` | `2.07631e-4` |
| reconstruction rotation mean (rad) | `5.976e-9` | `8.65439e-5` | `9.98223e-5` |
| controller position delta mean (m) | `-5.88891e-6` | `2.67372e-6` | `4.11745e-6` |
| controller rotation delta mean (rad) | `2.96751e-5` | `1.88351e-4` | `2.15691e-4` |
| end-to-end position delta mean (m) | `-2.92425e-6` | `1.49714e-5` | `1.84976e-5` |
| end-to-end rotation delta mean (rad) | `2.96913e-5` | `1.88228e-4` | `2.15568e-4` |

This claim is limited to the tested Stage16 distribution; it is not arbitrary
high-curvature equivalence.

## D. Git

- Branch: `stage18/robot-data-readiness`.
- Base: Stage15–17 promotion merge `a016dcba...`.
- Origin: private `Leo-6-maker/go2-arx5-hierarchical-vla`.
- CPU tests: `44 passed in 0.41s` on `corelab-5080-2` with the repository test
  suite and no GPU workload.
- Reusable code/docs/tests only are tracked. D3 data, WBC archive/source,
  checkpoints, images, videos, model outputs, and credentials remain outside
  Git.

## E. HF auth

`AUTHENTICATED`

With the complete Stage10 `env.sh` plus venv, `hf auth whoami` returned
`leoyuer`; the gated Cosmos resource required by the official processor was
accessible. No token was requested, printed, copied, or inspected.

## F. Actual GR00T

`SKIPPED` by Stage18 scope. No actual-model forward or unadapted model→WBC
execution was performed. The existing Stage17 actual-model evidence remains
separate and unchanged.

## G. GR00T provider

`PASS` by the existing Stage17 `GR00TTrajectoryProvider` contract; Stage18
derived data emits only the physical 10D/trajectory contract and never exposes
universal model tensors to WBC code.

The official Stage10 `Gr00tN1d7Processor` data smoke also passed on the D3
target-action stats: processed state `[1,132]`, universal action `[1,40,132]`,
decoded physical `[1,40,10]`, finite values, and round-trip max abs
`3.9073288e-06`. This is interface evidence only, not model/task evidence.

## H. Trajectory safety gate

`SKIPPED` for an actual base-model trajectory. Stage18 only ran the episode
data quality gate. Existing Stage17 safety-gate semantics are unchanged.

## I. Offline end-to-end interface

`SKIPPED` for actual GR00T by scope. The Stage17 known-good benchmark
orchestration and actual-model stop boundary remain frozen. Stage18 verifies
the data side up to the official processor boundary without WBC execution.

## J. Runtime orchestration

`PASS` by frozen Stage17 known-good benchmark evidence; no new native WBC run
was needed for data readiness.

## K. Architecture status

`VLA_WBC_INTERFACE_RUNTIME_READY`

The software thin waist is ready; physical calibration, production timing, and
robot safety thresholds remain future deployment/data contracts.

## L. Task-performance status

`ROBOT_SPECIFIC_GR00T_ADAPTATION_NOT_YET_PERFORMED`

Neither finite tensors nor processor compatibility is a robot task result.

## M. Stage18 gate

`ONE_SPECIFIC_DATA_PIPELINE_BLOCKER_REMAINS`

Specific blocker: D3 body/wrist camera metadata says 20 Hz, but preserved raw
timestamps are 25 Hz. The synchronizer surfaces the discrepancy and does not
repair it. The D3 bundle is also explicitly simulation-only, so it cannot be
used as real calibration.

## N. Next action

`START_ROBOT_SPECIFIC GR00T DATASET PREPARATION`

The next preparation task is to reconcile source-rate metadata and define the
measured real calibration/clock bundle before requesting a real 5–10 episode
pilot. No real collection is authorized by this report.

## Final integrity

Stage18 did not run hardware, ROS, ZMQ, PolicyServer, training, finetuning,
P1M, H200, actual GR00T inference, or native WBC execution. The node-local
fixture/processor outputs are private and not staged. Final process and Git
audits are performed immediately before handoff.

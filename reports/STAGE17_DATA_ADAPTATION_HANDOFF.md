# Stage17 Data-Adaptation Handoff

## Status

`ROBOT_SPECIFIC_GR00T_ADAPTATION_NOT_YET_PERFORMED`

Stage17 closes runtime and interface contracts only. The actual N1.7 output
was finite but rejected by an explicit test-only speed gate and was never sent
to WBC. No task-success claim is made.

## Future training example

The proposed V1 example is body RGB, wrist RGB where available, language, and
compact proprioception at time `t`, paired with a future absolute XYZ + SO(3)
EEF trajectory and a parallel gripper target. Full raw whole-body state,
achieved EEF, base state, arm joints, timestamps, calibration, and lifecycle
events remain in the recording.

The exact action period is not assumed. It must be supplied by the adaptation
dataset or explicitly labelled as a test value. See
`docs/contracts/GROOT_ADAPTATION_DATA_V1.md`.

## Source roles

- D1: representation/pre-adaptation analysis.
- D2/D3: simulation and pipeline validation.
- E1/E2: native WBC benchmark only.
- WBC random generator: controller support analysis.
- future real Go2+ARX5 demonstrations: robot-specific adaptation source.

Benchmark trajectories are not automatically training examples. They lack the
demonstration semantics needed to claim a learned task policy.

## Recommended next action

`START_ROBOT_SPECIFIC_GR00T_DATASET_PREPARATION`

The first data-preparation gate should define calibration/frame/TCP labels,
clock alignment, target-vs-achieved semantics, gripper conventions, and
quality filters before any training or finetuning is authorized.

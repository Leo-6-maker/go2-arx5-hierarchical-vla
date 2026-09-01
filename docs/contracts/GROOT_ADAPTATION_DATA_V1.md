# GR00T Adaptation Data Contract V1

## Status

Prepared for a future robot-specific adaptation stage. No training or
finetuning is performed by Stage17.

## Target example

Each training example aligns an observation at time `t` with a future absolute
EEF trajectory and a parallel gripper command:

```text
INPUT
  body RGB                         (optional only when unavailable)
  wrist RGB                       (where available)
  language instruction
  compact proprioception at t

OUTPUT
  future absolute EEF trajectory  [H,9] = XYZ + SO(3)/rot6d
  future gripper command           [H,1] or an explicitly timestamped command
```

The production action timestamps are part of the example and must not be
reconstructed from an assumed 20 Hz rate. If a dataset supplies a fixed
sample period, the period and its provenance are recorded. Until adaptation
data fixes the rate, the production value is `TBD_BY_ADAPTATION_DATASET`.

## Retained raw data

The raw recording retains richer state for audit and relabeling:

- body and wrist images with capture timestamps;
- language and episode metadata;
- raw base pose/velocity and arm/dog joint state;
- commanded and achieved EEF pose;
- gripper state/command;
- controller preview and lifecycle/termination events;
- calibration/frame/TCP identifiers and clock-domain metadata.

The model-facing compact proprioception is derived from this raw record and is
versioned separately. Achieved state must not be substituted for the target
label without recording that operation.

## Interface alignment

The target label is aligned to the Stage17 external interface:

```text
EEF target + gripper
  -> TrajectoryChunkV1 + GripperCommandV1
  -> TrajectorySafetyGateV1
  -> EpisodeReferenceBindingV1
  -> WBCPreviewV1
```

The gripper channel is parallel and is not a WBC input. Frame and TCP
semantics are mandatory. The adaptation dataset must provide the calibration
needed to map its EEF target into the native WBC frame; Stage17's synthetic
identity binding is not a physical calibration.

## Data-source mapping

| Source | Permitted contribution |
|---|---|
| D1 public manipulation | VLA representation and pre-adaptation analysis |
| D2/D3 | simulation and data-pipeline validation |
| E1/E2 | native WBC benchmark only |
| WBC random generator | controller training-distribution/support analysis |
| future real Go2+ARX5 demonstrations | primary robot-specific VLA adaptation data |

E1/E2 and WBC generator trajectories must not be mixed blindly into VLA
training. They are useful for controller qualification and interface tests,
not evidence that the base GR00T checkpoint learned the robot task.

## Preferred future collection flow

```text
human / task-space teleoperation
  -> desired EEF trajectory + gripper
  -> General WBC
  -> Go2+ARX5
  -> record synchronized RGB, language, raw proprioception,
     EEF target, EEF achieved, gripper, base, arm, and timestamps
```

Collection must first establish timestamp synchronization, frame/TCP
calibration, reset/lifecycle recording, and safe gate thresholds. No real
robot command path is created by this contract.

# Stage18 Data Adaptation Handoff

## Source roles

```text
D1 public manipulation data       -> VLA representation/pre-adaptation analysis
D2/D3 simulation                  -> data-pipeline fixtures only
E1/E2 native trajectories         -> WBC benchmark only
WBC random generator              -> controller support analysis
future Go2+ARX5 demonstrations    -> primary robot-specific adaptation data
```

WBC benchmark trajectories must not be mixed into VLA training as if they
were human/task demonstrations.

## Prepared, not executed, pilot flow

```text
P0: calibration/reach checks                 about 5 episodes
P1: simple grasp/place                       about 5–10 episodes
P2: progressive whole-body task-space tests  about 5–10 episodes
```

Before any real collection, resolve the D3 rate metadata issue, measure the
physical calibration bundle, define clock domains, and pass the real-pilot
quality gate. No real episode was collected in Stage18.

Preferred future capture path:

```text
human/task-space teleoperation
    -> desired EEF trajectory + gripper
    -> General WBC
    -> Go2+ARX5
    -> RGB, language, raw state, desired/achieved EEF, gripper,
       base/arm state, WBC events, and timestamps
```

The primary state/action ABI remains the Stage18 10D state and future desired
EEF+gripper label. Raw data retains richer whole-body state for audit.

## Stage18 gate

`ONE_SPECIFIC_DATA_PIPELINE_BLOCKER_REMAINS`: reconcile D3 declared camera FPS
with its effective source timestamp rate before treating this fixture as a
pilot-ready timing contract.

# PI0179 reach-first visualization boundary

The first visual milestone is a native-simulation replay in which the Go2+ARX5
EEF moves progressively toward a visible red target through the already
integrated GR00T -> provider/SafetyGate -> WBC path.  The replay recorder is
observational: it consumes source frames, state, predicted H40 trajectories,
SafetyGate state, WBC tracking telemetry, lifecycle state, and timestamps.  It
does not change physics, target placement, controller commands, or trajectory.

Grasp/contact remains a later design-only stage because the current WBC path
does not actuate the gripper.  The minimum future grasp demo is:

```text
approach red cube
-> enter grasp region
-> parallel gripper close
-> contact/attachment
-> lift
-> object-height success
```

The future evaluation must keep EEF control and gripper actuation as separate
channels.  Contact/attachment success must not be inferred from EEF reach
metrics, and adding grasp physics requires a later PI gate.

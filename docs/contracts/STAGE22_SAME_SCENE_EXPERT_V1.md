# Stage22 Same-Scene Expert Snapshot Contract V1

Each record is captured from one native IsaacGym WBC simulator instance at a
fixed expert replanning point. It contains native body/wrist RGB, the current
`StateV1`, and a deterministic expert future label generated from the current
achieved EEF pose to the same-scene red target.

The target position is retained only in provenance and label-generation
metadata. It is not included in the model-facing image/state/language input.
The expert is position-only smoothstep-to-hold with the current orientation
held constant; it does not model contact, grasp, or gripper dynamics.

The offline LeRobot fixture expands each snapshot into a one-window episode by
repeating the observation frame/state for the 40-point label horizon. This is
explicitly marked `snapshot_expanded_horizon` and is not evidence of a
continuous 40-frame native camera recording. It is used to exercise the
official loader/processor and to train on-policy replanning snapshots without
inventing future RGB.

The training ABI remains:

```text
StateV1 = EEF XYZ[3] + EEF rot6d[6] + gripper[1]
label = future desired EEF[40,9] + future desired gripper[40,1]
```

# TeleopCommandV1

`TeleopCommandV1` is the desired task-space label contract for future
Go2+ARX5 pilots. It is not a robot command transport and it has no actuation
method.

## Fields

```text
command_id
timestamp                    source clock domain
eef_9d                       XYZ 3 + GR00T_FIRST_TWO_ROWS rot6d 6
gripper                     one normalized command value
frame                       explicit pose frame
tcp_semantic_id             explicit TCP identity
source                      teleop/replay provenance
```

The timestamp is retained as supplied. A production action period is not
assumed. The command is recorded as the future desired label; achieved EEF and
measured gripper values remain separate raw state fields.

## Separation boundary

```text
TeleopCommandV1
       |
       +--> desired label in GR00TTrainingExampleV1
       |
       +--> TrajectoryChunkV1 / parallel GripperCommandV1 for offline checks

achieved EEF / measured gripper
       --> raw audit stream only
```

Stage19 only qualifies construction and replay of this contract. It does not
open a teleoperation link or send a command to Go2/ARX5.

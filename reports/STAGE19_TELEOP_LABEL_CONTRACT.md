# Stage19 Teleop and Label Contract

## Frozen ABI

```text
StateV1
  EEF XYZ       3
  EEF rot6d     6  (GR00T_FIRST_TWO_ROWS)
  gripper       1
  ----------------
                10D

future label
  desired EEF  [H,9]
  desired grip [H,1]
  timestamps   [H]
```

`TeleopCommandV1` is a desired task-space label with an explicit timestamp,
frame, TCP identity, source, EEF 9D value, and gripper value. It is not a
hardware command and is not a network message. Its gripper value travels in a
parallel `GripperCommandV1` channel and is not fed to WBC.

Raw episodes retain Go2/ARX5/base/IMU/WBC state and achieved EEF/measured
gripper streams. Achieved values are audit signals; they are never silently
substituted for desired training labels.

## Qualification

CPU tests verify 9D/1D construction, finite values, timestamp order, and
desired/achieved separation. The D3 replay confirms the same separation over
121 samples. The implementation lives in
`src/go2arx5_vla/data/teleop.py` and `src/go2arx5_vla/data/recorder.py`.

Production teleoperation, transport, and physical collection are outside
Stage19.

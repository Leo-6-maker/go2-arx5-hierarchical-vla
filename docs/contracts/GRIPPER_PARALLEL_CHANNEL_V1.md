# Parallel gripper channel V1

The delivered WBC does not contain a gripper actuator. The current boundary
is therefore:

```text
TrajectoryChunkV1
       |-- EEF XYZ + SO(3) --> WBCPreviewV1 --> delivered WBC
       `-- gripper[N,1] ----> GripperCommandV1 (future parallel path)
```

This repository defines timestamp, normalized command, raw convention, and
stale status only. It does not execute hardware or simulate a gripper in the
Stage14 WBC benchmark.

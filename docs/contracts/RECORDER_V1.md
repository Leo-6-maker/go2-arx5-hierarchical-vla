# RecorderV1

`RecorderV1` is a transport-independent, read-only recording abstraction.
The Stage19 implementation accepts asynchronous state, camera, desired
command, lifecycle, and safety streams and enforces finite values and strict
per-stream timestamp order.

## Recorded streams

- native/rich state: Go2, ARX5, achieved EEF, gripper measurement, base pose;
- body and wrist images with their own source timestamps;
- `TeleopCommandV1` desired EEF and gripper labels;
- lifecycle and safety events;
- calibration bundle and frame-graph identifiers.

The raw record retains the native source streams. Synchronization and
LeRobot/GR00T materialization are derived views and never replace the raw
record.

## Safety boundary

`RecorderV1` has no hardware backend, socket, ROS publisher, WBC execution
method, or command-send method. Stage19 qualifies only `replay_raw_episode`
against a D3 fixture. A future real recorder must be separately reviewed for
clock domains, disk durability, backpressure, and hardware authorization.

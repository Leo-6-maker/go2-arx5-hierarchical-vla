# CalibrationBundleV1 — Real-Robot Measurement Protocol

This document defines what must be measured before a real Go2+ARX5 episode is
accepted. It does not provide invented transforms or safety limits.

## Required measured components

1. **Frame graph:** freeze the names and direction of the Go2 body/world,
   ARX5 mount, ARX5 flange, TCP, body camera, and wrist camera frames.
2. **Mount and TCP transforms:** measure `T_Go2_ARX5` and `T_flange_TCP` using
   a documented fiducial/robot-calibration procedure and store the method,
   samples, repeatability, operator, and measurement time.
3. **Camera calibration:** measure intrinsics and camera-to-frame extrinsics
   for body and wrist cameras. Record resolution, distortion model, target
   type, calibration images, and reprojection diagnostics.
4. **ARX5 joint zero:** record the firmware/mechanical zero version and the
   measurement or homing procedure. Do not infer it from the simulation URDF.
5. **Gripper mapping:** measure the raw sensor/command convention and its
   normalized open/close mapping, including saturation and invalid readings.
6. **Clock synchronization:** identify every clock domain and measure offset,
   drift, and uncertainty using a shared event or timestampable sync signal.
   Record whether timestamps are device, host, or monotonic time.

## Bundle states

```text
REAL_TBD                 required physical value is absent
MEASUREMENT_IN_PROGRESS  partial bundle cannot qualify episodes
MEASURED_PENDING_REVIEW  all fields measured, reviewer has not accepted them
REAL_VALIDATED            reviewer accepted method and diagnostics
INVALIDATED               hardware, frame graph, clock, or calibration changed
```

`CalibrationBundleV1.real_robot_calibration` may be `true` only for
`REAL_VALIDATED`. The bundle ID and version are copied into every raw episode
and every derived dataset record. A simulation bundle must remain
`SIMULATION_ONLY`.

## P0 acceptance procedure

P0 calibration episodes are the first real-pilot gate. They must verify the
frame/TCP transform, camera image geometry, clock relationships, desired vs
achieved EEF semantics, and gripper mapping before any P1 manipulation data
is retained. Numeric acceptance tolerances are deployment-owner fields and
remain `REAL_ROBOT_TBD` until the actual robot, sensors, and calibration method
are selected.

Stage19 defines this protocol and mock/replay qualification only. No physical
measurement or actuation was performed.

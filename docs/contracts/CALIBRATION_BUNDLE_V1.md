# CalibrationBundleV1

`CalibrationBundleV1` versions the frame, camera, joint-zero, gripper, and
clock metadata used to interpret an episode.

## Fields

- `T_go2_arx5` and `T_flange_tcp` when physically measured;
- body and wrist camera intrinsics/extrinsics;
- ARX5 joint-zero version;
- raw-to-normalized gripper mapping;
- clock synchronization calibration/version;
- bundle ID, version, status, and provenance.

Unknown physical values are represented as absent/`REAL_TBD`; they are never
inferred from a simulation transform. A D3 bundle is explicitly
`SIMULATION_ONLY` with `real_robot_calibration=false` and is usable only as a
pipeline fixture. A real-pilot gate must reject such a bundle.

# Stage18 Calibration Audit

## D3 fixture status

The six selected D3 episodes use:

```text
bundle_id: D3_SIMULATION_CALIBRATION_BUNDLE_V1
version: D3_V0
status: SIMULATION_ONLY
mount: SIM_REFERENCE_UMI_MOUNT
tcp: SIM_PROVISIONAL_TCP
pose frame: sim_world
real_robot_calibration: false
```

Body/wrist intrinsics and simulation extrinsic metadata are retained as
provenance. `T_Go2_ARX5`, `T_flange_TCP`, physical ARX5 joint-zero calibration,
real gripper conversion, and physical clock-sync calibration remain unknown or
`REAL_TBD`.

## Gate consequence

`SIMULATION_ONLY` is a warning for the D3 pipeline fixture and becomes a
rejection when `simulation_fixture_allowed=false`. No simulation transform is
promoted as real calibration. A real 5–10 episode pilot therefore requires a
new measured `CalibrationBundleV1` and a versioned calibration QA record.

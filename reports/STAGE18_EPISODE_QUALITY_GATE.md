# Stage18 EpisodeQualityGateV1

`EpisodeQualityGateV1` is a non-corrective classifier:

```text
ACCEPT | WARN | REJECT
```

It checks calibration identity/status, required body/wrist streams, finite
state/target/gripper values, native/camera timestamp monotonicity, SO(3)
validity, target step/speed metrics, synchronization skew/gaps, image path
presence when requested, lifecycle records, and safety-event visibility.

It never clips, smooths, slows, retimes, interpolates rot6d linearly, or
silently changes labels. Every configured threshold is labelled as
`CONTROLLER_CONTRACT`, `GENERATOR_SUPPORT`, `BENCHMARK_DERIVED`, `TEST_ONLY`,
or `REAL_ROBOT_TBD`.

## D3 result

All six fixture episodes were accepted with warnings:

- `SIMULATION_CALIBRATION_NOT_REAL` (`REAL_ROBOT_TBD`);
- `D3_CAMERA_FPS_METADATA_MISMATCH` (`BENCHMARK_DERIVED`).

The same gate configured for a real-robot collection
(`simulation_fixture_allowed=false`) rejects the D3 bundle. No warning was
hidden and no episode was corrected.

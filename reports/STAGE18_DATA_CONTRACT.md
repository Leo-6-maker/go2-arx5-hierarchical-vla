# Stage18 Data Contract Audit

## Frozen robot-facing ABI

```text
StateV1 = current EEF XYZ 3
         + current EEF GR00T_FIRST_TWO_ROWS rot6d 6
         + current gripper 1
         = 10D

label = future desired absolute EEF XYZ + rot6d rows [H,9]
        + future desired gripper [H,1]
        + explicit point_timestamps [H]
```

The full raw episode keeps Go2/ARX5/base/IMU/contact/WBC fields when present,
plus desired and achieved EEF streams separately. The achieved stream is audit
data and is never silently substituted for the desired label.

## Implemented contracts

| Contract | Implementation | Non-negotiable behavior |
|---|---|---|
| `RawEpisodeV1` | `go2arx5_vla.data.readiness` | native-rate arrays copied/read-only; target/achieved separate |
| `CalibrationBundleV1` | same | simulation and real calibration status explicit; unknown physical values remain `REAL_TBD` |
| `SynchronizerV1` | same | source timestamps/indices/skew/gaps retained; no default 20 Hz |
| `EpisodeQualityGateV1` | `go2arx5_vla.data.quality` | `ACCEPT/WARN/REJECT`, no clipping/retiming/repair |
| `GR00TTrainingExampleV1` | same | 10D state and desired future EEF+gripper labels |
| LeRobot bridge | `go2arx5_vla.data.processor` | dependency-light logical record; official import is lazy |

The D3 fixture is a simulated pipeline proxy only. It is not training approval,
robot calibration, WBC closed-loop evidence, or task-performance evidence.

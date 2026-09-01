# Stage20S Dataset and Manifest Qualification

## SIM_VISUAL_REACH_V0

The first task family is a deterministic simulation fixture:

```text
RGB target (red circle) + language + current StateV1
    -> future desired EEF trajectory + gripper
```

The instruction is `move the end effector to the red target`. The model-facing
ABI is frozen at:

```text
StateV1 = EEF XYZ 3 + EEF rot6d 6 + gripper 1 = 10D
label   = future desired EEF [H,9] + future gripper [H,1]
```

Raw fixture records retain achieved EEF, base pose, camera timestamps, frame
graph, lifecycle events, calibration provenance, and the desired/achieved
separation. The desired stream is the training label; achieved EEF is audit
only.

## Materialization

| Fixture | Episodes | Rows | Valid H=40 windows | Official loader/processor |
|---|---:|---:|---:|---|
| easy ladder, 10 | 10 | 610 | 220 | used for adaptation |
| easy ladder, 25 | 25 | 1525 | 550 | used for adaptation |
| easy ladder, 50 | 50 | 3050 | 1100 | used for adaptation |
| easy ladder, 100 | 100 | 6100 | 2200 | used for adaptation |
| progressive offline fixture | 8 | 488 | 176 | PASS |

The progressive fixture was passed through the official GR00T
`LeRobotEpisodeLoader`, `ShardedSingleStepDataset`, collation, and processor
path. It produced physical `[40,10]`, universal `[1,40,132]`, and decoded
`[1,40,10]` actions with finite values; round-trip maximum absolute error was
`3.0950172e-08`.

The standalone `lerobot` package was not required by the official Stage10
GR00T v2 loader path. The materializer nevertheless writes the actual v2-style
parquet/video layout and explicit sidecar manifests, including action
timestamps, calibration bundles, episode provenance, and video provenance.

## Time semantics

The fixture uses `0.05 s` only as
`TEST_ONLY_SIM_VISUAL_REACH_V0`. It is not a production Go2+ARX5 action-rate
decision. Source timestamps remain the authority; no timestamp correction or
silent retiming is performed.

## Camera boundary

The fixture camera is a dependency-free procedural render and is labelled
`NATIVE_WBC_CROSS_SIM_REPLAY`. The native WBC runtime used for the closed-loop
test has no integrated Stage20S camera path, so this is not a same-scene native
camera qualification.

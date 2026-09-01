# Stage19 Recorder Replay Qualification

## Result

`PASS`

`RecorderV1` was exercised through `replay_raw_episode` on
`d3_A_d2_ep_00_V0`. The replay copied source streams into a read-only recorder
envelope without mutating the source or opening a robot link.

| Stream | Expected | Recorded |
|---|---:|---:|
| native state samples | 121 | 121 |
| desired `TeleopCommandV1` labels | 121 | 121 |
| body camera samples | 61 | 61 |
| wrist camera samples | 61 | 61 |
| lifecycle events | 2 | 2 |
| safety events | 0 | 0 |

The recorded state keeps achieved EEF/gripper values separate from desired
commands. Per-stream timestamps are strictly increasing and all copied numeric
values are finite. The recorder declares
`NONE_READ_ONLY_RECORDING_ONLY` as its backend and has no send/actuation API.

Machine-readable output:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/recorder_replay/stage19_recorder_replay.json
```

Output SHA256:
`3157d3951e296763aa4a9238575fa533ed5d561c29691b2a19d5deca5ff7ca31`.

This is replay/mock qualification only. A real recorder still needs separate
authorization and review for device clocks, storage durability, backpressure,
and lifecycle behavior.

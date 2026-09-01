# Stage22 Same-Scene Expert Oracle

## Result

```text
EXPERT_SAME_SCENE_RECEDING_HORIZON_PASS
```

Six independent native IsaacGym simulator instances were run on
`corelab-4080s`, one per target, with fixed reset seed `13`, native WBC, native
body/wrist cameras, synchronous paused reference replacement, and frozen
prefix `K=4`. The exact collection is recorded in the node-local
`expert_collection_summary.json` and per-episode manifests.

| Quantity | Result |
|---|---:|
| Episodes / targets | 6 / 6 |
| Native control steps | 144 / 144 valid |
| Replanning updates | 30 |
| Native snapshots | 36 |
| Episode status | 6/6 pass |
| Termination | 6 `SCHEDULE_END` |
| Reset/fall/invalid events | 0 |
| Target supplied to model input | no |

Each snapshot contains the exact native RGB observation and current `StateV1`
at the replanning point. Its future desired EEF label is generated from the
same observed EEF pose to the same-scene target using a deterministic
smoothstep position path and constant current orientation. The target actor
is retained in provenance and labels but is excluded from the model-facing
record.

## Label integrity

Across all 36 labels:

```text
initial position discontinuity: mean=0, median=0, p95=0, max=0 m
initial rotation discontinuity: mean=5.85e-10, median=0, p95=0, max=2.11e-8 rad
label path length: mean=0.14597, median=0.13362, p95=0.23845, max=0.26033 m
label max speed: mean=0.11226, median=0.10276, p95=0.18338, max=0.20021 m/s
```

The action period is `0.05 s` with provenance
`TEST_ONLY_STAGE22_SIMULATION`; production action timing remains
`TBD_BY_ADAPTATION_DATASET`.

This oracle qualifies expert label generation and native reference hot-swap
infrastructure. It is not a GR00T task-performance result.

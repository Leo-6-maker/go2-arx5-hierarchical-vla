# Stage19 D3 Timing Provenance

Date: 2026-09-01 (Asia/Singapore)

## Decision

`STAGE18_DATA_PIPELINE_QUALIFIED_WITH_D3_METADATA_DEVIATION`

The D3 discrepancy is classified as:

`FRAME_SUBSAMPLING_METADATA_DEFECT`

This is a fixture provenance/metadata deviation, not a SynchronizerV1
failure. Source timestamps remain authoritative and no D3 file was changed.

## Evidence

The six selected `*_V0` episodes were:

```text
d3_A_d2_ep_00_V0
d3_A_d2_ep_01_V0
d3_A_d2_ep_02_V0
d3_A_d2_ep_03_V0
d3_B_d2_ep_07_V0
d3_B_d2_ep_08_V0
```

| Stream | Samples | Effective rate from timestamps | Declared rate |
|---|---:|---:|---:|
| native state/command | 121 | 50 Hz | 50 Hz source configuration |
| body camera | 61 | 25 Hz | 20 Hz |
| wrist camera | 61 | 25 Hz | 20 Hz |

The source job configuration declares `camera_hz: 20.0` and `state_hz: 50.0`.
Its camera indices select every second native frame (`stride=2`), yielding
0.04 s source timestamp spacing. The raw generator also writes a literal
20 Hz camera metadata value. The combination explains the observed 25 Hz
timestamps versus 20 Hz declaration without requiring a clock correction.

## Immutable evidence hashes

The node-local audit was run by
`scripts/stage19_d3_provenance.py` on `corelab-5080-2`:

```text
render_jobs.json:       66a046c99ab65a6a560b1ddb11134e3ed8e8c14ce2c6f85a19d485483945ba9a
d3_build_raw.py:        75d74abe27cf12b1b57228b94f734db8149b5b9409adf47c0e6d1577a3f2f713
D3_DATASET_MANIFEST_V0: b81691381f6874d97cac68651bf70c60c22f0f925755fefa606127f920e93a8b
```

The generator audit confirmed that camera timestamps are derived from source
indices, the camera FPS literal is 20, and the index stride is supplied by the
job configuration. Per-episode source JSON hashes and the complete machine
readable result are in:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/d3_provenance/
```

The recorded result SHA256 is
`0ef2c3cd71e230ac542dbccc86a056f3638ff63a698658736b88a7dd4c7b88cf`.

## Policy

- source timestamps are the synchronization authority;
- no forced 20 Hz resampling or retiming was performed;
- production robot action timing remains `TBD_BY_ADAPTATION_DATASET`;
- the D3 issue is retained as a visible deviation in dataset provenance;
- D3 remains a simulation fixture and is not real-robot calibration or
  robot-specific GR00T training evidence.

# Project Status — Stage19 Robot Dataset Materialization

Date: 2026-09-01 (Asia/Singapore)

## Canonical state

- Stage18 was merged to `main` as
  `237be6fc37c20cf44ccccf5f0cfbd3cce629760f` before Stage19 work.
- Working branch: `stage19/robot-dataset-materialization`.
- Stage16 native WBC baseline is frozen as
  `NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`.
- Stage17 EEF-only VLA→WBC interface remains runtime-ready.
- Robot-specific GR00T adaptation and real data collection have not occurred.

## Stage19 result

`PASS WITH DEVIATION`

The D3 source-rate discrepancy is classified as
`FRAME_SUBSAMPLING_METADATA_DEFECT`: 50 Hz native source, 25 Hz camera source
timestamps, stale 20 Hz camera metadata. Source timestamps are authoritative;
no source files were rewritten or resampled.

The node-local materialization and qualification outputs are under:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/
```

The six-episode fixture passed official GR00T loader/processor qualification,
with 132 valid H=40 windows and finite `[40,10] -> [1,40,132] -> [1,40,10]`
processor round-trip. Synchronization stress and D3 recorder replay passed
without corrective edits.

## Gate

`READY_FOR_REAL_5_10_EPISODE_PILOT`

This is conditional readiness for a separately authorized P0/P1 pilot. P0
must perform and review real frame/TCP/camera/joint-zero/gripper/clock
calibration before P1 labels are retained. Stage19 did not operate hardware.

## Hard boundary

No physical collection, hardware command, ROS, ZMQ, PolicyServer, P1M, H200,
model training, finetuning, or actual GR00T closed-loop WBC execution was run.
No D3 data, MP4, parquet, WBC source/archive, checkpoint, token, or private
runtime artifact belongs in Git.

## Next gate

After explicit authorization: run the limited P0 calibration pilot, review the
real `CalibrationBundleV1`, then run 5–10 P1 simple manipulation episodes.

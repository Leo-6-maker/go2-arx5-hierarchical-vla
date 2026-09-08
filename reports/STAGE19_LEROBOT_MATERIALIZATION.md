# Stage19 LeRobot/GR00T Dataset Materialization

## Result

`PASS` for a node-local, six-episode LeRobot v2 fixture consumed by the
official Isaac-GR00T loader. The implementation is
`go2arx5_vla.data.materialization.materialize_lerobot_v2`; it is reusable code
in the canonical repository and does not modify D3 source data.

Node-local output:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/lerobot_fixture/
```

The output is approximately 764 KiB and contains six parquet episode tables,
twelve MP4 camera streams, standard v2 metadata, generated statistics, and
Stage19 provenance sidecars. The fixture uses `H=40` only as an explicitly
labelled test horizon.

## Materialized contract

| Item | Result |
|---|---|
| episodes | 6 |
| synchronized table rows | 366 (61 per episode) |
| body/wrist videos | 6 each, 25 Hz nominal from source timestamps |
| model state | `observation.state`, float32 `[10]` |
| action row | desired physical `[10]` = EEF `[9]` + gripper `[1]` |
| future window | 40 explicit timestamped desired points via sidecar/loader horizon |
| achieved EEF | retained by audit hash only; not a training label |
| frame/TCP/calibration | retained per episode in provenance sidecars |
| episode split | four train, two validation, episode-level |
| production timing | `TBD_BY_ADAPTATION_DATASET` |

The standard files are `meta/info.json`, `episodes.jsonl`, `tasks.jsonl`,
`modality.json`, `stats.json`, `relative_stats.json`, parquet data, and video
files. Stage19-specific files preserve the synchronized sample timestamps,
future desired/achieved hashes, calibration bundles, episode source hashes,
and video provenance.

## Timestamp and label policy

The table contains the source `timestamp` column. The effective fixture FPS in
`info.json` is 25 Hz because it is measured from the selected source
timestamps; the stale D3 20 Hz declaration is not used to retime the data.
The explicit action timestamp sidecar records observation time, all future
point timestamps, desired source indices, and achieved audit hashes. Desired
and achieved streams are never collapsed.

## Reproducibility

Materializer configuration SHA256:

```text
7d94fce2b282a5e01baf18ffb6ede3bc4c09e173e401d7fae108d4c9db853382
```

The builder and run manifests are node-local under
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/`. The post-qualification
manifest is:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/lerobot_fixture/meta/stage19_qualified_manifest.json
```

Its SHA256 is
`a8e35983b551aa9c85dc8265cedcde3c39f663dc4358be7623c8542392c0fcca`.
No parquet, MP4, D3 source, checkpoint, or private runtime artifact is Git
tracked.

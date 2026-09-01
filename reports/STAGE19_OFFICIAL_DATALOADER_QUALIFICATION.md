# Stage19 Official Dataloader Qualification

## Result

`PASS`

The materialized fixture was loaded through the existing Stage10 official
Isaac-GR00T data path on `corelab-5080-2` with no GPU workload:

```text
LeRobotEpisodeLoader
    -> extract_step_data
    -> official Gr00tN1d7Processor
    -> ShardedSingleStepDataset
    -> processor collator
```

The official GR00T v2 loader is available in the Stage10 source tree. The
standalone Python package named `lerobot` is not installed in that runtime;
this is not a blocker because the GR00T training path consumes the standard
LeRobot v2 layout through its shipped loader. This distinction is recorded,
not hidden.

## Evidence

- official episode loader: `gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader`;
- official training dataset: `gr00t.data.dataset.sharded_single_step_dataset.ShardedSingleStepDataset`;
- episodes loaded: `6`, each length `61`;
- valid no-padding action windows: `132 = 6 * (61 - 40 + 1)`;
- state/action modality columns and body/wrist video columns loaded;
- source parquet timestamp column validated as finite and strictly increasing;
- timestamp is not exposed by the episode loader because it is not a declared
  model modality; it remains in parquet and Stage19 timestamp sidecars;
- processor single-step output includes state `[1,132]`, action `[40,132]`,
  `action_mask [40,132]`, language, two images, and embodiment ID;
- collator path produced a finite batch for two samples;
- physical action shape: `[40,10]`;
- universal processor action shape: `[1,40,132]`;
- decoded physical action shape: `[1,40,10]`;
- decoded values finite;
- processor round-trip maximum absolute error:
  `3.162260880928969e-08`;
- official `stats.json` and `relative_stats.json` generation completed;
- official GR00T repository SHA:
  `51d4c89f72fda44cbf77285c6a8114b52676b8a1`;
- Python runtime: `3.12.3`;
- model forward, training, finetuning, WBC, and hardware: `false`.

Machine-readable output:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage19/official_dataloader/stage19_official_dataloader.json
```

Output SHA256:
`1462dc7e69b316d4a88b7f234a368ad5b161a4988a65111db3602dd1d5617eee`.
The qualified dataset manifest records the final stats and artifact hashes.

## Scope boundary

This qualifies dataset materialization, official loading, collation, and
processor ABI behavior. It does not qualify a GR00T model's task performance,
robot-specific adaptation, real calibration, or any physical command path.

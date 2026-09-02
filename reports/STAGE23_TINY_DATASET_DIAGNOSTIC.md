# Stage23 TINY1/TINY6 Diagnostic

Date: 2026-09-03 (Asia/Singapore)

## Scope and result

```text
TINY1/TINY6 ACTION PIPELINE DIAGNOSTIC: PASS
LABEL-AS-PREDICTION DIAGNOSTIC: PASS
OFFICIAL LOADER/COLLATOR/PROCESSOR: PASS
STAGE23 TRAINING: NOT EXECUTED
BASE-ONLY INFERENCE: NOT EXECUTED
WBC: NOT EXECUTED
```

The fixtures were derived only from the frozen Stage22 train fixture and only
from `red_target_train_a`. The script rejects a source fixture containing
holdout `e` or `f` markers and records both holdout reads as false.

## Frozen split

| fixture | source episodes | frames | selection |
|---|---:|---:|---|
| TINY1 | 1 | 40 | lexicographically first `red_target_train_a` snapshot, `expert-0000` |
| TINY6 | 6 | 240 | all six ordered `red_target_train_a` snapshots, `expert-0000`–`expert-0005` |

Both fixtures retain desired EEF actions and gripper labels only as training
labels. Achieved EEF remains part of the raw/source provenance but is not used
as a GR00T training target. Target actor coordinates are not included in model
input.

Source train fixture:

```text
path: /var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/native_expert/lerobot_train
episodes: 24
tree files: 85
tree bytes: 919661
tree SHA256: c371b07183d238ffe60b748ab50dbf811b676cd8d572760e934598e053893ad2
```

## Materialization and official path

Each fixture was materialized as a real LeRobot-v2 directory containing
parquet, videos, metadata, timestamps, and split lock. The qualification path
was:

```text
LeRobotEpisodeLoader
  -> ShardedSingleStepDataset
  -> official Gr00tN1d7Processor
```

No model forward was called. The official processor was resolved entirely from
the frozen Stage10 cache:

```text
Cosmos repo: nvidia/Cosmos-Reason2-2B
revision:    9ce19a195e423419c349abfc86fd07178b230561
resolution:  local_files_only=true, offline=true
GR00T SHA:   51d4c89f72fda44cbf77285c6a8114b52676b8a1
```

The installed native `lerobot` Python package was not required and was not
available in this Stage10 environment. The official GR00T loader and training
dataset implementation still completed the qualification.

| fixture | episodes loaded | shard rows | single-step finite | batch finite | universal output | decoded physical |
|---|---:|---:|---|---|---|---|
| TINY1 | 1 | 1 | true | true | `[1,40,132]` | `[1,40,10]` |
| TINY6 | 6 | 6 | true | true | `[1,40,132]` | `[1,40,10]` |

The loader exposed state, desired action, both RGB videos, and language. The
processor exposed the official `action_mask`; it did not expose achieved EEF
as a training label.

## Round-trip and identity checks

| check | TINY1 | TINY6 |
|---|---:|---:|
| materialized action max absolute error | `2.6876311e-08` | `2.6913313e-08` |
| materialized timestamp max absolute error | `0.0 s` | `0.0 s` |
| strictly increasing timestamps | true | true |
| label-as-prediction max position error | `0.0 m` | `0.0 m` |
| label-as-prediction max rotation error | `0.0 rad` | `0.0 rad` |
| relative round-trip max position error | `1.9930455e-17 m` | `6.3596013e-17 m` |
| relative round-trip max rotation error | `0.0 rad` | `0.0 rad` |
| label-as-prediction SafetyGate | `1/1 ACCEPT` | `6/6 ACCEPT` |
| label path-length/mean-speed/max-speed ratios | `1.0/1.0/1.0` | `1.0/1.0/1.0` |
| label roughness ratio | `1.0` | `1.0` |

The identity prediction therefore recovers the ideal evaluator result. The
evaluator is not the current explanation for the Stage22 model-profile
blocker.

## Node-local artifact seals

The derived fixtures are not committed. Their final node-local tree seals are:

| fixture | path | files | bytes | tree SHA256 |
|---|---|---:|---:|---|
| TINY1 | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_retry8/tiny1_lerobot_v2` | 13 | 152651 | `6213f21c3bc22cb505f866334401d0d1b591bc910ccfbf80096bc67b0ebc3420` |
| TINY6 | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_retry8/tiny6_lerobot_v2` | 28 | 298646 | `be99ebfff79795047f950c0fecfde41e8e8b1541921b32cdee43109e54c12b3a` |

Machine-readable diagnostic output:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_retry8/stage23_action_pipeline_diagnostic.json
```

## Resource and safety assertions

The six direct CPU contract checks passed on the Stage10 Python runtime:
split, universal-mask partition, one-reference SE(3) transform, label
identity, wrong-horizon rejection, and timing-contract provenance. The
repository also contains five pytest unit tests for these same boundaries;
the remote runtime did not have pytest installed, so the direct checks and
the full official loader/processor diagnostic are the executed evidence.

```text
host: corelab-4080s
python: 3.12.3
GPU used: false
model forward: false
backward: false
optimizer step: false
training/fine-tuning: false
PBS submitted: false
WBC executed: false
hardware control: false
validation e read: false
locked final f read: false
```

The explicit FFmpeg library path used for the final CPU qualification is an
environment reproducibility requirement, not a scientific transform or model
change. The run used the existing Stage13 public native library directory and
did not install packages or alter shared system libraries.

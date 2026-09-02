# Stage23 Tiny-Statistics and H2-Statistics Diagnostic

Date: 2026-09-03 (Asia/Singapore)

## Scope and result

```text
TINY1_H2STATS/TINY1_TINYSTATS/TINY6_TINYSTATS ACTION PIPELINE DIAGNOSTIC: PASS
LABEL-AS-PREDICTION DIAGNOSTIC: PASS
OFFICIAL LOADER/COLLATOR/PROCESSOR: PASS
STAGE23 TRAINING: NOT EXECUTED
BASE-ONLY INFERENCE: NOT EXECUTED
WBC: NOT EXECUTED
```

The fixtures were derived only from the frozen Stage22 train fixture and only
from `red_target_train_a`. The script rejects a source fixture containing
holdout `e` or `f` markers and records both holdout reads as false.

## Frozen split and statistics variants

| fixture | source episodes | frames | selection |
|---|---:|---:|---|
| `TINY1_H2STATS` | 1 | 40 | first ordered `red_target_train_a` snapshot, `expert-0000`; exact Stage22 frozen stats |
| `TINY1_TINYSTATS` | 1 | 40 | same snapshot; stats regenerated from the one-episode fixture |
| `TINY6_TINYSTATS` | 6 | 240 | all ordered `red_target_train_a` snapshots; stats regenerated from the six-episode fixture |

All variants retain desired EEF actions and gripper labels only as training
labels. Achieved EEF remains part of the raw/source provenance but is not used
as a GR00T training target. Target actor coordinates are not included in model
input.

Source train fixture:

```text
path: /var/cache/users/liuyu-nfs/go2-arx5-vla-stage22/native_expert/lerobot_train
episodes: 24
tree files: 85
tree bytes: 919661
Stage23 local-manifest SHA256: c371b07183d238ffe60b748ab50dbf811b676cd8d572760e934598e053893ad2
Stage22 canonical handoff SHA256: 2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789
canonical files/bytes/symlinks: 85 / 919661 / 0
canonical identity matches frozen Stage22 source: true
```

The two tree hashes use different, explicitly recorded algorithms. The
Stage22 digest is the authoritative source identity; the Stage23 local hash is
retained only as a diagnostic manifest identity.

Frozen Stage22 stats:

```text
stats.json:          fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
relative_stats.json: 8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
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

| fixture | episodes loaded | shard rows | single-step finite | batch finite | universal output | decoded physical | stats mode |
|---|---:|---:|---|---|---|---|---|
| `TINY1_H2STATS` | 1 | 1 | true | true | `[1,40,132]` | `[1,40,10]` | `STAGE22_FROZEN_COPIED` |
| `TINY1_TINYSTATS` | 1 | 1 | true | true | `[1,40,132]` | `[1,40,10]` | `TINY_FIXTURE_REGENERATED` |
| `TINY6_TINYSTATS` | 6 | 6 | true | true | `[1,40,132]` | `[1,40,10]` | `TINY_FIXTURE_REGENERATED` |

The loader exposed state, desired action, both RGB videos, and language. The
processor exposed the official `action_mask`; it did not expose achieved EEF
as a training label.

## Round-trip and identity checks

| check | `TINY1_H2STATS` | `TINY1_TINYSTATS` | `TINY6_TINYSTATS` |
|---|---:|---:|---:|
| materialized action max absolute error | `2.6876311e-08` | `2.6876311e-08` | `2.6913313e-08` |
| materialized timestamp max absolute error | `0.0 s` | `0.0 s` | `0.0 s` |
| strictly increasing timestamps | true | true | true |
| label-as-prediction max position error | `0.0 m` | `0.0 m` | `0.0 m` |
| label-as-prediction max rotation error | `0.0 rad` | `0.0 rad` | `0.0 rad` |
| relative round-trip max position error | `1.9930455e-17 m` | `1.9930455e-17 m` | `6.3596013e-17 m` |
| relative round-trip max rotation error | `0.0 rad` | `0.0 rad` | `0.0 rad` |
| label-as-prediction SafetyGate | `1/1 ACCEPT` | `1/1 ACCEPT` | `6/6 ACCEPT` |
| label path-length/mean-speed/max-speed ratios | `1.0/1.0/1.0` | `1.0/1.0/1.0` | `1.0/1.0/1.0` |
| label roughness ratio | `1.0` | `1.0` | `1.0` |

Normalization statistics are intentionally not interchangeable:

| fixture | `stats.json` SHA256 | `relative_stats.json` SHA256 | interpretation |
|---|---|---|---|
| `TINY1_H2STATS` | `fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23` | `8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1` | primary future O1 input; copied from Stage22 source |
| `TINY1_TINYSTATS` | `e4ac13cdc6030fe081148ef79931133dcb91405385486a982fa93de4f0caa9f3` | `a3602929d2980862a54039390f959ff42193ed4698d59fe6ae00aada47701b93` | retained one-snapshot diagnostic |
| `TINY6_TINYSTATS` | `12b80d02addcd535ea548b7d149bed07c9ae2a46b9c097fbb0a3d948d138926b` | `50dcbe936e03dc3848144ae45151eb29c92ebd9fc7cebbaeb272fed8a0fba137` | retained six-snapshot diagnostic |

The identity prediction therefore recovers the ideal evaluator result. The
evaluator is not the current explanation for the Stage22 model-profile
blocker.

## Node-local artifact seals

The derived fixtures are not committed. Their final node-local tree seals are:

| fixture | path | files | bytes | tree SHA256 |
|---|---|---:|---:|---|
| `TINY1_H2STATS` | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_p0/tiny1_h2stats_lerobot_v2` | 13 | 152958 | `541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157` |
| `TINY1_TINYSTATS` | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_p0/tiny1_tinystats_lerobot_v2` | 13 | 151002 | `5e876bd65f70f4e9435f90ab491f2e36610295e488c5856d1b054c15417b0cb5` |
| `TINY6_TINYSTATS` | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_p0/tiny6_tinystats_lerobot_v2` | 28 | 299073 | `c9a9a26c0eaea2c365b223b4a9b53c728e46b00d43ddf185888a7675ea143b0c` |

Machine-readable diagnostic output:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_p0/stage23_action_pipeline_diagnostic.json
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

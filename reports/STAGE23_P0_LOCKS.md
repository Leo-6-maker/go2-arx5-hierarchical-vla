# Stage23 P0 Locks

Date: 2026-09-03 (Asia/Singapore)

## Source identity

The Stage22 train fixture was rehashed with the exact Stage22 H200 handoff
algorithm:

```text
method: STAGE22_PREPARE_H200_HANDOFF_TREE_DIGEST_V1
file_count: 85
total_bytes: 919661
symlink_count: 0
tree_sha256: 2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789
matches_frozen_stage22_identity: true
```

The per-file entries and SHA256 values are retained in the node-local machine
diagnostic. No content change was found. The earlier
`c371b07183d238ffe60b748ab50dbf811b676cd8d572760e934598e053893ad2` value is
the Stage23 local manifest algorithm and is not the Stage22 source identity.

## Statistics provenance

The frozen Stage22 source statistics are:

```text
stats.json:          fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
relative_stats.json: 8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
```

| fixture | mode | official path | role |
|---|---|---|---|
| `TINY1_H2STATS` | `STAGE22_FROZEN_COPIED` | PASS | primary future O1 input |
| `TINY1_TINYSTATS` | `TINY_FIXTURE_REGENERATED` | PASS | retained one-snapshot diagnostic |
| `TINY6_TINYSTATS` | `TINY_FIXTURE_REGENERATED` | PASS | retained six-snapshot diagnostic |

The primary H2-stats fixture is byte-identical to the Stage22 source stats.
The two regenerated variants have distinct hashes and are deliberately kept
as separate datasets.

## Safety boundary

The following is the safety boundary for the earlier P0 action-pipeline
diagnostic only. It is not a statement that no later Stage23 Phase A work was
performed; the Stage23 final ledger records the subsequent Hopper O1 runs.

```text
P0 action-pipeline diagnostic training: NOT EXECUTED
model forward/inference: NOT EXECUTED
Hopper/PBS: NOT SUBMITTED
validation e: NOT READ
locked final f: NOT READ
WBC/hardware/ROS/ZMQ/P1M: NOT EXECUTED
```

Machine-readable evidence:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage23/action_pipeline_20260903_p0/stage23_action_pipeline_diagnostic.json
```

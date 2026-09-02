# Stage23 Final Report

Date: 2026-09-03 (Asia/Singapore)

## A. Classification

```text
PASS WITH DEVIATION
```

Primary diagnostic gate:

```text
STAGE23_ACTION_PIPELINE_AND_EVALUATOR_SANITY_PASS
```

The deviations are operational/provenance-only: official video decoding
required the existing Stage13 native FFmpeg library path, the environment did
not provide the standalone `lerobot` package, and the first tiny-fixture
diagnostic did not make its statistics provenance explicit. The P0 rerun now
does so and passes the exact Stage22 source identity and all three official
loader variants without a model forward.

## B. Action contract conclusion

```text
physical action: 10D
universal action: [40,132]
decoded physical action: [40,10] / batched [1,40,10]
active loss dimensions: 10
masked padding dimensions: 122
padding contributes to loss: false
EEF transform: T_reference^-1 @ T_absolute_target
reference: last observed state[eef_9d]
rotation: first two matrix rows, row-major rot6d
gripper: absolute separate channel
production timing: TBD_BY_ADAPTATION_DATASET
simulation timing: 0.05 s TEST_ONLY_STAGE22_SIMULATION
```

No action-pipeline, normalization, ordering, or evaluator defect was found.

## C. TINY fixtures and P0 locks

```text
TINY1_H2STATS: one ordered red_target_train_a snapshot plus exact Stage22 frozen stats
TINY1_TINYSTATS: same one snapshot plus regenerated one-fixture stats
TINY6_TINYSTATS: all six ordered red_target_train_a snapshots plus regenerated six-fixture stats
```

All variants were built from the train-only Stage22 materialization. Desired
labels were preserved; achieved EEF was not used as a training label; holdout
`e` and `f` were not read. The Stage22 canonical source digest was recomputed
with 85 files, 919661 bytes, zero symlinks, and SHA256
`2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789`.
The former `c371b071...` value is a separate Stage23 local-manifest hash.

The primary future O1 input is frozen as `TINY1_H2STATS`; the tiny-stat
variants remain retained diagnostics and are not interchangeable with O1.

## D. Label-as-prediction sanity

```text
TINY1_H2STATS: 1/1 identity pass, 0 m position error, 0 rad rotation error
TINY1_TINYSTATS: 1/1 identity pass, 0 m position error, 0 rad rotation error
TINY6_TINYSTATS: 6/6 identity pass, 0 m position error, 0 rad rotation error
all identity SafetyGate decisions: ACCEPT
all identity profile ratios: exactly 1.0 within the audited tolerance
```

This closes the evaluator-or-action-contract sanity gate. It does not imply
that the base model or a fine-tuned model has learned the task.

## E. Official GR00T path

```text
Stage10 runtime Python: 3.12.3
GR00T SHA: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
Cosmos revision: 9ce19a195e423419c349abfc86fd07178b230561
processor resolution: local_files_only=true, offline=true
TINY1_H2STATS: LeRobotEpisodeLoader -> ShardedSingleStepDataset -> processor PASS
TINY1_TINYSTATS: LeRobotEpisodeLoader -> ShardedSingleStepDataset -> processor PASS
TINY6_TINYSTATS: LeRobotEpisodeLoader -> ShardedSingleStepDataset -> processor PASS
```

All paths produced finite `[1,40,132]` universal actions and finite
`[1,40,10]` decoded physical actions. Official model forward was false.

## F. Execution boundary

```text
Stage23 training/fine-tuning: NOT EXECUTED
BASE-only inference: NOT EXECUTED
GPU workload: NOT EXECUTED
PBS job: NOT SUBMITTED
checkpoint: not created or changed
WBC: NOT EXECUTED
hardware/ROS/ZMQ/P1M: NOT EXECUTED
```

The executed CPU evidence includes six direct contract checks. Five pytest
tests were added for CI; pytest was not installed in the Stage10 runtime and
was not installed during this stage.

## G. Scientific decision

The action pipeline is sufficiently specified for a future learnability test.
The Stage22 profile blocker cannot be attributed to the audited 10D/132D
representation, relative transform, normalization selection, timestamp order,
or padding loss mask based on this diagnostic.

The possible future Hopper candidate is prepared but remains locked:

```text
O1 TINY1_H2STATS deliberate overfit: NOT EXECUTED
secondary TINY1_TINYSTATS diagnostic: NOT EXECUTED
O6 TINY6_TINYSTATS conditional diagnostic: NOT EXECUTED
```

The next gate is PI review and explicit authorization of TINY1_H2STATS
deliberate overfit, with the retained tiny-stat variants considered only for a
predeclared diagnostic reason.

Detailed evidence is in:

```text
reports/STAGE23_ACTION_PIPELINE_AUDIT.md
reports/STAGE23_TINY_DATASET_DIAGNOSTIC.md
reports/STAGE23_HOPPER_OVERFIT_PROTOCOL.md
```

# ADR-026: Action-Pipeline Learnability Diagnostic Before Hopper Fine-Tuning

Date: 2026-09-03

Status: Accepted for Stage23

## Context

Stage22 showed that diffusion-module tuning improved positional error but did
not close the future action-profile blocker. Before spending additional H200
time, the project needed to distinguish a model learnability problem from an
absolute/relative transform, normalization, universal-action padding, timing,
or evaluator defect.

## Decision

Stage23 uses two train-only fixtures:

```text
TINY1 = the lexicographically first snapshot in red_target_train_a
TINY6 = all six ordered snapshots in red_target_train_a
```

Both fixtures must pass the official materialized-data path and the exact
evaluator with expert labels supplied as predictions. That identity prediction
must produce zero task-space error, unit profile ratios, finite values, and an
accepted safety decision before any future Hopper fine-tuning is considered.

The physical action ABI remains 10D while the official universal tensor remains
132D. The first ten dimensions are active and the remaining 122 dimensions are
masked from loss. EEF actions are converted from absolute future poses to
one-reference relative poses using the last observed EEF state. Gripper remains
an absolute separate channel. Simulation timing remains test-only metadata;
production timing is not frozen by this ADR.

## Consequences

This diagnostic can establish that the action pipeline is coherent without
claiming that the base model learned a robot-specific task. It prevents a GPU
run from simultaneously changing the action representation and the model
recipe. It also keeps achieved EEF separate from desired EEF training labels.

O1/O6 Hopper overfit runs are prepared only; they require a separate explicit
authorization after the Stage23 evidence review. No real robot, WBC execution,
or deployment path is implied.

## Evidence

See:

```text
reports/STAGE23_ACTION_PIPELINE_AUDIT.md
reports/STAGE23_TINY_DATASET_DIAGNOSTIC.md
reports/STAGE23_HOPPER_OVERFIT_PROTOCOL.md
```

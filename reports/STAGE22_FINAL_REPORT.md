# Stage22 Final Report

Date: 2026-09-01 (Asia/Singapore)

> This report is an interim Stage22 record. The native expert/data gates are
> complete; the R0/R1 model-profile comparison remains pending the resource
> gate and will be updated in the same branch before promotion. The train /
> validation / final-test split was locked before any GPU inference.

## A. Classification

```text
PASS WITH DEVIATION
```

Completed: same-scene native expert supervision, on-policy replanning-point
fixture generation, official loader/materialization qualification, and CPU
contract tests. Deviation: the action-generation attribution comparison has
not run because the tested shared GPUs, including a nearly-free 32 GiB 4080S
device, failed the one-step R1 OOM/headroom gate. No full R0/R1 run or
model-profile inference is claimed.

## B. Expert oracle and data

```text
6 episodes, 36 snapshots, 144/144 native steps
30 reference updates, 0 reset/fall/invalid events
6/6 EXPERT_SAME_SCENE_RECEDING_HORIZON_PASS
```

The labels are exact same-scene, current-state-conditioned desired EEF
trajectories. `StateV1` is 10D; future labels are desired EEF `[40,9]` plus
gripper `[40,1]`. Achieved EEF remains an audit stream, never the training
label. Target positions are excluded from model input.

## C. Materialization and official path

The historical all-target loader fixture has 36 one-window H=40 episodes and
is explicitly marked `FULL_FIXTURE_NOT_FOR_TRAINING`. The locked training
fixture is a separate train-only materialization with 24 one-window H=40
episodes from `red_target_train_a/b/c/d`; validation is only
`red_target_holdout_e`, and final test is only `red_target_holdout_f`. The
official episode loader, single-step dataset, collator, and N1.7 processor
passed on the train-only fixture with finite `[1,40,132]` universal and
`[1,40,10]` decoded physical action shapes; round-trip max error was
`3.584885e-08`. No training occurred in this gate.

The tracked lock is `manifests/STAGE22_TARGET_SPLIT_LOCK_V1.json`; the
training launcher rejects any dataset without a matching train-only sidecar.

## D. R0/R1 model profile gate

```text
R0: implemented; checkpoint not yet evaluated
R1: implemented; one-step memory preflight pending
profile selection: held-out trajectory geometry, not training loss
output correction: none
```

Required metrics are mean/median/p95/max position and rotation error,
endpoint error, predicted/expert path-length ratio, speed ratios, roughness,
initial jump, SafetyGate, and full common-future overlap inconsistency.

## E. Visual conditioning

The counterfactual evaluator is implemented for same-state, same-language,
different-target native images. Until R0/R1 results are available, the only
permitted interpretation is `RGB_SENSITIVITY_ONLY`, not visual grounding.

## F. Claim boundary and next gate

```text
ASYNC_RUNTIME_NOT_QUALIFIED
SIM_MODEL_PROFILE_BLOCKER: NOT_YET_RETESTED
SIM_END_TO_END_VLA_WBC_CLOSED_LOOP_QUALIFIED: NOT_CLAIMED
ROBOT_SPECIFIC_GR00T_TASK_PERFORMANCE_NOT_QUALIFIED
```

The next action is to run the declared R1 one-step resource preflight, then
matched R0/R1 training and held-out profile evaluation on an uncontended
compatible GPU. If R1 cannot close the profile blocker, report that result
directly rather than adding unplanned tuning variables.

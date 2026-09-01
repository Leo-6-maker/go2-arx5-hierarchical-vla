# Project Status — Stage22

Date: 2026-09-01

## Current branch

```text
stage22/action-profile-adaptation
base: be45114da35932182bb8af0bbfec2eb2971b5a6b
origin: git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git (private)
latest pushed Stage22 tip before this handoff update: `28837c2`
```

## Completed

- Native same-scene expert oracle: 6/6 pass, fixed seed 13, K=4.
- On-policy replanning snapshots: 36 exact native RGB/state observations.
- Desired-vs-achieved separation and target exclusion from model input.
- Materialized LeRobot-v2-compatible fixture and official GR00T loader,
  collator, and processor qualification.
- R0/R1 launcher, non-corrective profile metrics, chunk-overlap metrics, and
  visual counterfactual evaluator.
- Locked split manifest and train-only materialization guard: train targets
  `red_target_train_a/b/c/d`, validation `red_target_holdout_e`, final test
  `red_target_holdout_f`; final test is not used for training or selection.
- Auditable occupied-GPU refusal and R1 one-step resource preflight, plus
  matched diffusion-seed aggregation and frozen profile gate.
- After explicit authorization to share GPUs with free VRAM, tested the
  5080-2 and 4080S R1 one-step preflights. The 5080-2 failed during
  `model.to(device)` and the 4080S reached `optimizer.step()` but failed with
  CUDA OOM before completing it. No full R0/R1 adaptation or profile inference
  was run; no other user's process was changed.
- A later clean 5080-2 retry using the existing Stage10 environment reached
  `optimizer.step()` but failed while allocating Adam state on the 16 GiB
  device (`15,792/16,303 MiB` peak, `511 MiB` headroom versus `2,445 MiB`
  required). The process left no external residual allocation.
- CPU tests for expert contracts, provider, timestamps, SafetyGate, and
  action-profile calculations.

## Pending

- R1 one-step GPU memory qualification on a compatible device with sufficient
  headroom (the tested clean 16 GiB and shared 32 GiB attempts were OOM).
- Matched R0/R1 short adaptation runs.
- Held-out profile/overlap/visual-conditioning measurements.
- Final decision on `SIM_MODEL_PROFILE_BLOCKER_CLOSED` versus
  `DIFFUSION_ACTION_TUNING_DID_NOT_CLOSE_PROFILE_BLOCKER`.

## Boundary

Simulation-only. No hardware control, real data collection, WBC training,
async transport, P1M, ROS, ZMQ, or task-performance claim.

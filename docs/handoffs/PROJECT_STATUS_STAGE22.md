# Project Status — Stage22

Date: 2026-09-01

## Current branch

```text
stage22/action-profile-adaptation
base: be45114da35932182bb8af0bbfec2eb2971b5a6b
origin: git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git (private)
```

## Completed

- Native same-scene expert oracle: 6/6 pass, fixed seed 13, K=4.
- On-policy replanning snapshots: 36 exact native RGB/state observations.
- Desired-vs-achieved separation and target exclusion from model input.
- Materialized LeRobot-v2-compatible fixture and official GR00T loader,
  collator, and processor qualification.
- R0/R1 launcher, non-corrective profile metrics, chunk-overlap metrics, and
  visual counterfactual evaluator.
- CPU tests for expert contracts, provider, timestamps, SafetyGate, and
  action-profile calculations.

## Pending

- R1 one-step GPU memory qualification.
- Matched R0/R1 short adaptation runs.
- Held-out profile/overlap/visual-conditioning measurements.
- Final decision on `SIM_MODEL_PROFILE_BLOCKER_CLOSED` versus
  `DIFFUSION_ACTION_TUNING_DID_NOT_CLOSE_PROFILE_BLOCKER`.

## Boundary

Simulation-only. No hardware control, real data collection, WBC training,
async transport, P1M, ROS, ZMQ, or task-performance claim.

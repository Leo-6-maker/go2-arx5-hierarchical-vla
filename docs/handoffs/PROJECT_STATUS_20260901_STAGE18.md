# Project Status — Stage18 Robot-Specific Data Readiness

Date: 2026-09-01 (Asia/Singapore)

## Canonical state

- Stage15–17 promotion merged to `main` as `a016dcba0a1afa727a084aa746a8ae725068e417`.
- Working branch: `stage18/robot-data-readiness`.
- Stage16 native WBC baseline is frozen as
  `NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`.
- Stage17 VLA→WBC software interface remains runtime-ready; actual
  robot-specific GR00T adaptation has not been performed.

## Stage18 implementation

```text
RawEpisodeV1
    -> CalibrationBundleV1
    -> SynchronizerV1
    -> EpisodeQualityGateV1
    -> GR00TTrainingExampleV1
    -> LeRobot-compatible logical record
    -> official GR00T processor boundary
```

The reusable CPU layer and tests are in `src/go2arx5_vla/data/`. D3 fixture
artifacts are node-local under
`/var/cache/users/liuyu-nfs/go2-arx5-vla-stage18/` and are not copied into Git.

## Current gate

`ONE_SPECIFIC_DATA_PIPELINE_BLOCKER_REMAINS`

The specific blocker is the D3 declared camera FPS (20 Hz) versus effective
source timestamp rate (25 Hz). Source timestamps are preserved and the issue
is visible; no silent retiming or metadata rewrite was performed.

The official processor smoke passed after launching with the complete Stage10
`env.sh` plus venv. It verified the D3 target record at processed state
`[1,132]`, universal action `[1,40,132]`, decoded physical `[1,40,10]`, and
finite round-trip (max abs `3.9073288e-06`). Existing D3 processor logs remain
prior corroborating evidence. No token action was taken and no model forward
was run in Stage18.

## Hard boundary

No real episode collection, hardware control, ROS/ZMQ/PolicyServer transport,
H200 use, training, finetuning, P1M return, or WBC/checkpoint modification
occurred.

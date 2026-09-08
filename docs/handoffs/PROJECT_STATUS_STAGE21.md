# Project Status — Stage21

Date: 2026-09-01 (Asia/Singapore)

## Canonical branch

```text
repository: /home/liuyu-nfs/Projects/go2-arx5-hierarchical-vla
branch:     stage21/same-scene-receding-horizon
parent:     58fe006a4986386b923bcef2b7a43d6f6559bbff
commit:     91724c56c13bfea8cce5e47cfbd9484bacb6bb76
origin:     git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git
```

The branch is a normal descendant of Stage20T. No prior history was reset or
rewritten. The configured remote is the private repository shown above.

## Stage21 decision

```text
classification: PASS WITH DEVIATION
gate:           ONE_SPECIFIC_MODEL_PROFILE_BLOCKER_REMAINS
async:          ASYNC_RUNTIME_NOT_QUALIFIED
```

Native camera and GT-first reference hot-swap infrastructure are qualified
for the controlled simulation harness. The actual model also traversed
same-scene RGB/state -> model -> provider -> gate -> binding -> adapter ->
native WBC in a synchronous paused simulation. It is not promoted as a
qualified receding-horizon VLA because the unmodified model's raw predicted
chunk handoffs were discontinuous.

## Evidence anchors

- camera smoke: `/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/continuity/camera_smoke/`;
- actual model inference/interface seal:
  `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage21/same_scene_model/` and
  `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage21/same_scene_interface_seal/`;
- GT hot-swap:
  `/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/receding_horizon/gt_k4_20260901_telemetry/`;
- actual model receding trace:
  `/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/receding_horizon/model_k4_20260901_telemetry/`.

The actual model run used seed 13, frozen `K=4`, native control dt
`0.01999999955 s`, and six accepted chunks over 24 native steps. It had no
reset, fall, invalid state, timeout, or explicit native joint-limit
termination. Task success was not evaluated.

## Next gate

Keep the native WBC and interface contracts frozen. Address the one specific
model-profile issue with a declared action/profile attribution experiment
before attempting asynchronous transport or claiming a qualified
same-scene receding-horizon system. Do not infer a production action period
from the `0.05 s` test fixture.

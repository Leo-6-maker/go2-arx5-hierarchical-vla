# Project Status — Stage20T

Date: 2026-09-01 (Asia/Singapore)

## Canonical branch

```text
repository: /home/liuyu-nfs/Projects/go2-arx5-hierarchical-vla
branch:     stage20t/native-wbc-timing-qualification
parent:     bb0d9cf59401a5418501a34141da574da04aeb56
origin:     git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git
```

The branch is stacked on the open Stage20S branch/PR. It is not a merge or
history rewrite of Stage16--20S.

## Scientific result

```text
classification: PASS WITH DEVIATION
primary:       MODEL_TRAJECTORY_PROFILE_LIMITATION_IDENTIFIED
secondary:     NATIVE_TIMING_BOUNDARY_CHARACTERIZED
```

The native baseline remains frozen as
`NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`.

## Evidence

- 33 native simulation runs: 15 model, 15 same-observation GT label, 3
  Stage16 control;
- original timing: model 0/3 complete, GT 2/3 complete, control 2/3 complete;
- seeds 13/14 support model-profile attribution;
- seed 15 aborts A/B/C at the shared reset boundary;
- all 33 gates `ACCEPT`, 14 reference completions, 19 tracking aborts;
- zero fall, invalid-state, timeout, or true joint-limit terminations;
- no threshold/checkpoint/controller semantics changed;
- the 0.75x native filter branch is a documented confound, so primary
  attribution uses unfiltered 1.00x--2.00x.

## Artifact location

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20t/timing_sensitivity_v1_full/
```

See `reports/STAGE20T_FINAL_REPORT.md`,
`reports/STAGE20T_MODEL_VS_GT_REFERENCE.md`, and
`reports/STAGE20T_STAGE16_TIMING_ENVELOPE.md` for the evidence boundary.

## Restrictions retained

No physical robot, hardware ROS, P1M, ZMQ, PolicyServer, camera inference,
new GR00T inference, training, finetuning, or production action-frequency
decision was performed in Stage20T.

## Next gate

```text
SAME_SCENE_RECEDING_HORIZON_VLA_WBC_GATE
```

The next stage must render native-scene RGB, execute only a chunk prefix,
re-observe, and preserve explicit chunk age/freshness/stale/out-of-order
semantics. It must also include a fixed-reset control so seed-dependent native
initial discontinuity is not mistaken for model behavior.

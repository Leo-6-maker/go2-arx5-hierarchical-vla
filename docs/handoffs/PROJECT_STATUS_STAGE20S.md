# Project Status — Stage20S

Date: 2026-09-01 (Asia/Singapore)

## Canonical state

- parent Stage19 tip: `33760f738029dda225ec91f49fb26fa2e18965db`;
- working branch: `stage20s/sim-vla-wbc-closed-loop`;
- Stage16 native baseline: `NATIVE_GENERAL_WBC_BASELINE_QUALIFIED_WITH_BOUNDARY_LIMITATION`;
- Stage19 data pipeline: qualified for limited real pilot, but no hardware
  authorization was used here;
- Stage20S scope: simulation-only.

## Result

`PASS WITH DEVIATION`

The SIM_VISUAL_REACH_V0 fixture materialized through the official GR00T loader
and processor, the resource-bounded custom adaptation ran, actual N1.7 output
was sealed, and a gated native WBC simulation executed it. The known-good
benchmark orchestration completed. The actual model-driven rollout stopped at
the native timing termination after 25 samples, so
`SIM_END_TO_END_VLA_WBC_CLOSED_LOOP_QUALIFIED` was not awarded.

## Evidence locations

Node-local artifacts remain under:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20s/
```

The reusable source and evidence boundary are documented in the Stage20S
reports. The progressive fixture is cross-simulator and offline; it is not a
native same-scene camera/base-coordination qualification.

## Hard boundaries

No physical robot, hardware ROS, motor command, teleoperation, real recording,
P1M, ZMQ, PolicyServer, or H200 was used. No model checkpoint, WBC source,
archive, dataset, video, or credential belongs in Git.

## Next gate

Isolate the native WBC timing-abort sensitivity in simulation without changing
controller semantics. Keep model-driven progressive WBC execution deferred
until the easy visual-reach lifecycle gate is qualified.

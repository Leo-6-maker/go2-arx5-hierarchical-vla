# Stage21 Actual GR00T to Native WBC Receding-Horizon Run

## Execution result

```text
SIM_SAME_SCENE_MODEL_RECEDING_HORIZON_EXECUTED
mode = SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON
```

Artifacts:

```text
/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/receding_horizon/model_k4_20260901_telemetry/model_receding_summary.json
/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/receding_horizon/model_k4_20260901_telemetry/model_receding_rows.jsonl
/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage21/receding_horizon/model_k4_20260901_telemetry/receding_metrics.json
```

The fixed seed was `13`, target marker `[0.30, 0.10, 0.58]`, and `K=4`.
There were 24 valid native steps, six actual-model chunks, five reference
updates, zero reset events, and no native task termination. All six chunks
were `ACCEPT` by `TrajectorySafetyGateV1`, then passed provider, binding,
adapter preview, and native loading. The actual model was never sent to WBC
before the gate.

All six requests came from the same native simulator instance
`native-isaacgym-pid-2691550`. Every update was `FRESH`; physics root/DOF
state, episode length, and arm/dog policy history remained unchanged.

## Latency boundary

Actual N1.7 forward latency was `0.1492 s` mean, `0.0633 s` median,
`0.4507 s` P95, and `0.5798 s` max; peak allocated VRAM was `6089.454 MiB`
on an RTX 5080. The offline shared-NFS mailbox made observed response wait
`29.7257 s` mean and `30.0168 s` P95. This is filesystem staging/cache
visibility latency, not model forward latency and not a real-time result.

The final achieved EEF was `0.10855 m` from the target marker. That number is
descriptive only: task success was deliberately not evaluated, and no
end-to-end task-performance claim is made.

## Promotion decision

The execution chain is functional, but Stage21 qualification is withheld by
the full-chunk profile and raw chunk handoff discontinuities documented in
`STAGE21_MODEL_PROFILE_AUDIT.md`:

```text
ONE_SPECIFIC_MODEL_PROFILE_BLOCKER_REMAINS
```

This is not a native-camera blocker and not a reference-hot-swap blocker.

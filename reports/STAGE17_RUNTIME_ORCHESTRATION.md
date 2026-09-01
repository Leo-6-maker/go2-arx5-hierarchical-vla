# Stage17 Known-Good Runtime Orchestration

## Result

`PASS`

The complete runtime architecture was validated with a known-good Stage16
trajectory, not with actual GR00T semantics:

```text
BenchmarkTrajectoryProvider
  -> TrajectoryChunkV1
  -> TrajectorySafetyGateV1
  -> EpisodeReferenceBindingV1
  -> GR00TToWBCAdapter
  -> WBCPreviewV1
  -> native WBC
```

Actual GR00T was not called in this run. No ZMQ, ROS, PolicyServer, or other
network transport was introduced.

## Native execution

- host: `corelab-4080s`
- reference: `E1_W0_00`
- source: `/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage16/input_binding_seed13/E1_W0_00.npz`
- mode: `BENCHMARK_CONTINUOUS`
- reset seed: `13`
- native IsaacGym/PhysX, headless, no camera/writer
- arm checkpoint SHA256:
  `99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a`
- dog checkpoint SHA256:
  `51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670`
- native config SHA256:
  `56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb`
- native WBC summary: `REFERENCE_COMPLETE`, 98 logged steps, no fall/reset

Native task-space summary:

| Metric | Result |
|---|---:|
| position mean / RMSE / p95 / max | `.012934 / .014983 / .026771 / .038226 m` |
| rotation mean / RMSE / p95 / max | `.136018 / .191413 / .455563 / .565715 rad` |
| time in tolerance | `.795918` |
| path completion | `.905684` |
| base displacement / path length | `.233496 / .529792 m` |
| base yaw change | `.455697 rad` |

## Interface and chunk lifecycle

The gate returned `ACCEPT`; initial position discontinuity was `0 m` and
rotation discontinuity was `2.107e-8 rad`. The identity external binding
preserved EEF positions exactly, and the preview was finite with shape `[9,9]`.

The abstract transport lifecycle produced the expected states:

```text
receive at age .03 s -> FRESH
execute at age .10 s -> FRESH
execute at age .40 s -> AGING
execute at age .90 s -> STALE
older chunk -> OUT_OF_ORDER
```

The lifecycle test includes invalid receive/execute timestamps and does not
mutate the latest accepted chunk when such an event is rejected.

## Separation from actual-model seal

This native run proves orchestration with a qualified benchmark reference. It
does not promote the unadapted base GR00T output to WBC execution and does not
establish task performance.

Node-local evidence:

`/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage17/runtime_orchestration/orchestration.json`

and its `native_wbc/native_external_summary.json`.

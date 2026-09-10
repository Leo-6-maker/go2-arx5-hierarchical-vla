# Reusable manifest-driven inference engine

This document records the PI0108 execution boundary added beside the frozen
qualification launcher.  The historical launcher and its evidence remain
unchanged and are not imported by the generic engine.

## Runtime lifecycle

```text
concrete manifest + observed runtime config
              |
              v
verify all bound artifacts and observe checkout/runtime
              |
              v
cross-bind admission (before model import)
              |
              v
allowlisted versioned GR00T adapter
              |
              v
ordered raw [1,40,132] arrays + fsync + raw seal
              |
              v
existing generic CPU decode/metrics/evidence inventory
```

`scripts/run_h200_inference_backend.py` is the committed entrypoint.  It
bootstraps the repository root and `src` from its own absolute location, so an
absolute-path invocation is independent of the caller's CWD and
`PYTHONPATH`.  `--self-check` performs no model import or execution.

The runtime config is an observed document, not an admission copy.  The
engine re-hashes checkpoint, mapping, statistics, cohort, and noise artifacts;
re-observes the checkout HEAD, branch, host, Python runtime, device, offline
mode, and admission identity; and rejects any mismatch.  It also requires a
fresh output root and preserves the manifest's exact checkpoint/window order.

## Ownership

`h200_engine.py` owns admission, ordering, counters, raw durability, failure
seals, and the handoff to `run_evaluation()`.  The allowlist in
`h200_adapters.py` accepts only the exact versioned
`groot_n1d7_structured_action_v1` adapter.  The adapter owns model imports,
processor/data preparation, and the source-mapped action-head sampler.  Model
frameworks are imported only from the adapter's post-admission `prepare()`.

The generic engine never selects an arbitrary Python module from a manifest,
never infers a checkpoint or window count, and never sorts away explicit
manifest order.  It seals every raw array before starting CPU-derived output.
An induced adapter or decoder failure leaves a durable partial raw seal and a
failure seal; the engine has no retry path.

The accepted H40/K16/T16, StructuredActionV1, raw `[1,40,132]`, decoded
`[40,10]`, model-input, sampler, and timestamp contracts are consumed from the
existing reusable validators.  Validation-E, Holdout-F, WBC, simulator,
hardware, training, scheduler submission, and merge remain outside this gate.

## CPU/static proof surface

`tests/test_reusable_h200_general_engine.py` covers arbitrary checkpoint
labels/counts, arbitrary explicit window order, mandatory artifact and runtime
cross-binding, wrong HEAD/runtime fail-closed behavior, foreign-CWD launcher
bootstrap, exact adapter allowlisting, deferred model imports, raw-before-
derived durability, induced late failure, source-mapped adapter symbols, and
separation from the historical canary.  No GPU, scheduler, model forward, or
training operation is part of these tests.

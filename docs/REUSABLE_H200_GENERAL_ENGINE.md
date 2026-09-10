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

## Freeze-hardening boundary

The adapter handoff is a frozen `AdapterRuntimeBindingV1`.  It is the only
artifact/path authority visible to the allowlisted GR00T adapter: dataset,
GR00T source, Cosmos cache, modality configuration, processor/base checkpoint,
per-label checkpoint and statistics, noise bank, CUDA device, and language are
verified against the observed admission before adapter resolution.  The
runtime configuration's artifact observations are cross-bound to this single
binding; a legacy `adapter_config` namespace is rejected.

Checkout cleanliness includes untracked files.  The runtime probe records all
porcelain-reported untracked paths, and any such path fails admission before
adapter preparation.  Failure seals carry the phase and monotonic lifecycle
counters from the actual failure: admission, adapter resolution/preparation,
model import/construction, forward attempts, sampler calls, raw sealing, and
derived evaluation are not replaced with an unconditional `false`.

`stage45c_compatibility.py` is a static proof surface for the historical
Stage45C contract.  It shows that `FULL24@2000`, the exact 24-window TRAIN
cohort and explicit timestamp fields, seed 4242, four-step Euler sampling,
runtime-V2 statistics, and the H40/K16/T16 ABI fit the generic manifest
without executing a model or changing the historical canary.  The proof is
not a qualification result and does not reopen Validation-E, Holdout-F, WBC,
simulation, hardware, PBS, training, or merge.
The detached copy is committed as
`manifests/STAGE45C_REUSABLE_ENGINE_COMPATIBILITY_PROOF_V1.json`.

The PI0102 one-window qualification contract is kept separate in
`stage45c_one_window_fixture.py` and
`manifests/PI0102_ONE_WINDOW_QUALIFICATION_FIXTURE_V1.json`: it locks one
`FULL24@2000` checkpoint, window index 0, its exact window/target identity,
and four expected sampler calls.  It is a compatibility fixture only and is
not a qualification result.

## Deferred main-consolidation cleanup

The optional `processor_checkpoint` selection is still exposed through the
historical `scripts/stage26_h1_c1_fixed_noise_local_flow.py` helper in this
freeze commit.  This is a P2 packaging concern, not a runtime or scientific
contract exception: the generic adapter passes only its immutable binding into
that helper, and the helper's default Stage26 behavior remains unchanged.
Main consolidation should move the reusable processor builder into the
`src/go2arx5_vla` package and leave Stage26 as a thin historical wrapper.  The
move is intentionally deferred until after the reusable infrastructure is
frozen, so it is not mixed into this static correction gate.

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
separation from the historical canary.  It also covers immutable adapter
binding negative cases, untracked importable-code rejection, strict sampler
numeric types, truthful failure phases/counters, deterministic local-flow
equivalence, and the exact static Stage45C compatibility projection.  No GPU,
scheduler, model forward, or training operation is part of these tests.

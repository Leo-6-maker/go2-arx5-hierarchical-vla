# CPU acceleration preparation — PARTIAL, no speed claim

Authority: `POST_PI0276_ACCELERATION_AND_MULTISCENE_PREP_V1`, release `44aac6acce75c777f6777e5a5f4dc83a9d7f426b`, canonical #6011425206.

Actual PI0276 Trainer arguments: batch1, accumulation2, worker0, pin_memory=true. The pin-memory value was recovered by inspecting saved pickle opcodes without constructing a Trainer/model. Saved training interval:15144.5441s,0.541 updates/s; this includes Trainer bookkeeping and is not GPU compute time. Loader wait, DMA and GPU utilization were not saved.

The isolated adapter retains the original processor, input/label/mask hashes, visit metadata and TRAIN schedule. Corelab uses the actual source-card episode paths; Hopper's staged layout remains an explicit alternative. Each worker constructs its processor lazily. A separate loader generator controls spawn/prefetch; only the owner commits consumption after two microbatches. Optional NPY caches retain source/statistics/processor identities and never contain neural features.

Implemented W0/W2 and conditional W4 CPU benchmark: three passes/config,64 warmup plus2048 measured calls/pass; all prefetched/failed work shares32768 calls. Failed CPU passes retain time and attempt accounting. W4 requires correct W2,>=15% median throughput improvement, memory<32GiB and five legitimate CPU slots. No framework upgrade or live Trainer change.

Actual checks passed: import/AST isolation, mocked optimizer-boundary cursors0/4/128/4096/8191/end, discarded-prefetch accounting, source-bound nontrivial native Euler/root conversion, scene factors and collection refusal. These are **not actual processed TRAIN parity, worker timing or CUDA/optimizer resume**. CPU_ORDER_HASH_REPORT.json states that limitation.

Timing remains NOT_MEASURED_RESOURCE: the main read-only report/closeout and archive I/O have priority, and this preparation has no separate verified idle allocation. Local PC lacks a compatible qualified processor/data environment; its attempted pure-math test exposed an existing NumPy/SciPy ABI mismatch, so the successful static check used the existing isolated Corelab CPU environment. No packages were changed and no dataset was downloaded.

Recommendation: retain worker0 for production now. W2 is prepared, **not CPU_LOADER_CANDIDATE or TRAIN_RUNTIME_QUALIFIED**. Once a legal idle CPU allocation exists, run the bounded script on the already present TRAIN bank. Only then seek the separate24-update matched GPU qualification. Workers cannot remove the existing float(cuda_tensor) synchronization sites; propose measured detached-metric consolidation only after equivalence authorization. No asynchronous checkpoints, batch/LR/loss/precision change or new H200 job is included.

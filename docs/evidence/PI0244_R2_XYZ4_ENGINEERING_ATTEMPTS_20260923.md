# PI0244 R2/XYZ4 Engineering Attempts — 2026-09-23

## CPU regression attempt 1

- Source commit: `9fd52f9ba639edec7078730b0d58b396e7302ce6`
- Runtime: Hopper Stage10 Python 3.12.3, CUDA disabled
- Model weights loaded: 0
- GR00T model forwards: 0
- Optimizer/scheduler steps: 0
- Result: failed in the self-test, before any model load or scientific run.
- Cause: the weight-one parity check called the fixed XYZ×4 loss and compared it with the native unweighted loss.
- Observed assertion: expected `2.1787754513324087`, got `2.1290663833316708`; absolute delta `0.04970906800073793`.
- Repair: parameterize only the local loss helper's XYZ weight so the CPU control calls weight 1, while the sealed training invocation retains weight 4. No trainer recipe, data, model, precision, mask, or gate changed.
- Disposition: preserve this failed test record; rerun the exact-runtime CPU regression after the repair.

## HF archive identity check

- The sealed R0 card names `leoyuer/pi0234-stage-a-admitted-v1`; authenticated Hub `model_info` returned 404 for that repository.
- The preserved R0 receipts themselves contain mixed `durable_path` repository prefixes: checkpoint-2500 names `leoyuer/pi0234-stage-a-admitted-v1`, while 5000/7500/10000 name `leoyuer/pi0243-stage-a-admitted-v1`. The original receipts and R0 card are unchanged.
- The configured token identifies account `leoyuer`. Its existing `leoyuer/pi0243-stage-a-admitted-v1` repository is public and readable; the proposed `pi0244x-xyz4-m16-10k-v1/` prefix had zero files at the live check.
- R2 therefore binds its new family to that existing public repository and the unused R2-only prefix. This is storage routing only; the R0 paths remain recorded verbatim and no R0 checkpoint was rewritten or uploaded.

## Project_Storage capacity proof attempt 1

- Prepared card SHA256: `808f84bf05669fcb0154804c664aa800336790dd7eebb31a56fc5d838daa88bc`.
- The mature `posix_fallocate` proof requested `125348740672` bytes on the exact output fileset. It failed closed when one approximately 10 GiB segment reported only `106496` allocated bytes. The helper removed all temporary reservation segments; no checkpoint or scientific output was written.
- Kernel I/O accounting showed approximately `125348773888` bytes written before cleanup, so this is specifically an NFS allocation-accounting mismatch, not evidence that capacity admission passed.
- A same-fileset 128 MiB dense-random write/fsync/readback probe passed: file size `134217728`, allocated blocks `179560448`, matching SHA256 `8f6c6eaec27e7ca7b36eebc76e48808fc114e9805eeb2116a27a0952db10f014`; available bytes changed from `495034261504` to `494900043776` during the probe and returned to `495034261504` after cleanup.
- Disposition: R2 remains unsubmitted. Add a narrowly scoped dense-write/fsync/readback fallback to the mature reservation helper, regression-test it on this fileset, then create a distinct sealed R2 card/output root and repeat the full worst-case reservation.

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

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

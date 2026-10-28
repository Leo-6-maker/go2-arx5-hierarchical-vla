# Numerical Requalification Draft

Status: PREPARED_NOT_GPU_QUALIFIED. This does not authorize an additional PBS
submission or additional engineering updates. It does not claim the failed
eight-update comparison passed.

Actual initial parameter/configuration identities and the first 32 forward
losses were exact, but gradients differed and the fixed full-state tolerance
failed. Investigate CUDA backward nondeterminism without relaxing that gate.
The candidate configuration declares CUBLAS workspace :4096:8 before CUDA
initialization, PyTorch deterministic algorithms with unsupported operations
raising, deterministic cuDNN settings and math-only scaled dot-product
attention. BF16, TF32 declaration, masked flow loss, AdamW, effective batch 32,
learning rate, sample/RNG identity and comparison tolerances remain unchanged.

The new configuration requires a fresh actual continuous versus full-resume
comparison in separate preserved output directories. Existing failures remain
failures. Only eight engineering updates remain, so no extra microbatch
benchmark updates are assumed. Source publication and CPU flag fixtures do
not prove a deterministic CUDA backward or qualify main training.

Before consuming the final fourth allocation, the owner must establish a
legal combined requalification/main/evaluation arrangement, current qualified
data, complete restoration/capacity bounds and original deadline compliance.
Nothing in this draft resets counters, discards failed checkpoints, launches
a job, selects a candidate or modifies WBC.

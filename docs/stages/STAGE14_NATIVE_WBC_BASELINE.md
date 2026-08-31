# Stage14: native WBC baseline

## Scope

Stage14 consolidates project-owned integration code and evaluates the
delivered WBC only in its qualified native IsaacGym embodiment. It does not
run GR00T, P1M, cameras, writer, ROS, hardware, or training.

## Native reference modes

- `NATIVE_GENERATOR`: delivered runtime's own generator;
- `BENCHMARK_CONTINUOUS`: E1/E2 continuous reference injected at 50 Hz;
- `BENCHMARK_40POINT_RECONSTRUCTED`: the same reference sampled into a
  timestamped 40-point GR00T-like chunk and adapted back to WBC preview.

The first seven-trajectory qualification uses `BENCHMARK_CONTINUOUS`. A paired
40-point ablation is only valid after continuous injection is coherent. The
three errors remain separate:

```text
reference reconstruction error
controller tracking error
end-to-end task-space error
```

## Stop conditions

Stop before a broad benchmark on reference injection failure, observation or
history mismatch, non-finite tensors, unexplained resets/falls, or uncertain
checkpoint/config binding. Never interpret the P1M bridge numbers as this
baseline.

## Reproducibility

Each rollout binds the integration Git SHA, delivered archive SHA, both
checkpoint hashes, config SHA, trajectory SHA, simulator/runtime, reference
mode, seed, and timestamps. Run artifacts are node-local and ignored by Git.

# Stage14 reference-interface ablation

Status: `DEFERRED`

The canonical CPU adapter and the native launcher both expose the two intended
reference modes:

1. `BENCHMARK_CONTINUOUS` — direct timestamped reference into the native
   canonical trajectory representation;
2. `BENCHMARK_40POINT_RECONSTRUCTED` — deterministic 40-point timestamped
   reconstruction followed by the same native canonicalization.

The adapter's CPU tests cover SO(3) conversion, quaternion interpolation,
stationary and rotation-only paths, short horizons, high curvature, endpoint
clamping, and command ages of 0/20/50/100/200 ms. Stage13B also measured a
known sparse-interface deviation, especially for high-curvature/high-rotation
paths.

No native paired rollout was launched in Stage14. The continuous external
reference gate is not qualified because the benchmark-to-native frame and
anchor transform is still unspecified. Running the 40-point mode before that
contract is fixed would conflate discretization error with an unbound spatial
frame error.

Next required experiment after the blocker is resolved: identical native
initial state, seed, checkpoint pair, simulator, and trajectory under both
modes, reporting separately:

- reference reconstruction error;
- controller tracking error;
- end-to-end task-space error.

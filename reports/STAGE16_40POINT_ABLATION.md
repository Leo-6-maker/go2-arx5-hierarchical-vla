# Stage16 Continuous-vs-40-Point Interface Ablation

## Protocol

Eight matched seed-13 trajectories were run with identical trajectory, reset, checkpoint, simulator, asset, and native controller settings.

- Mode A: `ALIGNED_NATIVE_CONTINUOUS_REFERENCE`
- Mode B: `GR00T40_RECONSTRUCTED_REFERENCE`
- Mode B chain: continuous trajectory -> 40 timestamped EEF points -> `TrajectoryChunkV1` -> canonical adapter -> native nine-point preview -> WBC.
- Actual GR00T was not called (`actual_gr00t_called=false`). No camera or visual inference was used, and the GR00T horizon was not modified.

The comparison separately reports reconstruction error, native controller tracking error, and end-to-end task-space error.

## Aggregate result

Values are case-average summaries over the eight matched trajectories. For each paired penalty, positive means Mode B is worse than Mode A.

| Quantity | Continuous | 40-point | 40-point minus continuous |
|---|---:|---:|---:|
| controller position mean (m) | .032245729 | .032233453 | -.000012276 |
| controller rotation mean (rad) | .172182801 | .172224706 | +.000041904 |
| end-to-end position mean (m) | .041032083 | .041033157 | +.000001074 |
| end-to-end rotation mean (rad) | .172164215 | .172208735 | +.000044520 |
| controller position p95 (m) | .086867048 | .086726169 | -.000140879 |
| controller rotation p95 (rad) | .421326191 | .420890135 | -.000436056 |
| end-to-end position p95 (m) | .091259845 | .091263065 | +.000003220 |
| end-to-end rotation p95 (rad) | .421429234 | .420993015 | -.000436219 |

## Reconstruction component

Across cases, the 40-point reconstruction position mean was `0.00006188 m` and rotation mean was `0.00002022 rad`. The largest per-case reconstruction p95 was `0.00055401 m` and `0.00018890 rad`; the largest per-case absolute errors were `0.00085227 m` and `0.00020129 rad`. These are small relative to the native controller tracking errors.

The matched continuous and 40-point lifecycle was identical: seven reference completions and one boundary tracking abort in each mode. This does not mean a real GR00T interface has been validated; it means the reconstructed-interface penalty was not material in this controlled synthetic ablation.

## Dependence

At the eight-case descriptive level, reconstruction position error correlated with path length (`0.9944`) and translation-speed p95 (`0.9615`), while reconstruction rotation error correlated with angular-speed p95 (`0.9993`) and orientation demand (`0.9846`). End-to-end 40-point error showed no material monotonic dependence in this small sample (position versus angular speed `-0.1564`; rotation versus angular speed `-0.0594`).

Conclusion: `40-POINT ABLATION: RUN`; no measurable interface penalty was established here, but actual GR00T/visual inference remains outside Stage16.

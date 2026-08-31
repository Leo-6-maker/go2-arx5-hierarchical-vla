# Stage14 preliminary native WBC baseline

Status: `NOT QUALIFIED — REFERENCE_INJECTION_BLOCKER`

These are native IsaacGym diagnostic runs using the delivered arm and dog
checkpoints, native robot asset, native history/normalization, and
`BENCHMARK_CONTINUOUS`. They are not a generalization baseline because the
benchmark-to-native frame/anchor contract is still unresolved.

Common binding:

- archive SHA256:
  `b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167`;
- native robot URDF SHA256:
  `96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06462e0e49a8a2067e25`;
- arm checkpoint SHA256:
  `99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a`;
- dog checkpoint SHA256:
  `51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670`;
- control/physics: 50/200 Hz, decimation 4;
- integration code SHA bound in run manifests:
  `b9b58f857a21f6d5b7426ce43ca4007bc0e6e224`.

## Results

| trajectory | status | steps | position mean / p95 (m) | rotation mean / p95 (rad) | completion | base displacement (m) |
|---|---|---:|---:|---:|---:|---:|
| `E1_W0_00` | full | 100 | 0.0818 / 0.4429 | 0.5957 / 1.6248 | 1.000 | 0.6945 |
| `E1_W0_01` | full | 100 | 0.0832 / 0.4133 | 0.6284 / 1.6257 | 0.948 | 0.6683 |
| `E1_ang_0_00` | partial | 25 | 0.2663 / 0.5948 | 1.7545 / 2.3505 | 0.000 | 0.0328 |
| `E1_curvature_1_00` | full | 100 | 0.0802 / 0.3925 | 0.6330 / 1.6811 | 0.965 | 0.6911 |
| `E1_coupling_1_00` | full | 100 | 0.0865 / 0.4460 | 0.8697 / 2.1899 | 0.963 | 0.7157 |
| `E2_static_0_0` | full | 100 | 0.2450 / 1.0146 | 0.9944 / 2.1842 | 1.000 | 1.1828 |
| `E2_cont_0_0` | full | 100 | 0.0424 / 0.1250 | 0.5553 / 1.5838 | 0.960 | 0.9630 |

No success threshold is asserted. The orientation-only partial run ended after
the native timing termination condition (`timing_err` exceeded the configured
fraction of trajectory length); it did not end due to a NaN or process crash.

## Interpretation boundary

The E2 continuous candidate shows bounded position error in this preliminary
injection, while the E2 static candidate does not. That contrast is useful for
debugging, but cannot yet answer whether internal body coordination resolves
E2 redundancy because the absolute/anchor contract is not paired. The native
generator smoke itself was finite and naturally tore down, but ended after 25
steps under the same deterministic evaluation because of a timing-progress
termination. This is runtime evidence, not a tracking claim.

The P1M values from Stage13C remain separate and are not compared as WBC
performance:

- `93/100` arm-target clipping;
- position mean/p95 `0.527/0.864 m`;
- rotation mean/p95 `1.761/3.018 rad`;
- classification: `UNQUALIFIED_BRIDGE_TRACKING`.

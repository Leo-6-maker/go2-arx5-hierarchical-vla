# Stage15 aligned native diagnostic baseline

## Scope

These are native IsaacGym runs only, with the delivered arm and dog
checkpoints, native history/normalization/action path, seed 13, one environment,
headless mode, no camera/writer/GR00T, and `MODE_NATIVE_CONTINUOUS` references
created by the Stage15 binding. They are an aligned diagnostic set, not a
24--36 trajectory generalization benchmark.

## Results

| trajectory | native class | status | steps | position mean / p95 (m) | rotation mean / p95 (rad) | tolerance fraction | base displacement (m) |
|---|---|---|---:|---:|---:|---:|---:|
| `E1_W0_00` | ARM_EASY | natural | 100 | 0.0128 / 0.0266 | 0.1342 / 0.4518 | 0.80 | 0.2342 |
| `E1_W0_01` | ARM_EASY | natural | 100 | 0.0235 / 0.0452 | 0.1628 / 0.3792 | 0.74 | 0.1648 |
| `E1_ang_0_00` | ARM_EASY/orientation | natural | 100 | 0.0122 / 0.0364 | 0.1320 / 0.3753 | 0.87 | 0.1969 |
| `E1_curvature_1_00` | ARM_EASY | natural | 100 | 0.0218 / 0.0456 | 0.1553 / 0.3751 | 0.71 | 0.1646 |
| `E1_coupling_1_00` | ARM_EASY | natural | 100 | 0.0175 / 0.0364 | 0.1804 / 0.4582 | 0.74 | 0.2339 |
| `E2_static_0_0` | STATIC_BASE_RELOCATION | timing partial | 25 | 0.2063 / 0.5261 | 0.5766 / 0.9820 | 0.08 | 0.0328 |
| `E2_cont_0_0` | GEOMETRICALLY_INFEASIBLE | negative control | 100 | 0.0379 / 0.0891 | 0.6174 / 1.3843 | 0.00 | 0.5952 |

All completed runs produced finite controller/state tensors and natural
teardown. The static case terminated at 25 steps through the native timing
condition, with the first target approximately `0.545 m` from the achieved
TCP. It is not a valid static-relocation success result. The continuous case
was deliberately retained as an infeasible negative control after native
reclassification and is not a native continuous-base qualification result.

## Limit observation

The old P1M run had `93/100` arm target-limit clipping under the unqualified
target asset. Stage15 does not expose a native runtime clip counter; an offline
comparison of the 525 logged arm target vectors with native URDF bounds found
2 samples outside the raw x5_joint5 lower bound (both in
`E1_coupling_1_00`) and none in the other six runs. The native runtime itself
uses its configured action/drive path and no artificial limit enlargement was
introduced. This is not equivalent to the P1M clipping statistic.

## Interpretation

The five centered/easy or nominal references, including the orientation case,
track coherently in the aligned native setup. The single native static
placement needs a dedicated native-valid qualification treatment, and no
continuous-base cohort exists in the reused shapes. Consequently Stage15 does
not promote this seven-run set to a broad native General-WBC baseline.

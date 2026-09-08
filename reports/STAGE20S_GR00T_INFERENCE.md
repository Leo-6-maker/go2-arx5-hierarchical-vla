# Stage20S Actual GR00T Inference

## Runtime

- Hugging Face status: `AUTHENTICATED` as `leoyuer` in the Stage10 runtime;
- GR00T source commit: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`;
- base model index SHA256: `407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59`;
- inference runtime: Python 3.12.3, PyTorch 2.9.0+cu128, CUDA 12.8;
- GPU: RTX 5080, peak allocated memory about 6089.454 MiB;
- seeded inference latency: `0.462736 s`;
- seed: `2020`.

The authentication token was not exposed. Model weights and output arrays
remain node-local.

## Output contract

The custom embodiment forward path produced finite:

```text
raw universal action: [1,40,132]
decoded physical action: [1,40,10]
EEF channel: [40,9]
gripper channel: [40,1]
```

The seeded easy output manifest records raw hash
`0db6b495...` and decoded hash `6688f37a...`; the full hashes are kept in the
node-local inference manifest rather than duplicated into source code. The
actual output was passed to the Stage20S offline interface seal and then to
the gated native WBC simulation.

## Prediction audit

For the 10-episode overfit checkpoint on the easy fixture, decoded model EEF
versus future desired label was:

| Error | Mean | P95 | Max |
|---|---:|---:|---:|
| position (m) | 0.0184152 | 0.0384457 | 0.0417183 |
| rotation (rad) | 0 | 0 | 0 |

The fixture holds orientation constant, so zero rotation error is a property
of this test distribution and is not an orientation-generalization result.

A 100-step ladder checkpoint produced a finite output but was rejected by the
test-only safety gate because its maximum declared-timing speed was about
`2.3312 m/s` against the test-only `2.0 m/s` reject threshold. This is a
trajectory/gate observation, not evidence that the base model has a known
production action rate; production timing remains dataset-defined.

# PI0155 generation-path diagnosis

**Status: PARTIAL / BLOCKED after the first model query.** The accepted
PI0153 A packet was not rerun. CPU provenance/fixture supplementation passed,
the nested Cosmos loader was bound to the exact local snapshot, and both real
processor builders passed in isolated CPU processes. The first fixed N2
`seed2223@2000` model query then failed inside the unchanged four-step sampler
graph; PI0155 requires fail-closed stop after a model attempt exception.

## Bound inputs and runtime

- branch HEAD: `128980d7c95424f90211fac4391618e768ab9a06`
- checkpoint tree: `09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152`
- N2 stats: `0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb`
- source noise array: `5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd`
- host/device: `corelab-4080s`, `NVIDIA GeForce RTX 4080 SUPER`, UUID
  `GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad` (32,091 MiB free at admission)
- runtime: Python 3.12.3, Torch 2.9.0+cu128, CUDA 12.8
- Cosmos: revision `9ce19a195e423419c349abfc86fd07178b230561`, tree
  `1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9`,
  loaded from the sealed local snapshot with HF offline mode.

## Actual attempt accounting

| item | observed |
| --- | ---: |
| model constructions | 1 successful |
| query attempts | 1 |
| conditioning/model-backbone forward | 1 entered, 0 completed, 1 unknown |
| conditioning entered / completed / unknown | 1 / 0 / 1 |
| action-head entered / completed / unknown | 0 / 0 / 0 |
| model-side calls accounted / entered / completed / unknown | 1 / 1 / 0 / 1 |
| raw arrays saved | 0 (failure preceded sampler return) |
| retries or alternate devices | 0 |

Failure: `RuntimeError: Inference tensors cannot be saved for backward` while
the sampler's action-head graph consumed conditioning produced under
`torch.inference_mode()`. This is a runtime/lifecycle boundary failure, not a
task or target-progress result. The complete immutable failure seal is at
`/var/cache/users/liuyu-nfs/stage45f_pi0155_generation_probe_20260913/failure_seal.json`
with SHA256
`baae8ad299730ae15c2eb195ef255426bd387333feb2efa37305381d5f51afaa`.

## CPU evidence retained

The supplement classified TRAIN as
`REFERENCE_ORIGIN_DERIVED_NOT_PROVEN_MEASURED_EEF`, marked historical C2
actual `x0` as `UNKNOWN_NOT_RECORDED`, and recorded the existing expert/hold
full-frame references (expert relative path `0.12269180229491666 m`; hold
path `0 m`). Its node-local report hashes are indexed in
`reports/stage45f_pi0155_cpu_supplement.json`.

## Boundary and recommendation

No training, ICL, closed-loop/WBC/simulator/hardware, PBS, Validation-E, or
Holdout-F operation occurred. The single recommended next experiment is a
new PI-authorized mechanical recovery of this same-input comparison after
placing the accepted sampler call behind an explicit no-grad boundary (or
cloning only the inference tensors), with the same checkpoint, seed, noise,
processor inputs, and remaining budget. No scientific interpretation should
be made from this failed query.

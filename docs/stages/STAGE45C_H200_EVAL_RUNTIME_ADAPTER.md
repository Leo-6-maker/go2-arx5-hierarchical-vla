# Stage45C H200 inference runtime adapter

This canonical-branch artifact implements the Corelab continuation authorized by
PI_GATE_STATE_0091.  It is an execution adapter, not a new scientific
protocol.  The canonical Stage45C protocol remains
`manifests/STAGE45C_TRAIN24_EVALUATION_PROTOCOL_FROZEN_V1.json` with SHA256
`0f8c564f534ce3b29a5029cc927f93a03697181b4b1df0aa9e80566af4cd020f`.

The adapter binds four read-only checkpoint identities in this fixed order:

| label | role | tree identity | decode statistics |
| --- | --- | --- | --- |
| `BASE` | reference | `8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4` | BASE processor statistics `c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9` |
| `STRUCTURED@1000` | historical TINY1 coverage control | `30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137` | historical StructuredAction statistics `0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206` |
| `FULL24@1000` | primary coverage endpoint | `38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c` | runtime V2 statistics `3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337` (payload `752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75`) |
| `FULL24@2000` | secondary exposure endpoint | `2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5` | runtime V2 statistics `3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337` (payload `752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75`) |

The exact TRAIN24 dataset is bound to tree
`2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789`, split
lock `7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246`, and
24 windows in target-major/snapshot-major order.  The Golden Release V2
promotion identity is `a6f803e257ceeb0434027d8a35e32a7b8c66baf8226234bd6299cebf417446f9`;
its release payload identity is
`8306b76f327b260310a22a5d22551ebf299366f8fea4124bcbfbdab87e807666`.  The
checkpoint integrity/delta audit is
`a672e39b45e025f9a7681c1a1a45d807dd9be2a4641e54db60b036252bb59539`.

## External binding

`manifests/STAGE45C_H200_EVAL_RUNTIME_BINDING_TEMPLATE_V1.json` is deliberately
pathless.  It records every identity and the complete path-key set, with all
values `null`.  A Vanda runtime release may make a private copy and fill the
same keys; the adapter rejects missing, symlinked, mismatched, or pre-existing
execution paths.  No node-local path is part of the repository.  The historical
comparator is `RECOVERED_CANONICAL_READ_ONLY`, records recovery evidence
`c968f9e2d4fdc0f4975637e81b70e57716e55dec808c16f04958f7165cda01b9`, has no
committed materialization path, and cannot be substituted.

PI0091 keeps the P1 checkpoint tree identity separate from the PI0087 training-return identity.  The fields are `checkpoint_tree_sha256` and `training_return_digest_sha256`; the latter is verified under its own `stage45b/checkpoint-N` namespace law and is never compared directly to the canonical checkpoint tree digest.

The path keys are:

`base_checkpoint`, `structured_1000_checkpoint`, `full24_1000_checkpoint`,
`full24_2000_checkpoint`, `full24_dataset`, `snapshot_root`, `groot_source`,
`cosmos_root`, `modality_config`, `base_processor_config`,
`base_processor_statistics`, `structured_1000_statistics`,
`full24_runtime_statistics`, `golden_release_manifest`,
`groot_runtime_manifest`, `full24_manifest`, `checkpoint_integrity_audit`,
`runtime_admission`, and `output_root`.

## Runtime semantics

After the identity preflight, the adapter reuses the accepted GR00T input
construction: `body_rgb`, `wrist_rgb`, `state` and the fixed language
instruction.  RGB tensors are 96x96x3, StateV1 is length 10, and numeric target
XYZ is absent from model input.  The official public sampler is not called.
One CPU `torch.Generator` is seeded with 4242 and produces one float32
`[1,40,132]` noise tensor per TRAIN24 window.  The same 24 noise tensors are
reused for all four checkpoints.  Each window follows the source-mapped
four-step Euler graph at timestep IDs `[0,250,500,750]` and `dt=0.25`.

Every raw tensor is written and hashed before decoding.  The historical
comparator calls the existing `evaluate_raw_prediction` implementation with
the accepted historical statistics.  Runtime V2 uses a narrow compatibility
bridge that applies the identical Stage40 inverse-normalization, active-mask,
StructuredActionV1 K16/T16 materialization, and native row-packed conversion;
only the checkpoint-specific q01/q99 statistics source is late-bound.  The
BASE reference uses the Golden Release processor's immutable BASE statistics
decode path.  The
existing `aggregate_evaluation_rows` and Stage44R `comparison_metrics` are then
used unchanged.  The evidence retains raw file/array hashes, decoded hashes,
checkpoint identities, noise hashes, sampler step observations, and per-
checkpoint aggregates.

## Invocation boundary

`--audit-only` and `--plan-only` are login-node/static modes and report zero
model construction, model forward, GPU, training, optimizer, Validation-E,
Holdout-F, WBC, simulator, hardware, and scheduler submission exposure.
`--execute` is only for the separately authorized H200 envelope
`hpc/pbs/stage45c_train24_checkpoint_eval_h200.pbs`; it requires a clean
canonical final-execution checkout, a scheduler job ID, a visible H200, and a fresh output root.
It never writes checkpoints, performs backward/optimizer work, or submits a
job itself.  Any identity, shape, finiteness, or counter mismatch fails closed.

The preparation return token is
`CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_READY`.  The canonical branch
contains the history-preserving integration of the exact runtime-adapter
support commit; the frozen protocol remains unchanged.

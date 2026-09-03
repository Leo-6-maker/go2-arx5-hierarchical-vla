# Stage25 — BASE to O1@200 Parameter-Delta Protocol

Date: 2026-09-03 (Asia/Singapore)

## Status

```text
PREPARATION: PASS
DELTA_COMPARISON: PREPARED_NOT_EXECUTED
NEXT_GATE: HOPPER_CPU_STATIC_DELTA_EXECUTION
```

This protocol is the CPU/static gate required before authorizing another
H200 training run. It compares the exact Stage22 BASE checkpoint with the
exact Stage23 O1@200 checkpoint directly from safetensors. It never constructs
a model and never runs CUDA, forward, inference, training, PBS, WBC, or
hardware code.

The machine-readable preparation record is
`reports/STAGE25_O1_PARAMETER_DELTA_PROTOCOL.json`.

## Immutable O1 provenance

The Stage23 Phase-A relay was independently checked before the protocol was
written:

| item | value |
|---|---|
| Stage24 source commit | `c5ded6e119df920402451497aacdc69274ca8fc1` |
| Stage23 frozen source | `deb920a070141e6a32a88cd0da6719828647e1e6` |
| GR00T source | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| Phase-A archive SHA256 | `720aca9a1a1c7146919620df4000a76cc228a71729b057a2a0d155787ea50dcd` |
| payload manifest SHA256 | `dfeb6b44410b0042e53bac261e49acf7dbaa23ee2a61839ab65d4171b6c42b6a` |
| payload files verified | `252 / 252` |
| O1 historical checkpoint | `/scratch/e1192548/stage23_o1_20260903/runs/597966.hopper-m-02/candidate_200/checkpoint-200` |
| O1 index SHA256 | `ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7` |
| O1 checkpoint seal | `checkpoint_seal/checkpoint_file_seal.tsv` |
| O1 checkpoint seal SHA256 | `7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806` |
| O1 trainer state | `global_step=200`, SHA256 `2aa506bf605bce28809739970b6d3bc86059dba5490101c7da4fdba94512e4d9` |
| O1 run manifest | SHA256 `4dfb69bbd0bb36562c17046735001454c8dc5cfe5e9e0195ae3ca032259a3148` |

The Phase-A relay contains the O1 per-file seal, metadata, logs, and
trainer-state evidence, but not the two full O1 safetensor shards. Their
sealed entries are retained in the JSON record. This is an execution-input
gap, not a preparation failure; the compare command refuses to proceed until
the exact bytes and the seal are supplied.

## Frozen BASE identity

The compare command must reproduce both identities before reading any model
tensors:

```text
BASE index SHA256: 407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59
BASE tree SHA256:  8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
BASE tree prefix:  payload/GR00T-N1.7-3B
O1 index SHA256:   ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7
```

The O1 directory must match every `candidate_200/*` entry in the immutable
checkpoint seal, including file size and SHA256. The serialized key set,
shape, and stored dtype must match BASE exactly. No directory name or model
name is accepted as an identity substitute.

## Required category-specific comparison

The exact 14 serialized keys are frozen as:

```text
action_head.state_encoder.layer1.W
action_head.state_encoder.layer1.b
action_head.state_encoder.layer2.W
action_head.state_encoder.layer2.b
action_head.action_encoder.W1.W
action_head.action_encoder.W1.b
action_head.action_encoder.W2.W
action_head.action_encoder.W2.b
action_head.action_encoder.W3.W
action_head.action_encoder.W3.b
action_head.action_decoder.layer1.W
action_head.action_decoder.layer1.b
action_head.action_decoder.layer2.W
action_head.action_decoder.layer2.b
```

For each key, the harness reports the exact BASE/O1 shard, shape, dtype,
canonical stored-dtype tensor SHA256, absolute delta L2/RMS/max-abs, relative
delta L2, BASE/O1 cosine, and changed-element fraction using equality at the
stored dtype. It reports rows `4,5,6,7,10,11,24,25,26,27` separately:

```text
4–7:   deterministic unused/control comparison rows
10:    NEW_EMBODIMENT
11:    UNITREE_G1_SONIC
24–27: pretrained reference rows
```

For `action_head.action_encoder.W1.W`, every audited row additionally reports
the active physical columns `0:10` and inactive padded columns `10:132`,
including total and per-column-normalized metrics. The harness does not infer
that non-ID10 rows must remain byte-identical; any interpretation must respect
the pinned optimizer and weight-decay semantics.

## Shared-module grouping

Every serialized model key is assigned exactly one group using the frozen
Stage23 recipe and pinned N1.7 key prefixes:

| group | serialized prefix/rule | expected state |
|---|---|---|
| `CATEGORY_PROJECTORS` | `action_head.state_encoder.*`, `action_head.action_encoder.*`, `action_head.action_decoder.*`, `action_head.position_embedding.*` | trainable (`tune_projector=true`) |
| `DIFFUSION_ACTION_HEAD_SHARED_TRAINABLE` | `action_head.model.*` | trainable (`tune_diffusion_model=true`) |
| `VISUAL_LLM_VLLN_EXPECTED_FROZEN` | `backbone.*`, `action_head.vlln.*`, `action_head.vl_self_attention.*` | frozen (`tune_visual/llm/vlln=false`) |
| `OTHER_EXPECTED_FROZEN` | conservative fallback for any remaining serialized key | frozen until source-explained |

For every group the report contains key count, parameter count, changed-key
count, aggregate delta L2/RMS, aggregate BASE norm, relative delta, and the
maximum per-key relative delta. Any changed element in an expected-frozen
group is a fail-closed STOP condition pending a source-level explanation.

## Direct comparison algorithm

The executable command is:

```text
python scripts/stage25_o1_parameter_delta_audit.py audit \\
  --base-checkpoint <exact-base-tree> \\
  --o1-checkpoint <exact-o1-checkpoint-200> \\
  --o1-seal <immutable-checkpoint-file-seal.tsv> \\
  --output <external-report.json>
```

It opens safetensors with `framework="pt", device="cpu"`, streams tensor
chunks along the leading dimension, and keeps all arithmetic on CPU. Raw
tensor hashes use contiguous stored-dtype bytes; numerical deltas use float64
accumulators. The output is written outside both checkpoint trees and refuses
to overwrite an existing report.

## Fail-closed rules

The harness stops before tensor comparison if any of the following occurs:

- BASE index/tree identity mismatch;
- O1 index or immutable per-file seal mismatch;
- missing or extra O1 files;
- serialized key-set, shape, or dtype mismatch;
- any required category-specific key or category row is absent;
- non-CPU tensor is returned, a tensor is non-finite, or a safe path escapes;
- an expected-frozen group changes without a source-level explanation.

The harness does not modify either checkpoint and does not silently repair,
cast, clip, retime, or otherwise alter model data.

## Preparation boundary

```text
MODEL_CONSTRUCTED: false
CUDA_USED: false
FORWARD: false
INFERENCE: false
TRAINING: false
PBS_SUBMITTED: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
WBC: false
HARDWARE: false
```

Stage25 preparation ends here. The next authorized step is a separate
`HOPPER_CPU_STATIC_DELTA_EXECUTION` using the exact two checkpoint trees.

# Stage25 — BASE to O1@200 Parameter-Delta Protocol

Date: 2026-09-03 (Asia/Singapore)

## Status

```text
PREPARATION: PASS
DELTA_COMPARISON: PREPARED_NOT_EXECUTED
NEXT_GATE: GPT_PI_REVIEW_AFTER_STAGE25_SUPPLEMENTAL_HISTORY_BINDING_PATCH
```

This protocol is the CPU/static gate required before authorizing another
Hopper/H200 action. It compares the exact Stage22 BASE checkpoint with the
exact Stage23 O1@200 checkpoint directly from safetensors. It never constructs
a model and never runs CUDA, forward, inference, training, PBS, WBC, or
hardware code. Hopper remains HOLD pending GPT/PI review of this supplemental
history-binding patch.

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

The whole-seal digest is an independent frozen identity gate. Before opening
or reading any tensor, `audit` computes the supplied seal digest and requires
the following output fields to agree:

```json
{
  "expected_o1_seal_sha256": "7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806",
  "observed_o1_seal_sha256": "<computed>",
  "verified": true
}
```

An altered but internally parseable/self-consistent per-file seal is rejected
before tensor reads. The rejection is written as
`REJECT_O1_SEAL_SHA256_MISMATCH` and exits nonzero.

## Two-layer provenance scope (Stage25 correction)

The original Stage23 core seal is immutable and is not regenerated or edited:

```text
STAGE23_CORE_SEAL: IMMUTABLE_HISTORICAL
checkpoint_file_seal.tsv SHA256: 7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806
SEALED_CORE: exactly the candidate_200 rows from that seal
```

The O1 checkpoint root is partitioned before any safetensor read into the
sealed core and this exact supplemental metadata allow-list:

```text
experiment_cfg/conf.yaml
experiment_cfg/config.yaml
experiment_cfg/dataset_statistics.json
experiment_cfg/final_model_config.json
experiment_cfg/final_processor_config.json
```

The five files are classified as
`PRESENT_DAY_UNSEALED_METADATA`. They are required to be regular, non-symlink
files, and the count must be exactly five. Their deterministic manifest field
is `STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256`, computed as

```text
sha256(relative_path + "\0" + bytes + "\0" + file_sha256 + "\n")
over sorted relative_path
```

The Stage23 relay and repository records do contain exact historical hashes
for all five paths, through the immutable payload manifest and the
`candidate_200.run_manifest.json` artifact map:

| path | prior binding status | bytes | SHA256 |
|---|---|---:|---|
| `experiment_cfg/conf.yaml` | `EXACT_HISTORICAL_HASH_FOUND` | 5487 | `d183317ff91f530000956aa9ba67ba1e33bad3f1d50e2f95b8c6626f4df8da76` |
| `experiment_cfg/config.yaml` | `EXACT_HISTORICAL_HASH_FOUND` | 5456 | `6d2d799b3a37ecf45dcf65a3479ec82246669a65d673fd53ce92373d6690ba69` |
| `experiment_cfg/dataset_statistics.json` | `EXACT_HISTORICAL_HASH_FOUND` | 58966 | `28ef7a15fc7c540dd38733eddbf55ebdc3fa2f62c3bafa8b75bfff8bbee54450` |
| `experiment_cfg/final_model_config.json` | `EXACT_HISTORICAL_HASH_FOUND` | 1513 | `4b88c3dd5fc45e871687526df79c4e93e26c803d428d8de54ea3a3e39c6d5246` |
| `experiment_cfg/final_processor_config.json` | `EXACT_HISTORICAL_HASH_FOUND` | 2548004 | `84b890c59dee08ffe2fce36b4c9f95fe689d993e0d3f0cb80c8a4e8da3fee823` |

This historical evidence does not retroactively add the five files to the
Stage23 `checkpoint_file_seal.tsv`; they remain supplemental and are not used
for tensor identity. Tensor identity is only the immutable Stage23 core seal
plus the exact O1 index SHA256. Any regular file, symlink, directory escape,
or path outside the union of these sealed-core rows and the exact five paths
fails closed before tensor reads. A sixth `experiment_cfg` file is therefore
not accepted.

The historical binding is a separate required provenance layer. After the
exact five-path scope passes and before tensor loading, the verifier requires
for every supplemental path:

```text
present bytes  == historical bytes
present sha256 == historical sha256
historical status == EXACT_HISTORICAL_HASH_FOUND
```

The audit output reports
`SUPPLEMENTAL_METADATA_HISTORY_BINDING_STATUS`,
`SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS`,
`SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES`, and
`STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256`. Each path also reports
`historical_bytes`, `present_bytes`, `historical_sha256`, `present_sha256`,
`exact_match`, and `historical_evidence_sources`. A missing historical record,
non-exact historical status, byte-count mismatch, or SHA256 mismatch is a
fail-closed provenance STOP. The preparation record leaves the present-root
manifest value null because the full O1 checkpoint is not in the Stage23
relay; the value is computed during the bounded checkpoint audit.

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
checkpoint seal, including file size and SHA256, and must contain exactly the
five supplemental metadata paths above. The serialized key set, shape, and
stored dtype must match BASE exactly. No directory name or model name is
accepted as an identity substitute.

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
- O1 index or immutable sealed-core mismatch;
- missing sealed-core files, missing supplemental files, or any unsealed extra;
- supplemental metadata symlink, path traversal, or non-regular file;
- supplemental historical binding missing, non-exact, or size/SHA256 mismatch;
- serialized key-set, shape, or dtype mismatch;
- any required category-specific key or category row is absent;
- non-CPU tensor is returned, a tensor is non-finite, or a safe path escapes;
- an expected-frozen group changes without a source-level explanation.

The harness does not modify either checkpoint and does not silently repair,
cast, clip, retime, or otherwise alter model data.

An audit result with `status=STOP_UNEXPECTED_FROZEN_DELTA` exits nonzero;
`status=PASS` exits zero. Unknown audit statuses are fail-closed and also exit
nonzero.

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

Stage25 preparation and supplemental historical-binding closure end here. The
next gate is
`GPT_PI_REVIEW_AFTER_STAGE25_SUPPLEMENTAL_HISTORY_BINDING_PATCH`. Hopper,
H200, PBS, and numerical delta execution remain HOLD until a new explicit
gate.

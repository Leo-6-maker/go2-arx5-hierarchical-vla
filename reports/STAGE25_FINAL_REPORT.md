# Stage25 Final Report — BASE to O1@200 Parameter-Delta Gate Preparation

Date: 2026-09-03 (Asia/Singapore)

## A. Classification

```text
PASS WITH DEVIATION
```

The Corelab preparation gate passed. The deviation is explicit: the Phase-A
relay contains O1@200 metadata and immutable per-file seals but not the full
O1 safetensor shards, so the numerical BASE→O1 comparison is prepared and
fail-closed, not executed.

```text
PREPARATION: PASS
DELTA_COMPARISON: PREPARED_NOT_EXECUTED
NEXT_GATE: HOPPER_CPU_STATIC_DELTA_RESUME_ON_CORRECTED_HEAD
```

## B. Git and scope

```text
branch: stage25/o1-parameter-delta-gate
authoritative base: c5ded6e119df920402451497aacdc69274ca8fc1
GR00T source: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
```

Only the direct-safetensor audit harness, protocol, reports, ADR, handoff,
and CPU/static tests are in scope. No checkpoint, model shard, archive,
private runtime, credential, or lab WBC artifact is staged.

## C. O1@200 provenance recovered

```text
historical path: /scratch/e1192548/stage23_o1_20260903/runs/597966.hopper-m-02/candidate_200/checkpoint-200
O1 index SHA256: ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7
O1 checkpoint seal SHA256: 7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806
O1 per-file seal: available
O1 trainer global_step: 200
O1 trainer_state SHA256: 2aa506bf605bce28809739970b6d3bc86059dba5490101c7da4fdba94512e4d9
O1 run manifest SHA256: 4dfb69bbd0bb36562c17046735001454c8dc5cfe5e9e0195ae3ca032259a3148
full O1 bytes in Phase-A relay: false
```

The Phase-A archive SHA256 is
`720aca9a1a1c7146919620df4000a76cc228a71729b057a2a0d155787ea50dcd`; its
payload manifest SHA256 is
`dfeb6b44410b0042e53bac261e49acf7dbaa23ee2a61839ab65d4171b6c42b6a`, with
252/252 payload files verified.

## D. Two-layer provenance scope correction

The Stage23 core seal remains immutable and unchanged:

```text
STAGE23_CORE_SEAL: IMMUTABLE_HISTORICAL
checkpoint_file_seal.tsv SHA256: 7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806
SEALED_CORE: exactly candidate_200 rows from that seal
```

The O1 checkpoint-root verifier now admits exactly these five supplemental
paths, each as a regular non-symlink file:

```text
experiment_cfg/conf.yaml
experiment_cfg/config.yaml
experiment_cfg/dataset_statistics.json
experiment_cfg/final_model_config.json
experiment_cfg/final_processor_config.json
```

They are `PRESENT_DAY_UNSEALED_METADATA`, and are not used for tensor
identity. The verifier computes
`STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256` using the sorted
`relative_path`, byte-count, and file-SHA256 records. Any missing core file,
missing supplemental, sixth supplemental, arbitrary extra, symlink, or path
escape fails closed before tensor reads.

The immutable Stage23 relay and `candidate_200.run_manifest.json` contain
exact historical hashes for all five files, classified as
`EXACT_HISTORICAL_HASH_FOUND`. This evidence does not retroactively add them
to the Stage23 core seal. Their present O1-root manifest remains
`OBSERVED_ON_HOPPER_PENDING_REEXECUTION`, and the verifier now requires every
present `{bytes, sha256}` pair to equal its corresponding historical record
before tensor loading. Missing history, non-exact history status, or any
per-file mismatch is fail-closed.

The audit reports `SUPPLEMENTAL_METADATA_HISTORY_BINDING_STATUS`,
`SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS`,
`SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES`, and per-file historical/present
size, SHA256, exact-match, and evidence-source fields. The closure is
implemented without promoting supplemental metadata into the Stage23 core
seal or tensor identity.

No scientific metric, category ID, active/inactive partition, or H1/H2 logic
was changed by this correction.

## E. Harness and frozen schema

```text
DELTA_HARNESS_READY: true
MODULE_CLASSIFICATION_FROZEN: true
CATEGORY_PROJECTOR_KEYS: 14
CATEGORY_ROWS: 4,5,6,7,10,11,24,25,26,27
W1_PARTITION: active 0:10; inactive 10:132; per-column normalized
EXPECTED_FROZEN_GROUP_DELTA: STOP
```

The direct comparison reports exact key/shard/shape, BASE/O1 stored dtypes,
the dtype comparison rule, stored-dtype tensor hashes, delta norms, cosine,
changed-element fractions, category rows, W1 active/inactive partitions, and
aggregate deltas for every serialized model key grouped by the pinned Stage23
recipe. The serialized key count is reported both before and after the
explicit tied-alias reconciliation. The header-only audit found 537/1030
common keys with BASE `torch.bfloat16` and O1 `torch.float32`, all under
`action_head.*`; 493 common keys match and `backbone.*` has zero mismatches.
Only the three source-proven action-head scopes may use this exact pair;
numerical comparison for those keys decodes both tensors to float64 while raw
stored-dtype hashes remain strict.

The hardening gate now verifies the frozen whole-seal SHA256 before any tensor
read and emits `expected_o1_seal_sha256`, `observed_o1_seal_sha256`, and
`verified`. A `STOP_UNEXPECTED_FROZEN_DELTA` result exits nonzero; `PASS`
exits zero. Static tests cover an altered-but-self-consistent seal rejection,
the two exit-code cases, and the no-tensor-read rejection path.

## F.1 Serialized key-set reconciliation correction

Hopper reported 1031 BASE index keys versus 1030 exact O1 index keys. The only
BASE-only key was `backbone.model.lm_head.weight`; O1 retains the paired
`backbone.model.model.language_model.embed_tokens.weight`. The pinned
Transformers v4.57.3 Qwen3-VL save path identifies `lm_head.weight` as a tied
key and safe serialization stores shared storage once. The exact O1 index and
the pinned save path therefore provide a source/provenance explanation for
this one alias omission.

The verifier now admits only this exact BASE-only key, requires the paired key
in both indexes, requires the BASE mapping to
`model-00001-of-00002.safetensors`, and compares the paired key normally. Any
other missing/extra key or common-key shape mismatch still fails closed. The
only stored-dtype exception is the exact, source-proven action-head scope set
documented below; all other dtype mismatches fail closed. This is a mechanical
verifier correction; it changes no model bytes, scientific inputs, category
IDs, active/inactive partition, or experiment authorization.

## F.2 Pinned action-head storage-dtype provenance

Hopper's header-only audit found 537/1030 common keys with unchanged shapes,
BASE `torch.bfloat16`, and O1 `torch.float32`. All 537 are under
`action_head.*`: `action_head.model.transformer_blocks` (448),
`action_head.vl_self_attention.transformer_blocks` (64), decoder (4), encoder
(6), state encoder (4), timestep encoder (4), `proj_out` (4), position
embedding (1), and `vlln` (2). The other 493 common keys match in stored dtype;
the header audit found zero `backbone.*` mismatches.

The pinned Stage23 wrapper sends `load_bf16=true` only to the nested Qwen
backbone and does not pass a top-level Transformers `dtype`. The complete
action head's ordinary modules therefore start with default fp32 parameters;
the pinned Transformers 4.57.3 loader loads floating BASE tensors into those
existing parameter dtypes. The Trainer's `bf16=true` controls AMP autocasting,
not parameter storage, and `save_pretrained` serializes the current fp32
action-head storage. `backbone_trainable_params_fp32=false` affects only the
optional backbone trainable-parameter cast and does not explain or alter these
action-head keys. `tune_vlln=false` freezes `vlln` and `vl_self_attention`, but
freezing does not recast their storage; they are therefore legitimately part of
the representation-only difference.

The pinned source anchor is `isaac-gr00t`
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`: `gr00t/model/gr00t_n1d7/setup.py`
passes `load_bf16` to the nested backbone load without a top-level dtype;
`gr00t/model/gr00t_n1d7/gr00t_n1d7.py` constructs and freezes the action-head
submodules; `gr00t/model/modules/qwen3_backbone.py` scopes BF16 loading and the
optional FP32 cast to the backbone; `gr00t/model/modules/embodiment_conditioned_mlp.py`
has no dtype override for category-specific parameters; and
`gr00t/experiment/experiment.py` supplies Trainer BF16 AMP and saves the
current state dict.

The verifier now allows only these exact prefixes and dtype pair: projector
prefixes (`action_head.state_encoder.*`, `action_head.action_encoder.*`,
`action_head.action_decoder.*`, `action_head.position_embedding.*`),
`action_head.model.*`, and frozen `action_head.vlln.*` /
`action_head.vl_self_attention.*`. It still requires exact shape, finite values,
and raw stored-dtype tensor SHA256 values; it uses decoded float64 values for
deltas and changed-element equality only under these scopes. A mismatch outside
them, or a different pair, remains fail-closed. The correction is
semantic-preserving serialization handling, not a scientific-contract or
experiment change.

## F. Execution return

```yaml
ROLE: CODEX_CORELAB
STAGE: Stage25 provenance-scope correction
STAGE24_BASE_SHA: c5ded6e119df920402451497aacdc69274ca8fc1
O1_200_HISTORICAL_PATH: /scratch/e1192548/stage23_o1_20260903/runs/597966.hopper-m-02/candidate_200/checkpoint-200
O1_200_SEAL_SOURCE: checkpoint_seal/checkpoint_file_seal.tsv
O1_200_EXPECTED_INDEX_SHA256: ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7
O1_200_PER_FILE_SEAL_AVAILABLE: true
DELTA_HARNESS_READY: true
MODULE_CLASSIFICATION_FROZEN: true
CPU_STATIC_TESTS: compile and standard-library protocol self-checks
MODEL_CONSTRUCTED: false
FORWARD: false
INFERENCE: false
TRAINING: false
PBS_SUBMITTED: false
RESULT: PASS
SUPPLEMENTAL_METADATA_HISTORY_BINDING_IMPLEMENTED: true
SERIALIZED_KEY_SET_RECONCILIATION: PASS_EXPECTED_TIED_ALIAS_OMISSION
BASE_INDEX_KEY_COUNT: 1031
O1_INDEX_KEY_COUNT: 1030
ALLOWED_BASE_ONLY_KEY: backbone.model.lm_head.weight
REQUIRED_PAIRED_KEY: backbone.model.model.language_model.embed_tokens.weight
UNEXPECTED_MISSING_KEYS: []
UNEXPECTED_EXTRA_KEYS: []
SERIALIZED_DTYPE_CAST_RECONCILIATION: PASS_EXPLICIT_ACTION_HEAD_SCOPES
ALLOWED_DTYPE_CAST_SCOPES: CATEGORY_PROJECTORS, DIFFUSION_ACTION_HEAD_SHARED_TRAINABLE, ACTION_HEAD_VLLN_EXPECTED_FROZEN
OBSERVED_DTYPE_MISMATCH_COUNT: 537/1030 common keys
BASE_STORED_DTYPE: torch.bfloat16
O1_STORED_DTYPE: torch.float32
UNLISTED_DTYPE_MISMATCHES: FAIL_CLOSED
SCIENTIFIC_CONTRACT_CHANGED: false
DOWNSTREAM_AUTHORIZATION_STILL_VALID: true
NEXT_GATE_REQUEST: HOPPER_CPU_STATIC_DELTA_RESUME_ON_CORRECTED_HEAD
```

## G. Stop gate

Do not start H2/H1, do not submit PBS, do not use Hopper/H200 for a new
experiment, and do not run WBC. Hopper may resume only the same authorized
CPU/static BASE→O1 comparison against the corrected head. Its completed
result is the next GPT/PI gate; this correction itself makes no scientific
interpretation of BASE→O1 parameter movement.

See `reports/STAGE25_O1_PARAMETER_DELTA_PROTOCOL.md` and its JSON companion
for the complete executable schema.

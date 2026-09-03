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
NEXT_GATE: GPT_PI_REVIEW_AFTER_STAGE25_SUPPLEMENTAL_HISTORY_BINDING_PATCH
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

The direct comparison reports exact key/shard/shape/dtype, stored-dtype tensor
hashes, delta norms, cosine, changed-element fractions, category rows, W1
active/inactive partitions, and aggregate deltas for every serialized model
key grouped by the pinned Stage23 recipe.

The hardening gate now verifies the frozen whole-seal SHA256 before any tensor
read and emits `expected_o1_seal_sha256`, `observed_o1_seal_sha256`, and
`verified`. A `STOP_UNEXPECTED_FROZEN_DELTA` result exits nonzero; `PASS`
exits zero. Static tests cover an altered-but-self-consistent seal rejection,
the two exit-code cases, and the no-tensor-read rejection path.

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
NEXT_GATE_REQUEST: GPT_PI_REVIEW_AFTER_STAGE25_SUPPLEMENTAL_HISTORY_BINDING_PATCH
```

## G. Stop gate

Do not start H2/H1, do not submit PBS, do not use Hopper/H200, and do not run
WBC. The next decision gate is GPT/PI review of this Corelab supplemental
history-binding patch. No interpretation of BASE→O1 parameter movement is
made by this preparation report.

See `reports/STAGE25_O1_PARAMETER_DELTA_PROTOCOL.md` and its JSON companion
for the complete executable schema.

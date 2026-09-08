# Stage24 A0 checkpoint recovery (Historical Pre-Recovery Record)

> Superseded on 2026-09-03 by `reports/STAGE24_STATIC_CHECKPOINT_AUDIT.md`.
> This report records the earlier authorization blocker and remains useful as
> provenance history. It is not the current node-access or checkpoint status.

Date: 2026-09-03 (Asia/Singapore)

This is a read-only artifact-location and provenance recovery report. It does
not construct a model, read a tensor, run inference, submit PBS, or use a GPU.

## Required return

```text
ROLE: CODEX_RECOVERY_INVESTIGATOR
STAGE: Stage24 A0 fail-close recovery
GIT_SHA: 76033d94a26e88464760826606807ab47ec99b7b
EXECUTION_CONTEXT: corelab-amd-server; current Corelab login-node context
R0_LINEAGE_COMPLETE: true
STAGE17_HISTORICAL_MODEL_PATH: /var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B
STAGE17_INDEX_SHA256: 407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59
STAGE10_CORELAB_COPY_STATUS: DNS_AND_PORT_REACHABLE; SSH_AUTHORIZATION_BLOCKED; PATH_NOT_CHECKED
LOCAL_CACHE_SEARCH_STATUS: NO_VERIFIED_LOCAL_COPY; 5 INDEX_SHA_MISMATCH CANDIDATES
EXACT_BASE_SOURCE_SELECTED: NONE
EXACT_BASE_CONTENT_CLASSIFICATION: NO_VERIFIED_LOCAL_COPY
INDEX_SHA256_VERIFIED: false
TREE_SHA256_VERIFIED: false
SHARD_HASH_MANIFEST_STATUS: STAGE10_REFERENCED_BUT_NOT_AVAILABLE; STAGE22_BASE_SHARD_HASHES_NOT_PRESENT_IN_CURRENT_RELAY; PHASEA_CANDIDATE_ONLY
FULL_CHECKPOINT_RELAYED: false
MINIMAL_TENSOR_EXTRACTION_USED: false
EXTRACTED_TENSOR_KEYS: []
EXTRACTION_EVIDENCE_SHA256: null
ID10_CLASSIFICATION: ID10_INDETERMINATE_FROM_WEIGHTS
A1_STATIC_COUPLING_STATUS: NOT_RUN_EXACT_BASE_BYTES_UNAVAILABLE
ACTIVE_BLOCK_PER_COLUMN_RMS: NOT_RUN
INACTIVE_BLOCK_PER_COLUMN_RMS: NOT_RUN
PRETRAINED_CATEGORY_COMPARISON_STATUS: NOT_RUN_A0_BLOCKED
NEW_MODEL_FORWARD: false
NEW_INFERENCE: false
NEW_TRAINING: false
GPU_JOB: false
HOPPER_H200_USED: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
RESULT: BLOCKED
BLOCKER: CORELAB_5080_2_SSH_AUTHORIZATION_BLOCKS_STAGE10_PATH_CHECK_AND_NO_VERIFIED_LOCAL_COPY
NEXT_GATE_REQUEST: GPT_PI_REVIEW_AFTER_STAGE24_A0_RECOVERY
```

## R0 — lineage and seal availability

| lineage item | evidence | status |
|---|---|---|
| Stage17 Corelab source | `artifact_manifests/stage17_actual_groot_output.json`, `scripts/stage17_actual_groot_inference.py` | historically bound on `corelab-5080-2`; path is not proof of present bytes |
| Stage17 index | `407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59` | matches the later Stage22/23 base index identity |
| Stage22/Hopper source | `/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B` | historically bound; tree `8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4` |
| Stage23 O1 source | sealed campaign/run manifests | records the Stage22-bound path and the same index SHA |
| Stage10 shard manifest | `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/logs/checkpoint_sha256.txt` | referenced by the historical script, not present in the repository or current Corelab context |
| Stage22 handoff checksum list | `manifests/SHA256SUMS.sha256` under the historical handoff root | documented as historical handoff evidence; the current Corelab H2 audit relay does not contain the base payload |
| Stage22 current run manifests | `597351_r0_run_manifest.json`, `597351_r1_run_manifest.json` | record base shard sizes and `base_model_index_sha256`, but base shard SHA fields are null |
| Stage23 Phase-A checkpoint seal | `checkpoint_seal/checkpoint_file_seal.tsv` | candidate checkpoint files only; candidate index SHA is `ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7`, not the base |

The lineage is therefore complete as a historical binding, but no currently
available authoritative per-shard hash manifest was found for the Stage10
base bytes. No absent manifest was invented.

## R1 — Stage10 Corelab copy access classification

The current execution context is `corelab-amd-server`. The configured alias
`corelab-5080-2` resolves to `corelab-5080-2.lan` at `192.168.1.103`; TCP port
`65522` accepts a connection. A non-interactive SSH probe then returned:

```text
liuyu-nfs@corelab-5080-2.lan: Permission denied (publickey,password).
exit 255
```

This is classified as `AUTHORIZATION`, not route/DNS or port failure. The
remote Stage10 directory was not claimed missing because the authorization
failure prevents a read-only `stat`/inventory. No SSH key, authorized-keys
file, password, VPN setting, or remote permission was changed.

## R2 — bounded local/cache recovery

The locator used seven explicit, known roots and an index-first maximum depth
of 8. Existing roots were limited to the Stage22 metadata and Stage23 Phase-A
seal. Five index files were found, all under Phase-A candidate metadata. Every
one had:

```text
actual index SHA256:   ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7
expected base SHA256:  407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59
classification:        INDEX_SHA_MISMATCH
```

Because the index did not match, the locator did not parse those candidate
weight maps or hash any referenced shard. It returned the fail-closed result
`NO_VERIFIED_LOCAL_COPY` with exit code 1. The node-local JSON evidence is:

```text
/var/cache/users/liuyu-nfs/go2-arx5-stage24-recovery-20260903/locator_current_v2.json
SHA256 7ec82655ac9e59b30be32f2844ca277ccd85fb3165689cd5e22c874230d379
```

The tracked locator is intentionally bounded and refuses to classify an exact
copy from a directory name alone. It requires the exact index SHA, referenced
shard verification, and the frozen Stage22 model-tree SHA before returning
`EXACT_O1_BASE_BYTE_EQUIVALENT_LOCAL_COPY`.

## R3/R4 stop gate

The Stage10 path remains the preferred cheap recovery candidate, but it cannot
be inspected from this context without an already-authorized SSH path. No
credential or permission change is authorized. A large Hopper relay was not
requested, and no minimal tensor extraction was attempted because no exact
source bytes were available in an authorized read-only context.

Consequently A0 remains blocked and A1 is not run. In particular, the ID10
category rows and `action_encoder.W1.W[10, :, :]` are still unavailable, so no
10D-versus-122D static coupling claim can be made. A2-A7 remain explicitly
out of scope until A0/A1 are reviewed.

## Execution boundary

```text
new model forward: false
new inference: false
new training: false
GPU/PBS job: false
Hopper H200 used: false
validation e read: false
holdout f read: false
WBC/ROS/ZMQ/P1M/hardware: false
checkpoint/tensor binaries committed: false
```

The exact next recovery action is either an already-authorized read-only path
to `corelab-5080-2` or a cryptographically bound minimal tensor bundle from
the authoritative exact base. Until one exists, stop for GPT/PI review.

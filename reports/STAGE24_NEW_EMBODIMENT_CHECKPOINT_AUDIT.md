# Stage24-A0 NEW_EMBODIMENT Checkpoint Audit (Historical Pre-Recovery Record)

> Superseded on 2026-09-03 by `reports/STAGE24_STATIC_CHECKPOINT_AUDIT.md`.
> The earlier report correctly stopped when Corelab SSH access and exact
> checkpoint bytes were unavailable. Access is now restored and the exact
> checkpoint has passed the A0 identity gate; this file is retained as the
> fail-closed history, not as the current Stage24 status.

Date: 2026-09-03 (Asia/Singapore)

## Result

```text
ROLE: CODEX_CORELAB
STAGE: Stage24 custom-embodiment adaptation audit and Hopper-readiness gate
STAGE23_CLOSEOUT_COMMIT: a72f539f174f67f443b05b331f01a21544efd015
STAGE24_BRANCH: stage24/new-embodiment-adaptation-audit
GIT_SHA: a72f539f174f67f443b05b331f01a21544efd015
ISAAC_GROOT_SHA: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
GROOT_WBC_SHA: a0732b642c0333077e127a2f56ab0014c196bca4
EXACT_O1_BASE_CHECKPOINT_BOUND: false
BASE_MODEL_INDEX_SHA256: 407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59
BASE_MODEL_TREE_SHA256: 8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
ID10_CLASSIFICATION: ID10_INDETERMINATE_FROM_WEIGHTS
ID10_COMPARISON_SUMMARY: not run; exact base safetensor bytes are unavailable on Corelab
ID10_INACTIVE_CHANNEL_COUPLING: NOT_IDENTIFIABLE_FROM_STATIC_WEIGHTS
PRETRAINED_EEF_CATEGORY_COMPARISON: NOT_RUN_A0_STOP
TEMPORAL_PREFIX_CLASSIFICATION: NOT_RUN_A0_STOP
H8_SUMMARY: NOT_RUN
H16_SUMMARY: NOT_RUN
H24_SUMMARY: NOT_RUN
H40_SUMMARY: NOT_RUN
NORMALIZATION_COMPATIBILITY_CLASSIFICATION: NOT_RUN_A0_STOP
PHYSICAL10D_DEGENERATE_STATS_SUMMARY: NOT_RUN_A0_STOP
PRETRAINED_RELATIVE_EEF_SCALE_COMPARISON: NOT_RUN_A0_STOP
O1_EXACT_EXPOSURE: bound from sealed manifest; B1 x N200, 200 optimizer-example presentations, 200 noise/time draws
OFFICIAL_REFERENCE_EXPOSURES: Stage23 pinned source audit; SO100 B32 x N2000 nominally 64000 presentations
DENSE_H40_RELATIVE_EEF_UPSTREAM_PRECEDENT: true
NEXT_HOPPER_EXPERIMENT: NONE
NEXT_HOPPER_RATIONALE: A0 fail-closed blocker prevents a scientifically valid A6 decision
FUTURE_HOPPER_PROTOCOL_READY: false
C1_EXECUTED: false
NEW_MODEL_FORWARD: false
NEW_TRAINING: false
HOPPER_PBS: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
WBC_EXECUTED: false
HARDWARE_EXECUTED: false
RESULT: BLOCKED
BLOCKER: EXACT_O1_BASE_CHECKPOINT_SAFETENSOR_BYTES_UNAVAILABLE_FOR_READ_ONLY_VERIFICATION
NEXT_GATE_REQUEST: AUTHORIZED_EXACT_BASE_CHECKPOINT_RELAY_OR_GPT_PI_REVIEW_OF_A0_BLOCKER
```

## What is bound

The sealed Stage23 campaign and Stage22 handoff independently bind the O1
base model identity to:

```text
model path:
/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B

model.safetensors.index.json SHA256:
407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59

GR00T-N1.7-3B tree SHA256:
8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4

GR00T source:
51d4c89f72fda44cbf77285c6a8114b52676b8a1
```

The Phase-A seal records `full_checkpoint_weights_transferred=false` and
states that full model safetensors were intentionally not copied. The Stage22
relay likewise records `checkpoint_weights_included=false`. The candidate
checkpoint metadata present in the seal has index SHA
`ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7`, which
is an output-checkpoint metadata artifact and is not substituted for the
required base index.

## Corelab availability checks

On the current Corelab host (`corelab-amd-server`), the sealed scratch path
and the local node-cache candidates were checked read-only. The exact base
directory and its `model.safetensors.index.json`/shards were not present.
The only matching index files found locally were under the Phase-A
`checkpoint_*_metadata` directories; no base model shard bytes were present.

The already-known Corelab GPU aliases were checked without starting a
workload: `gpu5080-1` and `gpu5080-2` rejected the current public-key
authorization, while `gpu4080s` timed out on its configured port. No key,
credential, remote authorization file, or Hopper path was inspected or
modified.

## Stop gate

Without the exact base shard bytes, the required read-only tensor audit cannot
be performed safely:

- exact category-specific state/action encoder and decoder keys cannot be
  loaded from the exact O1 base;
- ID10 cannot be compared against IDs 24/25/26/27/11 or unused-looking rows;
- active columns `0:10` versus inactive columns `10:132` cannot be measured;
- no weight-derived classification or coupling claim is made.

Therefore Stage24-A stops at A0. A1-A7 are `NOT RUN` under the directive's
fail-closed rule. Existing Stage23 sealed predictions, normalization files,
and pinned source evidence remain untouched; no validation `e`, holdout `f`,
model forward, inference, training, PBS submission, WBC, ROS/ZMQ, P1M, or
hardware action was performed in Stage24.

The exact checkpoint must be relayed into the Corelab audit environment, with
the sealed index/tree/file hashes reverified, before A0 can be resumed. No
newer HF checkpoint or generic model path is an acceptable substitute.

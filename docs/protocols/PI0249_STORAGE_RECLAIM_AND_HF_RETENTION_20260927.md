# PI0249 Storage Reclaim — aggressive scratch retirement with pinned-HF preservation

Date: 2026-09-27 (Asia/Singapore)

Authorization: `PI0249_STORAGE_RECLAIM_GO`
Decision: `GO_AGGRESSIVE_SCRATCH_CHECKPOINT_RETIREMENT_WITH_PINNED_HF_PRESERVATION`
Status: `AUTHORIZED_NOT_EXECUTED_BY_GPT`

This storage authorization is additive to PI0249 scientific authorization `PI0249_R_ANCHORED_XYZ_PRIORITY_GO`. It changes storage/retention only. It does not change the PI0249 loss, data, seed/RNG design, optimizer, screen matrix, gate, WBC, or scientific budget.

## 1. Objective

Scratch pressure is now treated as an engineering blocker. The execution agent is authorized to reclaim scratch aggressively by deleting obsolete checkpoint payloads, including old failed-training checkpoints, while preserving only checkpoints that still matter to the active scientific lineage or reproducibility.

The rule is:

1. Preserve scientific facts and provenance.
2. Preserve current useful model states either locally or as pinned full Trainer checkpoints on HF.
3. Delete duplicate, superseded, engineering-invalid, or scientifically exhausted local checkpoint payloads.
4. Do not keep ~20.9 GB local checkpoints merely because a historical run once existed when the same complete payload has already passed fixed-revision HF readback.
5. Do not retrain to recreate something deleted if a pinned durable source already exists.

Target after cleanup:
- preferably >=100 GiB scratch free;
- and, when a quota command is available, >= max(100 GiB, 3 x current full-checkpoint bytes) user quota headroom.
`df` alone is not sufficient evidence of quota headroom; report filesystem free space and quota separately when possible.

## 2. MUST PRESERVE as useful model lineage

These are scientifically useful checkpoints and must remain recoverable. A local scratch copy is not required if the full Trainer state is already on pinned HF with exact inventory/readback.

### A. Current PI0249 initializer — highest priority
PI0248 R-final2000:
- HF repo: `leoyuer/pi0246-gate-pass-campaign`
- revision: `d089c09380c568e720c3dfa6bec5aa064acd7c4f`
- full tree SHA256: `4422a24d2117206681c423a0f8d3137dff7eb614c990cb0334a8cdaf73900298`

This is the fixed initializer for both new PI0249 K/X arms. Do not delete the only recoverable copy. If the scratch copy is retired, first verify the pinned HF full Trainer inventory/readback and record a retirement receipt. It may then be re-materialized immediately before K/X execution.

### B. Best completed fixed16 milestone
PI0247 U-final2000:
- revision: `0296f5dca5cb46daf05b51f90c53e1df905a7203`
- tree SHA256: `dace0dd423c4c1c6297b26bd6da09aee1e99edf4238638c5a317356e417a920f`

This is the first model that passed the registered fixed16 B0/B1 joint gates. It remains an important scientific milestone even though support expansion later failed. Preserve on HF; local scratch payload may be deleted after fixed-revision verification.

### C. PI0247 M64 parent used by PI0248
PI0247 M64-final5000:
- revision: `093c450fd49052c8fd3285bf38111a30a1f59009`
- tree SHA256: `1811bfa5c29fb7ce2e911ade0a0b476607e32ac1c1322cce68b8dff04449d776`

Preserve on HF because it is the common ancestor of PI0248 C/R and an important support-expansion failure point. Local scratch payload may be deleted after readback verification.

### D. New PI0249 K/X checkpoints
While an arm is active, preserve the newest resumable full Trainer checkpoint locally until the next full checkpoint is durably archived/read back. After durable verification, rolling retirement is explicitly authorized.

For final K2000/X2000:
- if a final checkpoint is scientifically valid, upload complete Trainer state and perform pinned-revision full-file readback before removing the last local full-state copy;
- if the run is engineering-invalid before a valid scientific result and is not a permitted exact-resume source, no HF upload is required merely to preserve bad weights.

## 3. KEEP ON HF, DELETE LOCAL SCRATCH COPIES when verified

The following historical checkpoints are useful as reproducibility references but do not need to occupy scratch if their fixed HF revision/full inventory is already verified:

- PI0248 C1000/C2000 and R1000/R2000;
- PI0247 M64 checkpoint1000/2500/5000;
- PI0247 L/U checkpoint1000/2000;
- PI0246 C/T/P/PT complete checkpoints;
- PI0244 R0/R2 final scientific checkpoints;
- PI0243 Stage-A complete checkpoints;
- any other canonical scientific checkpoint whose durable receipt + fixed-revision full readback already prove exact recoverability.

Do not duplicate-upload an already pinned, fully read-back checkpoint. Instead record that the HF copy is the retained authoritative payload and retire the local duplicate.

## 4. DIRECT DELETE — no HF upload required

The execution agent is explicitly authorized to delete the following from scratch without uploading the checkpoint payload, after first writing a small deletion inventory:

1. partial/incomplete checkpoints from engineering failures;
2. checkpoints from runs that failed before a valid scientific model result and are not a sealed resume source;
3. duplicate restored copies, inference materializations, HF download caches, tmpfs/staging mirrors and redundant model views whose authoritative source is pinned elsewhere;
4. superseded engineering attempts from the same scientific run after a later first-valid attempt is complete;
5. smoke/debug checkpoints such as old one-step or probe-only weights;
6. corrupted, truncated, serialization-failed or quota-failed checkpoint directories that cannot represent a valid Trainer state;
7. local copies of valid scientific FAIL checkpoints when a complete pinned-HF copy already exists;
8. duplicate checkpoint payloads left behind after successful rolling archival.

For every direct-delete item, the inventory needs only:
- path;
- approximate bytes;
- run/job/arm identity if known;
- class: `ENGINEERING_INVALID`, `DUPLICATE_DURABLE`, `SUPERSEDED_ATTEMPT`, `PARTIAL_CHECKPOINT`, or `SCIENTIFIC_FAIL_DURABLE_ELSEWHERE`;
- reason it is not the unique source of an active or future authorized resume.

A complete SHA of a known-corrupt multi-GB tree is not required solely to justify deletion if the failure provenance already establishes it is not a valid checkpoint.

## 5. DO NOT DELETE

Do not delete the following merely to make space:

- current PI0249 cards, seals, execution code, schedule, RNG-key table, cohort, TRAIN data, processor/statistics or model-input packets;
- small scientific evidence: run manifests, durable receipts, HF readbacks, logs, screen reports, CPU replay reports, source/noise manifests, update audits and closure files;
- unique NPZ/tensor evidence that has not been durably published elsewhere and is required to reproduce an existing published metric;
- the only remaining complete Trainer state of R2000 or of an active K/X exact-resume checkpoint;
- active job directories or files currently being written;
- the U-M16 qualified milestone unless its full pinned HF copy has been re-verified;
- any FULL/CALIB/DEV/F/WBC artifacts not explicitly classified by this document.

This authorization is checkpoint-retirement authority, not authority to delete datasets, raw expert data, source code, scientific ledgers or unique evaluation evidence.

## 6. Recommended cleanup order

Use this order until the target free space is reached:

1. incomplete/engineering-failure checkpoint payloads and tmp/cache/model-view duplicates;
2. local PI0243/PI0244/PI0246 scientific-FAIL checkpoints already durable on HF;
3. local PI0247 L and obsolete intermediate checkpoints already durable on HF;
4. local PI0248 C and R1000 copies already durable on HF;
5. local PI0247 M64 1000/2500 copies already durable on HF;
6. only if still needed, local PI0248 R2000 or PI0247 M64-final5000 copies after immediate pinned-HF re-verification and explicit retirement receipt.

Prefer retaining a locally staged R2000 initializer if enough space remains for PI0249. If not, it is authorized to retire and later re-materialize from its pinned revision.

## 7. Upload policy before deletion

Upload first only when all are true:
- the checkpoint is scientifically useful or an active lineage ancestor;
- no complete pinned HF copy with fixed-revision readback already exists;
- the local copy is currently the only complete recoverable Trainer state.

For such a checkpoint:
1. inventory full Trainer files;
2. upload to an isolated HF prefix;
3. obtain fixed revision;
4. read back every file and compare bytes/hash/tree;
5. write durable receipt;
6. only then remove the local payload.

Do not upload engineering-invalid or partial checkpoints simply to avoid deleting them.

## 8. Active PI0249 rolling policy

PI0249 K/X remains sequential. Storage may be managed aggressively:
- before K: capacity proof;
- after K1000 durable readback: K1000 local may be retired once no longer needed for an imminent exact resume;
- after K2000 durable readback and training closure: retire K local payloads except a minimal inference view if screening is imminent;
- repeat independently for X;
- never keep K and X full-state local duplicates solely for convenience when pinned HF is authoritative.

If an active arm fails after some optimizer updates but before a durable resume point:
- preserve logs/audits;
- delete unusable partial checkpoint payloads;
- actual consumed updates still count toward the PI0249 raw/update budget.

## 9. Required closure

Publish one additive storage-cleanup manifest to Issue #11 containing:
- before/after scratch free bytes;
- quota headroom if available, otherwise `UNKNOWN`;
- each deleted path, class and bytes reclaimed;
- each preserved useful checkpoint and its HF revision/tree/readback identity;
- any local checkpoint intentionally retained and why;
- confirmation no TRAIN data, processor/statistics, unique evidence, active-job files or current scientific cards were deleted.

Expected terminal status:
`PI0249_STORAGE_RECLAIM_PASS — CONTINUE_PI0249`

If the target cannot be reached without deleting a unique useful checkpoint lacking durable HF protection:
`PI0249_STORAGE_RECLAIM_BLOCKED_UNPROTECTED_USEFUL_CHECKPOINT`
and stop deletion at that boundary.

No new science, model queries, optimizer updates, WBC/sim, CALIB, DEV/F or PR merge are authorized by this storage document.

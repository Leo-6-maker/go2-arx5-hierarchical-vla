# Scratch root tidying and retired checkpoint deletion, 2026-10-03

Direct user storage-only authorization: tidy `/home/svu/e1192548/scratchdir` into `Projects/go2-arx5`, deleting expired checkpoints. The logical root resolves to `/scratch/e1192548`. The actual directories were already under the project tree; the clutter was compatibility symlinks.

Removed **72 explicit root compatibility links**, preserving all their targets. Scratch now contains `Projects`, five temporary active-training aliases and the shell's `Cache` alias. The five training aliases are `pi0227_h200_prep_20260919`, `pi0244_fixed16_10k_v1`, `pi0247_generation_to_sim_20260926`, `stage22_h200_cosmos_v2_20260902`, and `stage22_h200_runtime_deb920_20260902`. All **161 frozen execution files** remained readable and hash-identical after this operation. Alias removal freed no model payload bytes. No directory was moved again.

Deleted **15 large files in five explicitly retired checkpoint trees**: PI0263 F1000/F2000/X1000/X2000 and superseded PI0264 L1024. F/X's completed, valid quality FAILs and all existing full-Trainer HF readbacks were inspected. L2048 remains the protected ancestor; L1024 has a full HF readback and no active use. Each local tree's trainer state/config/index was matched to the pinned receipt, and payload sizes were matched before deletion. The existing deletion helper verified paths, ownership, active references, frozen sources, inode identity and complete hardlink coverage. All 15 payloads were verified absent, with five explicit metadata-only retirement markers.

| This batch | GiB |
|---|---:|
| Logical payload deleted |97.240510|
| Unique allocated bytes unlinked |97.240543|
| Available before execution |317.458984|
| Available after execution |414.700195|
| Measured available-space change |97.241211|

The earlier preview observed 336.922 GiB free. UNIFORM4096 was saved between preview and deletion, consuming about19.45 GiB; the execution's before/after measurements are reported separately. Exact current quota headroom remains unverified; capacity is not quota proof.

Existing HF revisions, Git, datasets, processors/statistics and small failed-experiment evidence remain. Current PI0269 four official checkpoints, ROT2048, L2048, P2000, N/R final secondary dependencies, foundation weights and baseline models are protected. Other large model trees with unresolved retirement or baseline use are explicitly listed in `CLOSEOUT.json`; they were not deleted because of age. Invalid/mock model indexes were recorded and skipped. This is an explicit completed batch, not a claim that every historical model is retired.

Training stayed healthy on PBS644116.hopper-m-02. UNIFORM reached4096 with18-file complete2048/4096 Trainers; PREFIX actually started and reached8 at the closeout snapshot. UNIFORM2048 HF readback is PASS. UNIFORM4096 foreground archival PID15642 is running separately on login-node CPU; no early quality query, runtime mutation or training resubmission occurred.

selected/deployment remain NONE. The mainline continues both4096 endpoints, four full inventories/readbacks and the frozen complete evaluation. Active compatibility aliases remain until their consumers finish; the shell Cache alias is an explicit remaining configuration dependency.

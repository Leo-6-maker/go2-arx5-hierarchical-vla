# Corelab Phase-2 storage organization and safe retirement — 2026-10-13

Status: BOUNDED_USER_AUTHORIZED_MAINTENANCE. Existing local-PC Codex is sole owner. Not a new scientific PI/GR00T/WBC/Hopper job. Do not interrupt ongoing PI0290, archive jobs, other users, or private WBC.

## Source boundary

Before action, on the local PC read all four original audit files in its work/corelab-storage-audit-20261013 folder: AUDIT_REPORT.md, FULL_STORAGE_INVENTORY.csv, RETENTION_PLAN.json, README_STORAGE.md and the prepared path checker. SHA256-pin all originals. These are **not mounted in the ChatGPT runtime**; do not derive exact deletion targets from this protocol. Without readable and internally consistent source data, mark BLOCKED_AUDIT_SOURCE and delete 0.

Facts supplied for reconciliation: AMD and 4080S share NFS physical regular files; 4080S SSD is local, so AMD PI0256/PI0284/PI0290 softlinks can appear broken without loss of SSD targets. Inventory 448 entries and ~1.81m files; 24 migrate candidates, 9 cache candidates; only seven small files already checked by full hash. PI0276 SSD ~337.6GiB and PI0209 home ~53.9GiB are sizes, not cleanup eligibility. Two paused archive processes depend on PI0250/PI0276. One private WBC path lacks permissions, static reference scan is bounded, and personal quota is unknown. Preserve all these caveats.

Separate earlier verified cleanup: reports/maintenance/2026-10-13-corelab-storage/ACTUAL_RECEIPT.json retired 7 derived Corelab BF16 model-only views, 14 shards, freeing 44,017,266,688 allocated bytes (~40.99 GiB); full Trainers, old failures, WBC and original episodes were preserved. Do not recount or repeat this retirement. Current PI0290 independent-tool pilot has 8/8 original successes (4/4 held-out) and original RGB/physics/episode artifacts to protect.

## Canonical path roles — configure from actual audit, not assumed absolute mountpoints

- Shared NFS: immutable historical episodes and numeric evidence, source manifests, human navigation/index, published fixed archival receipts. Same NFS inode observed from two nodes counts once.
- 4080S SSD: PI0290 active physics/runtime and raw episodes, working render/cache and explicitly labelled disposable inference-only derivatives. Document that these paths are not addressable from AMD merely by a shared-NFS alias.
- Home: small repository/config/documentation. PI0209 ~53.9GiB is UNKNOWN pending role and durable-copy proof.
- Private WBC assets: read-only, permission-limited, protected from migration/pruning.
- HF: durable remote models/datasets only if the exact repo_type, full revision, prefix, original per-file SHA, complete Trainer/data inventory and visibility/permission are verified. Hub authentication alone is not a backup.
- Staging/quarantine: finite owner-scoped directory, preserve metadata and provenance; a rename does NOT reclaim disk space.

Produce per-host PATH_ROLE_MAP with realpath, st_dev, st_ino, link count, logical/allocated bytes, correct consumer path. Do not blindly replace broken AMD symlinks or duplicate SSD data onto NFS. Prefer node-specific path resolution/manifest over misleading common symlink. No active path mutation while PI0290 runs.

## Five conservative classes

1. IMMUTABLE_PROTECT: entire PI0250/PI0276 dependency trees until stopped processes and references are explicitly resolved; active PI0290 raw PNG/RGB/physics/videos/manifests, private WBC, currently used baselines, sole Trainer/optimizer/scheduler/RNG/processor/stats, original task data and negative scientific evidence. No move/delete.
2. MIGRATE_COPY_ONLY: only the 24 original retention-plan candidates, excluding any protected descendants. Begin with seven previously hashed small files; validate full checksum on source AND destination, count, metadata, rights and path resolution on applicable host(s). **Do not remove source in this phase.**
3. CACHE_DELETE_ELIGIBLE: only original nine cache candidates explicitly identified as rebuildable/disposable by RETENTION_PLAN, if not in use and not unique. Before each unlink, verify exact file list, unchanged SHA/inode/size, no FD/cwd/process/active lease, and affirmative recoverable provenance. Use manifest enumerated exact file unlink and rmdir for empty dirs, never generic rm -rf over a PI directory.
4. REDUNDANT_MODEL_ONLY_VIEW: at most two new derived inference view candidates. Not full Trainer. Eligible only with exact derivation identity, archived model + processor/stats and/or locally protected identical full copy, full-byte checksum, documented reconstruction, zero active refs and no task dependence. Existing 7 retired views excluded.
5. UNKNOWN_HOLD: PI0209, any missing rights or bounded reference scan/incomplete archive, personal-quota uncertainty, hardlinks/symlink conflicts, any full Trainer or unique checkpoint without complete independent snapshot. Zero delete.

Scientific quality FAIL does not imply discard. Immutable result receipts and all historical TRAIN/DEV/CONFIRM scores persist. A directory with 337.6GiB of PI0276 is **not** a deletion target because paused archive processes still depend on it. Do not SIGKILL/SIGCONT those processes to free files. If a supposedly deleted file remains open and space is not released, report it, do not kill its owner.

## Finite maintenance GO

M0 — pin four source audits and seven sample files, display exact candidate-by-candidate decision, recheck NFS/SSD/home device/inode/links/quota/known process refs. Do not re-walk/hash all 1.81m entries. The known static-ref scan cap is not proof of no users.
M1 — implement/validate storage README and manifest-only navigation; preserve original shared/runtime paths. Bounded seven-file copy-only pilot first, then eligible copy staging from up to 24 listed candidates with explicit capacity checks, SHA checks and no source deletions. Any live-path change or bad link stops that candidate.
M2 — reclaim only up to nine proven cache candidates and at most two newly identified proven redundant derived model-only views; max **50GiB newly freed allocated bytes** in this phase (earlier 41GiB excluded). Each file's expected SHA, inode, size, exact path and independent recovery must be verified immediately before unlink. If size exceeds cap or metadata drifts, SKIP. No bulk deletes/recursion outside exact manifest.
M3 — record before/after per device, unchanged protected hashes, explicit deleted/copied/held counters and all SKIP reasons; do not overclaim freed quota from df, which may be affected by other writers. No daemon, recurring job or further stage without review.

Stop on: missing source audit, any process/FD/unknown-owner dependency, closed permission subtree, incomplete immutable HF copy, mismatch hashes, unclear NFS/SSD identity, unexpectedly active PI0290 path, storage shortage, tool errors, or budget reached. On failure, preserve source and return exact blocker. No broad first-delete-then-repair.

## Deliverables

Save private path-level originals/receipts only on local PC/Corelab. In GitHub commit sanitized:
- SOURCE_AUDIT_SHA.json; PATH_ROLE_MAP.json; RETENTION_DECISIONS.json
- MIGRATION_COPY_RECEIPT.json; DELETION_RECEIPT.json; RESOURCE_BEFORE_AFTER.json; UNRESOLVED_AND_NEXT_ACTIONS.md.

Under reports/maintenance/2026-10-13-corelab-storage-phase2/<session> with source SHA, actual run code SHA, command summaries, file counts/sizes and all failures. Post a complete final maintenance record to Issue #47, independent readback, brief pointer to #11 only if data paths relevant to PI0290 changed. This is an existing local-PC owner task, not a GitHub cloud Codex launch. No new GPU/model/teacher/physics, no WBC, no hardware, no changes to PI0290 results or selected/deployment=NONE.

Authority deliberately excludes deleting unique large checkpoints, any full Trainer, or PI0250/PI0276/PI0290/private WBC. A later exact named-file cleanup/owner decision is required for those protected groups.

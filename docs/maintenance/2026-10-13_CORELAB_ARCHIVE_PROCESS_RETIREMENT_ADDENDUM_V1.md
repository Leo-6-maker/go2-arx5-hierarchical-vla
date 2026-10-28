# 2026-10-13 Corelab stopped archive-process retirement: maintenance addendum V1

**Authority:** User explicitly confirmed the two stopped old archival processes are no longer needed for current experiments and may be retired. **Scope:** process retirement only; existing storage GO remains controlling. Not a science GO.

**Parent maintenance GO:** docs/maintenance/2026-10-13_CORELAB_STORAGE_SAFE_CLEANUP_V1.md at 6b4cf12ef55ed45730069fd15d32e397b7fd7b16, Issue #47 comment 6129475191.

This addendum **supersedes only the parent's categorical prohibition on terminating the TWO individually verified, user-owned, STOPPED PI0276 legacy archival processes**. It does not permit deletion of PI0250, PI0276, PI0290, PI0209, unique checkpoints, private WBC, original physics/images/videos, failed-run evidence, or full Trainer originals. The parent's new-space reclamation cap remains 50 GiB allocated, maximum nine named caches plus two independently recoverable derived model-only views; the earlier 41.0 GiB cleanup is separate. The existing local-PC agent alone may act.

## 1. Narrow process candidate identity

Earlier handover observed legacy PI0276 parent process PID 3455361 and predictions worker PID 3455362, both state T. PIDs are HISTORICAL LOCATORS, not authorization to signal current processes by PID alone. If PID reused, commandline or user changed, state is not T/t, associated job/lease is actively running, or any process belongs to a different project/user, SKIP and report.

Before signaling, the existing local-PC owner must:
1. Read, SHA256-pin and reconcile the actual local AUDIT_REPORT.md, FULL_STORAGE_INVENTORY.csv, RETENTION_PLAN.json and README_STORAGE.md at work/corelab-storage-audit-20261013, plus its path checker. If unreadable, data deletion is BLOCKED_AUDIT_SOURCE, with deletion count zero. This ChatGPT environment does not contain these local originals.
2. For each PID, record current /proc start_time_ticks, UID, PPID, PGID/SID, state, executable, commandline, cwd, open FDs and descendants. Confirm user-owned PI0276 archival role using TWO independent runtime observations and verify PI0290/current science jobs and other users are unaffected.
3. Preserve the current partial archive outputs and original logs as files; record their provenance and size, and inventory known HF archive/manifest status. Stopping old computation is not a mandate to delete its data.
4. If BOTH actual process identities are still exactly the eligible STOPPED archival roles and no unrelated process would be harmed, terminate only the explicitly matched process IDs, leaf before parent. For stopped tasks, a direct individual SIGKILL is permitted after identity checks; NEVER send blanket pkill/killall, process-group signals, sudo, SIGCONT, or resume+TERM that could let an old job briefly continue writing.
5. Record exact target PID, start_time_ticks, signal result, observed exit or zombie status, after-state of FDs and disk allocation, and whether outputs stayed unchanged. If identity ambiguous, SKIP. Never claim freed disk space merely because a PID exited.

If existing phase-2 cleanup started before this addendum, only the future process-retirement step can be added after the owner reads and acknowledges this immutable version. Never rewrite parent evidence.

## 2. Protection after process retirement

Removing the paused-process dependency is **NOT** automatic approval to delete its PI0250/PI0276 files. Reclassify only the relevant path dependency from PROCESS_HELD to INDIVIDUAL_FILE_REVIEW. A file can be unlinked ONLY when the parent storage protocol's precise source SHA, inode/link identity, no consumer/FD, complete independent recovery source, full backup hash and original manifest classification pass. The original 337.6 GiB PI0276 SSD directory is NOT a bulk deletion target; the PI0209 53.9 GiB home directory remains UNKNOWN_HOLD. No recursive rm -rf of experiment roots.

Source domains remain distinct: original Panda/PI0276 task data, complete Trainer, model-only derivations, PI0290 complete dual-RGB/contact pilot and old WBC assets are not interchangeable. The eight original PI0290 pilot episodes (held-out 4/4), commissioning failures, camera frames, physics and score receipts are PROTECTED. M1 is still COPY_AND_VERIFY only; it does not delete migrated sources. No increase in original 50 GiB bound.

For more than 50 GiB of verified eligible large-checkpoint reclamation or deletion of a formerly protected directory, first deliver a separate exact candidate list, per-file duplicate proof, independent immutable archive and expected actual allocated-byte savings, for further user review.

## 3. Mandatory receipts

Add one sanitized PROCESS_RETIREMENT_RECEIPT.json to the existing phase-2 receipt directory with: PID locator, immutable runtime identity, reason/status, signal sent, exit observation, referenced-file preservation, FD closure, false-positive checks, genuine recovered bytes, unresolved risks. Exact private paths and process details remain local.

Continue existing M0–M3 SOURCE_AUDIT_SHA, PATH_ROLE_MAP, RETENTION_DECISIONS, MIGRATION_COPY_RECEIPT, DELETION_RECEIPT, RESOURCE_BEFORE_AFTER and UNRESOLVED_AND_NEXT_ACTIONS, including per-device before/after and ALL SKIP reasons. Post actual maintenance receipts to Issue #47; scientific outcomes/PI0290 unchanged. GitHub publication is not a remote execution receipt.

Forbidden throughout: new Hopper PBS/GR00T, WBC optimization/runtime, new MuJoCo physics, teacher sampling, hardware, tampering with other user tasks, irreversible deletion without backup. selected/deployment remain NONE.

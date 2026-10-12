# Corelab Phase-2 Actual Maintenance

Status: **PARTIAL_WITH_SKIPS**. This is maintenance evidence, not a scientific GO or a claim that storage cleanup is complete. Existing local-PC remains the sole mutation owner. Private exact paths, commands, FDs and credentials are not published.

## Actual Results

- M0: all four original audit sources plus the path checker were read and SHA256-pinned. Complete CSV/JSON reconciliation passed for 448 entries and 1,812,834 original regular-file counts, without rescanning that population.
- M1: the original seven-file copy pilot passed. **22 of 24 candidates, 2,206 regular files** were copied and fully content-read-back verified. All sources remain. Copied content is **140,069,069 logical bytes**, with **144,109,568 allocated bytes** in the destination regular files. Copying within shared NFS does not reclaim space.
- Two candidates were skipped because their two known external static references each were not individually resolved. They were not moved or deleted.
- M2: **zero files deleted and zero newly reclaimed bytes**. All nine original cache candidates were skipped: affirmative rebuild/recovery and complete reference visibility are not proven. No newly eligible derived inference view was identified; the limit of two is not an obligation to delete two.
- The 50 checkpoint-index entries remain protected: 48 partial/weights groups and two full-Trainer-component groups, all without resume qualification in the source audit. No full Trainer or unique checkpoint was removed.

## Stopped Archive Processes

The owner validated both historical PI0276 process identities against current UID, start ticks, exact role and T state, and used individual pidfd SIGKILL, leaf before parent. No group signal, SIGCONT, or signal to another process was used.

The first action signaled the leaf, then stopped on a post-exit FD `PermissionError`. Its actual partial receipt was preserved. The remaining exact stopped parent was revalidated before a separate bounded continuation; the leaf was not signaled again.

Final observations: leaf **Z**, exit observed, FD visibility **UNKNOWN_PERMISSION**; parent **ABSENT**, exit observed. This proves exited computations, not that the zombie PID was reaped or that every FD was independently inspectable. Two individual signals, zero other processes signaled.

Three original logs, **3,591,251 logical bytes**, retained unchanged content hashes and metadata. The original retirement script's nonrecursive tar glob matched no nested archives; therefore its generic archive-unchanged flag does not prove preservation of the large tar contents. A later read-only check observed the two specifically named source/predictions archives with their original audited sizes; it is **post-only metadata**, not pre/post verification or full-byte readback. Both remain protected and were never deleted.

## Paths And Resources

Shared NFS regular files are one shared asset. Node-local SSD is accessible through the owning executor, not automatically through an AMD path of the same name. Existing NFS aliases to 4080S SSD may be broken on AMD without indicating data loss. No existing alias or active PI0290/PI0291 path was rewritten.

The public [host-aware role index](PATH_ROLE_INDEX.json) links PI0291's actual resource binding, while exact realpaths/device/inode/link identities remain private. Deployment of remote navigation or a new path checker needs its own owner receipt; this public index does not manufacture that proof. The old generic checker was read, not silently declared runtime-qualified.

Known measured maintenance CPU is **2.430096401 seconds** across the recorded parse, copy, reference checks, final process continuation and archive-stat probe. An additional **30 CPU-second conservative upper bound** covers tiny unmetered probes and the initial partial action; the owner must aggregate it once into PI0291 accounting. Maintenance introduced no physics, GPU workload, GR00T model operation, Hopper PBS or WBC runtime.

Current filesystem figures are post-only snapshots; the original audit is not a synchronized immediate-before baseline. Changes caused by other writers are not counted as reclaimed bytes. Personal quota remains **UNKNOWN**, and NFS's raw zero inode-availability result is not interpreted as exhaustion.

Earlier Phase-1 retirement remains separate: **44,017,266,688 allocated bytes (40.994 GiB)** from seven derived BF16 views. It was not repeated or included in this phase's zero reclamation.

## Evidence

[Actual receipt](ACTUAL_RECEIPT.json), [source pins](SOURCE_AUDIT_SHA.json), [copy receipt](MIGRATION_COPY_RECEIPT.json), [retirement receipt](PROCESS_RETIREMENT_RECEIPT.json), [retention decisions](RETENTION_DECISIONS_SUMMARY.json), [deletion receipt](DELETION_RECEIPT.json), [resource snapshot](RESOURCE_BEFORE_AFTER.json).

The owner subsequently wrote and read back private host-aware navigation registries on shared NFS: 57 entries on each host, all 57 observed on 4080S; AMD observed 49 and explicitly deferred its eight SSD-local entries to their owning executor. Both registries were fetched to local-PC. See [navigation receipt](NAVIGATION_READBACK.json). No existing path or symlink was changed. This adds 0.022783714 measured CPU seconds to the separate maintenance accounting.

GitHub publication/readback remains an owner step. `selected=NONE; deployment=NONE`. Scientific results and original datasets were not changed by this maintenance.

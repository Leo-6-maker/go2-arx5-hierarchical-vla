# R0/R2 runtime critical path V3.2 — CPU only

Base: PR #395 at `7930e9a6ccac4d0dff0c2bb402323b22d2ca9100`. This independent successor implements the attachment’s directly authorized first engineering phase. R1 SHIFT7 80→180 and R3 ≤32/64 remain proposals, with no new GPU grant inferred from unused branches. No active Goal is created or resumed.

## Fixed contracts

Same V3.1 request/ACK schema, bindings, explicit clock modes, 400 ms TTL, complete remaining 50 ms check, source-time renewal, original full-H40 compiler, EEF/gripper mapping and recorded inputs/outputs. No model/codec/statistics/precision/sampler changes; no training, WBC, native or safety admission. The legacy Receiver default still revalidates files at each boundary; a new capsule can opt into an explicitly verified immutable observation and in-memory compiled reference. Installation identity and every original STOP/TTL/history check still run. The cache belongs to the receiver process; an exited process cannot adopt an old receiver directory.

The input boundary reads, hashes and decodes the same bytes once (maximum 4 MiB encoded and uncompressed). Arrays have immutable byte backing; file changes after reception cannot rewrite the issued observation. Different pins require another validated boundary. Compiled cache identity must match the exact installed pin. On archive submission, original input/action/compiled hashes must still match their issued pins; a source-file mutation cannot silently change evidence.

## Bounded persistence

The existing Journal and OwnerLock own one archive thread. Reserve before capture/dispatch, not after ACK. At most four reservations, running archives and queued archives in total; reserve byte capacity in advance. Queue saturation backpressures with a finite wait/STOP check. Seal one quiescent per-request subtree and compare source manifest before/after copying, archive manifest, raw hashes and bytes. Fsync files and nested directories before the Journal COMPLETE receipt. Archive errors latch a new-dispatch stop; already dispatched evidence may drain. No evidence deletion or automatic retry.

Reservation lacking a COMPLETE verified archive is UNKNOWN after process death, even if its actual neural cost is unknown or zero. Reopening the owner audits completed receipt/data hashes; unknown or changed evidence blocks adoption. A completed identity cannot be reserved again. A hung archive has finite drain timeout and the existing service deadline; its evidence remains UNKNOWN.

## CPU evidence panel

Planned zero-generation cached replay: 16 sealed V3.1 ordinary outputs × two observation boundaries = 32 CPU cases. FILE_APPLICATION_CACHE_EMPTY starts with an empty application observation cache, reads a saved file inside the clock, and **does not flush or control the kernel filesystem cache**. DELIVERED_MEMORY_RECORDED_RGB validates already delivered recorded bytes, records excluded preparation separately, and **does not measure a live camera**. Baseline file/hash verification and saved neural-output preparation are outside ACK timing and are separately reported. No future-window prefetch is counted as camera speed.

Compare complete nominal/reference H40 byte-for-byte, unchanged EEF wire payload, emitted atom values/native clocks and archive bytes. Source/request identities and monotonic timestamps are newly issued; they are not falsely asserted equal to old runtime identities. Report cached CPU latencies separately from real GPU latency; no GPU throughput, 2 Hz, wall-clock or native qualification follows from this panel. GPU/native packages are not imported.

The Core CPU service uses the existing registry/Journal/user-service route, no neural lock, a fresh sealed source, 2 CPUs, 2 GiB RAM and a finite ≤900 s deadline. Pending request evidence is at most 4 MiB; retained completed live files remain under the existing 64 MiB finite tmpfs cap and are not independently deleted.

## Safety remains blocked

Three historical table contacts, 1/38 expert false alarm, unqualified STOP latency and unknown controller-selected reference remain. Offline request caching does not resolve them. Independent signed readmission, including any SIM_PAUSED_NATIVE profile, is still required; P0/pilot/DEV/TEST are not launched. Historical RP8/JOIN10/DEV/TEST and both V3.1 negative timing results remain intact.

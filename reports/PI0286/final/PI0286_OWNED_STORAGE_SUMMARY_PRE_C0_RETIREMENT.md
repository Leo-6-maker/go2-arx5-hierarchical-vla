# PI0286 owned storage: read-only snapshot

Scoped only to each PI0286 root and Core PI0286 SSD tier. Regular files counted once per `(host, device, inode)`; external parent/expert/pristine directory aliases are not traversed. Files are live, so this is a timestamped snapshot, not an atomic maximum. Metadata `st_blocks×512` is actual reported allocation, not physical backend dedup/quota.

| Host | Unique logical files (bytes) | Unique allocated files (bytes) | Directory allocation (bytes) |
|---|---:|---:|---:|
| core | 25,167,794,582 | 25,184,743,424 | 180,224 |
| hop | 27,601,745,103 | 27,602,743,296 | 413,696 |

Aggregate including directories: **52,788,080,640 B (49.1627 GiB)**. Cap100GiB=107,374,182,400 B; current headroom=54,586,101,760 B.
Three new full14 canary checkpoints: 3×20,468,176,040=61,404,528,120 B. Static aggregate projection **106.3502 GiB**, excess **6,818,426,360 B (6.3502 GiB)**, before later raw growth. C0 canary large payloads are already retired; formal C0 full Trainer remains intact.

## Core largest directories

| Directory | Unique allocated bytes |
|---|---:|
| ROOT/durable/episodes | 16,342,740,992 |
| ROOT/models/C0_REPLAY1024-1024 | 6,292,754,432 |
| ROOT/data | 1,386,774,528 |
| ROOT/source-NFS-engineering-archive | 583,245,824 |
| ROOT/archive/correction-closed | 350,355,456 |
| ROOT/correction | 78,938,112 |
| ROOT/evidence | 72,491,008 |
| SSD/source/evidence | 35,454,976 |
| SSD/runtime | 17,125,376 |
| ROOT/archive/report-cpu-supplement | 9,392,128 |
| ROOT/diagnostics | 6,520,832 |
| ROOT/transfers | 4,849,664 |

## Conditional duplicate retirement candidates — report only

1. Hopper consumed C0 transfer delta: 5,670,088,704 B. Alone insufficient for the projected canary excess. Preserve full Trainer and all small evidence; close any transfer/archive consumer first.
2. Core C0 materialized BF16 view shards: 6,288,175,104 B. No matching argv/maps/open-FD references were found at13:16. The resident evaluator still logically depends on C0; owner must confirm no reload/restart dependency and preserve reproducible exact inputs before removal. No Core full-master/delta remains in the scoped model tree.

Combined duplicate payload capacity is11,958,263,808B; this is not authorization or proof that removal is safe now. All formal full Trainer checkpoints, parent, expert data, failed/valid raw and metadata remain protected.

C0 first full-master shard `de31bfff...` differs from original parent `c3388582...`: no parent hardlink reuse is supported. C1 immutable canary shard identity is currently unobserved. Hardlinks can preserve complete checkpoint bytes only for actual equal SHA+size, same filesystem, immutable targets, and verified closed consumers; frozen-module claims alone are insufficient. No hardlink or deletion was performed.

## Planned C1 model-only transfer after canary retirement

Delivery reports planned delta5,670,085,448B plus BF16view≈6,288,169,448B, peak11,958,254,896B; after consumption only6,288,169,448B view remains. With current baseline and one complete new C1main full14 checkpoint, the static post-canary transfer projection is85,214,511,576B (79.3628GiB). These future files are not yet actual inventory, and later raw growth remains additional. The three-canary transient remains the larger106.35GiB projection. No storage relocation within owned namespaces alone reduces the cross-host100GiB total.

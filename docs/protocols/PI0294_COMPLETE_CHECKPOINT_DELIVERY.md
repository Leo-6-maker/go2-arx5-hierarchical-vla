# PI0294 Complete Checkpoint Delivery

Status: PREPARED_NOT_ACTUAL_HF_DELIVERY. Byte verification cannot establish a
working policy or a successful independent restore.

The checkpoint publisher reuses the existing native full-state byte verifier.
PI0292 keeps its original registered steps and cyclic sampler; PI0294 reconstructs
its own position-addressable sampler from the exact frozen dataset manifest,
including consumed cursor, horizon, seed, task/stage buckets, and explicit empty
bucket handling. Saved sampler text is not accepted as its own independent proof.

Every model shard, optimizer, scheduler, RNG, Trainer state, statistics, sampler,
processor configuration/statistics/embodiment mapping, and save receipt is hashed.
Only safetensors headers are inspected. No tensor materialization, model import,
GPU computation, pickle execution, training, or physical simulation occurs.
Symlinks, unexpected files, changed statistics, incomplete model indexes, and
post-upload byte changes are rejected. Engineering M2/M4 is explicitly not a
policy candidate; M512 remains development, and no inventory grants selection.

Publication uses the existing authenticated private HF dataset repository under
a fresh checkpoint-specific prefix. Full state files stream directly from the
immutable checkpoint; there is no second local archive copy. The manifest pins
the original training request/allocation/source and the delivery source. Large
files are verified by fixed-revision LFS SHA and size, not downloaded again.
Small files are read back exactly; READY is committed last and verified at its
final revision. An unknown commit outcome is preserved without automatic retry.

The current CPU delivery entry requires an exact sole-owner CPU reservation,
known terminal native training origin, fixed run/source/lease/manifest, one
allowed CPU, at most 30 minutes, at most 32 GiB reserved temporary growth, the
original 180 GiB owned-space cap, and the original deadline. It refuses pending
neural work or GPU reservations. Its bounded space/time monitor does not reset
scientific counters. This entry therefore does not yet authorize simultaneous
checkpoint uploads during active training or another GPU allocation.

No local checkpoint deletion occurs. A successful private delivery remains
distinct from the strict numerical smoke gate, formal training, DEV selection,
fresh-observation closed-loop scores, and final policy qualification. Those
actual scientific receipts remain required. WBC, Go2, hardware, and a second
owner are outside this operation.

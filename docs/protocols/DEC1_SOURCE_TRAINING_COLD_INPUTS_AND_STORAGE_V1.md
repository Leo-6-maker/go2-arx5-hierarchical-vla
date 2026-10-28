# DEC1 Source Training: Actual Cold Inputs And Bounded Storage

Status: **INPUT_BYTES_PREPARED_NOT_NATIVE_REGISTERED_OR_TRAINING**. This uses
authority `a7a7ecd9a6be62a7221af25a5e32a0009b076a84`, the original PV1 clock
and remaining budget. It is not another GO or a change to either recipe.

## Actual Input Evidence

The existing owner reread both registered parent controls and every campaign
child: 29 CoreLab children and 12 Hopper children were CLOSED with no live
recorded process; Hopper's own native queue was empty at that read. Saved
CORR parameter inventories exist in B0 children03/04/10. Thirteen actual
small source files were read back, not reconstructed from fixture metadata.

Both arm input sets now have the exact CORR@2048 complete-parent manifest,
original R0 statistics, original 139-parent bank/READY, original C1, original
population bytes and source-pair recipe. Mature metadata enumeration,
MatchedSampler, primary_recipe and fixed TRAIN probe functions were reused;
the old full-B0 builder was neither patched nor fed a fabricated B0 PASS.
No `PV1_B0_DEV_ADMISSION.json` was created. The new explicit
`DEC1_SOURCE_TRAIN_STAGE_V1` binding records no baseline-relative release or
LOCKBOX permission. Each arm starts at zero with its own optimizer, stops at
1024 for the twelve archived TRAIN-only queries, and later fully resumes to
4096 under the existing native Trainer.

The common metadata view has 160930 overlapping windows per arm. Identical
support hashes do not imply identical exposure: A_CONTROL still draws zero
new-bank optimization samples; B_COVERAGE uses the original fixed lane
schedule. Shared task, C1 and retained-original cursors are unchanged. These
are cold input bytes, not actual samples consumed or optimizer updates.

## Measured Historical Timing, Not New Training

Two real prior CORR-parent step logs cover steps1-512 and513-2048, 2048
contiguous updates in total. Their maxima are 5.646339230239391s and
5.4172047059983015s. An illustrative 6.3s/update upper includes more than10%
over the larger observed maximum. New PV1 data/IO may differ; this is not a
new PV1 throughput measurement or a guaranteed bound.

At that declared rate, two3h initial phases and two6.5h continuation phases
would fund both endpoints plus qualification/load/save/closeout and the
archived probes. Their19h plus an explicitly unmeasured24h matrix ceiling
and seven60s accounting pads is43.1167h inside48h. Queue delays, all CPU,
transfer/storage counters, the original final20h protection, the original
cutoffs and native site admission must still pass the complete fresh bridge.
This arithmetic is not a sealed admissible forecast or an allocation.

## Exact Frozen-Shard Sharing

The actual parent complete payload is25541242392bytes. A header-only read
cross-checked against the real saved parameter inventory identifies exactly
one complete shard,4986649584bytes, containing433 known frozen backbone
tensors and no unknown or trainable tensor. The two other shards are mixed
or trainable and are ineligible. No tensor or model was loaded for this check.

`checkpoint_storage.py` adds a narrowly opt-in post-save wrapper for the new
DEC1 source scope. The mature Trainer first writes its normal complete model,
optimizer, scheduler, RNG, sampler, processor and statistics checkpoint. Only
then may a manifest-sealed policy replace that one newly saved duplicate with
a same-filesystem hard link to the exact original parent member. Complete
manifest hashes, both complete file hashes, all tensor names/trainability,
ownership, inode/mtime witnesses and parent permissions are checked. Mixed
shards, changed bytes, symlinks, foreign jobs and unknown existing links stop.
No parent permission or tensor is changed; no checkpoint is retired. Legacy
scopes without that sealed opt-in retain their original save behavior.

A hard link is **not an independent backup**. The source remains logically
protected by its fixed complete manifest and consumer ownership; its original
mode is not changed. The accounting witness records the preexisting parent
inode, so an owner can distinguish new physical allocation from a logical
file that refers to already existing bytes. Never erase historical peaks.

Repeated native saves at an already complete step do not invoke the writer:
the first full state and all members are checked and preserved. This prevents
a second save from truncating the parent-linked shard. Partial/corrupt states
are not repaired or overwritten. A durable sharing intent precedes the exact
link replacement; an incomplete or mismatched receipt blocks subsequent
training configuration until read-only reconciliation. The source guard is
still called before/after storage work.

## Still Required Before Full Pipeline Admission

Four unshared payloads would exceed64GiB. One possible presealed lifecycle is:
finish A1024/probes and A4096; then publish A1024's full state to fixed HF,
independently fully read back its members on CoreLab, preserve its recovery
identity there and on HF, and only after actual closure/ref release retire
that exact Hopper copy under the existing maintenance rules. Then run the two
B phases with the same frozen-shard sharing. All four logical full states
remain recoverable; there is no proposal to discard an effective checkpoint.

Using a conservative26GB payload upper, two existing shared-state increments
plus one complete transient save require68026700832bytes, leaving only
692775904bytes below64GiB **before other outputs**. This tight margin is a
real risk, not solved merely by a hard-link unit test. Source capsules, traces,
temporary files, metadata and any restore/archive work must be included in an
actual sealed peak/lifecycle plan. Closed trace archives need exact readback
and consumer closure before any eligible local duplicate retirement. Every
later phase must pass the actual pre-write resident headroom guard.

One full parent-sized upload plus independent complete readback costs up to
51082484784 logical transfer bytes, leaving17636991952bytes of the64GiB cap
for all other new transfer. Checkpoint delivery/retirement and those measured
storage transitions are **not executed or qualified by this implementation**.
The owner must not declare the whole source pair natively ready until the
complete fresh budget inputs and standard cold registration actually pass.

Hopper's execution-hook root cause remains unconfirmed. The old exact held
job is terminal and settled, but this does not authorize blind same-hook
resubmission. An alternative platform still needs explicit accounting and
actual numerical/precision/model qualification. No H200-only guard was
removed. The target task ABI and native interface commissioning also remain
separate unfinished work. Selected/deployment remains NONE; no new model,
GPU/PBS, query, training update, physics or WBC task occurred here.

# PI0292 Data Milestone

This is an intermediate actual receipt. PI0292 S0-S6 is still active, not complete.

## S3 Update: Scheduler and Quota Holds

After the data milestone below, the first bounded smoke submission was actually
rejected: `qsub: Budgets-am_hook: Project has already ended.` The historical
project CFP03-CF-004 is not currently usable. No job ID was issued; a fresh own-job
query is empty. The failed submission is preserved and conservatively counts as
one of the three allowed submission calls. GPU allocation/model loads/updates
remain zero, and S3 is NOT_RUN rather than a model failure.

The actual `hpc project date` command now requires the user's account-manager
login. The owner has requested that the user authenticate directly on Hopper and
identify a currently authorized project, without sending credentials. The old
Home project-balance text is historical and is not reused as a current budget.

The actual personal quota tool reports scratch 940.7 GB used / 1.0 TB hard limit
(2026-10-13 13:43 local report). Even the generous binary-unit interpretation
leaves only 83.3 GiB, below the prepared 80 GiB new smoke peak plus 20 GiB margin.
Exact byte quota is not supplied by that display; shared filesystem free space
does not override this personal limit. More quota or an explicitly authorized,
verified storage binding is needed. No historical Hopper model/data deletion was
performed. The resource ledger retains an additional 600 CPU-seconds upper bound
for unmeasured submission/preflight metadata, not a claim of zero usage.

`S3_RUN_MANIFEST.json` and `S3_ADOPTION_DELTA.json` bind the original run/lease,
source c187d957 and data/statistics to the existing launcher. Hopper's new
workspace entry is still NOT_DEPLOYED. No hot source switch, new top-level run,
deadline reset, or claimed workspace deployment took place.

Independent CPU work implements the durable paused-relay packet state machine:
strict student fields, original 256x256 PNGs, observation/action/ACK identity,
frozen SafetyGate, prefix8 only, and persistent no-replay reservation. Synthetic
tests cover corruption, stale/reordered/duplicate messages, timeout, reconnection,
and exactly 80 prefixes. Lost ACK means unknown physical progress, not permission
to replay. This is CPU-only protocol evidence, not a deployed dual-SSH endpoint,
teacher physical qualification or neural closed-loop success. Those remain NOT_RUN.

## Earlier Data Milestone Snapshot

- Authority: fixed release `261bc5b37f20aaff1ad7d5742252d3dc1a91fa65`, Issue #11 comment 6137102979.
- Run `20261013T025824Z-localpc-pi0292`; T0 `2026-10-13T02:58:24Z`; new-work cutoff `2026-10-14T01:58:24Z`; deadline `2026-10-14T02:58:24Z`. No reset.
- All24 successor attempts completed in DEV16 then TRAIN8 order. Original dispatcher PID/start exited before the separate CPU bank pipeline started. Both environment canaries also completed, excluded from TRAIN/DEV. Original56 were not recollected.
- Bank raw admission: `ARX5_BANK_V2_QUALIFIED`; counts `{'TRAIN': {'planned': 64, 'admitted': 64}, 'DEV': {'planned': 16, 'admitted': 16}}`. Original PI0291 remains PARTIAL; original ledger/statistics and the initial canary audit failure are preserved.
- New-bank CPU pipeline: `BANK_V2_CPU_AND_EXPERT_PROBE_QUALIFIED`, current `None`. Adapter gate `PASS_BANK_V2_TRAIN_DEV_AND_TRAINER_LOADER_CPU`. New statistics fitted once on admitted TRAIN only. DEV is never optimized or fitted.
- All-window numerical receipt: TRAIN 38464 windows; DEV 9616 windows. TRAIN normalized XYZ maximum-component error 8.57280324506604e-09 m, strict gate passed when the numerical receipt says PASS. DEV has 44 inverse-clipped components, 9 state out-of-range components, and maximum-coordinate reconstruction error 0.001337304592551547 m. DEV is reported, not subjected to the TRAIN-only strict bound; no refit or threshold change.
- Expert probe seal: `SEALED_BEFORE_MAIN`; actual windows `192`, expert rejects `0`. Evaluation code, canonical Gate and two-noise identities sealed before any main training; no model-generated trajectory is claimed.
- Hopper original56 dependency: all56 original archives and all members verified. Actual RGB/official processor/collator tensors match Corelab with workers0/2. New-bank consumer `PASS_BANK_V2_FULL_DATA_AND_CPU_INTERFACE`; completed new archives `26/26`. Original56 archives will not be downloaded again.
- New private HF stage `BANK_V2_FIXED_HF_FULL_SET_AND_ACTUAL_NEW_EPISODE_READBACK_PASS`; fixed revision `6075aac0cdc44b8d252d7540efbac208701c30b3`. READY-last, full file-set/hash verification and actual sample readback are required; pending stages are not PASS.
- Packaging repair flag `True`. The first pre-upload failure, when present, is retained in HF_PACKAGING_FIRST_FAILURE.json: the allowlist omitted already sealed C++ audit sources and README. The explicit repair admits only those exact paths and rechecks/reuses existing raw archives; no original data, source capsule or scientific threshold changed.
- Existing N1.7 full-shard identity and Cosmos cache match their fixed official revisions, without historical model redownload. Models loaded `0`; PBS submissions `0`; neural smoke/main/offline/relay remain NOT_RUN.
- Corelab actual totals: 26 starts, 416000 integrations, 16666 paired observations; measured CPU 12923.228333s plus retained conservative/reserved 600.00s. Hopper measured CPU 381.024862s; stage reservations are separate in its ledger. Quota remains UNKNOWN, not inferred from shared free space. Scientific deletions, WBC and hardware calls remain zero.
- Latest code available to this projection: `1f17a09799cca7109c39a5a83c547d242d0315d0`. Executed pipeline source and step receipts are in BANK_PIPELINE/adapter evidence; physical source remains `35290803cd82e10a164daaa53af5743dd43054ee`.

The zero-submission count and quota UNKNOWN in the earlier snapshot above are
superseded by the S3 update and S3_PREFLIGHT_BLOCKERS.json, not silently rewritten
as if the first refusal had not occurred. Proceed only through the adopted
data/model/resource gates. Fixed-base ARX5 results do not prove mobile Go2, WBC,
or hardware success. selected=NONE; deployment=NONE.

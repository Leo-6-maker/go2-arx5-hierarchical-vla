# PI0162 guarded synchronized diagnostic

Status: **BLOCKED** (fail-closed at the authorized simulator-constructor
budget).  Authority: Issue #11 comment `5652576782` / PR #69 comment
`5652578721`.  Execution HEAD: `630694960e060a40bfb802e40b432c293bfff08c`.

## What was verified

- The source-backed audit found no active superseded PI0160 process on
  `corelab-4080s`; v1 and v5 have no reliable admission/result source and stay
  `UNRESOLVED_NO_RELIABLE_SOURCE`.  v2/v3/v4/v6/v7 retain their original
  source-backed lower bounds and hashes; the historical reported PI0160
  overage is not rewritten.
- The fixed campaign ledger is outside all versioned output directories:
  `/var/cache/users/liuyu-nfs/stage45f_pi0162_campaign_20260913/ledger.json`.
  It uses one campaign id, an atomic file lock, durable pre-call reservations,
  no refunds, and process handoff/restart preservation.
- CPU regression: 5/5 tests passed, including concurrent launcher exclusion,
  constructor-failure consumption, restart/directory persistence, exhausted
  cap rejection before the underlying call, and existing-raw no-duplicate
  behavior.

## Execution outcome

Three constructor reservations were consumed exactly as required by the
guard.  The first attempt used the system Python 3.12 and failed at the
IsaacGym import boundary (`No gym module found for the active version of
Python (3.12)`).  The second used the qualified Python 3.8 executable but was
started without the historical process-start `LD_PRELOAD`/`LD_LIBRARY_PATH`
layer and failed before native construction (`libpython3.8.so.1.0` loader
error).  Both are preserved as entered/uncertain failures; neither is
refunded.  The third attempt was started with the corrected loader environment,
but the PI0162 constructor cap was already exhausted and the C0/C1 calls were
rejected before `_build_native`.

Consequently:

```yaml
conditions_started: 2
qualified_q0: 0
simulator_constructor: 3 / 3
explicit_control_tick: 0
reference_update: 0
GR00T_constructor: 0
GR00T_forward: 0
conditioning: 0
action_head: 0
model_side_call: 0
training/backward/optimizer: 0
WBC: 0
Validation-E/Holdout-F: 0
ICL/DAgger: 0
H200/PBS/hardware: 0
```

No C0/C1 image, StateV1, EEF, expert trajectory, model raw array, or decoded
metric exists in this packet; none is fabricated.  The existing PI0160 raw
directories remain immutable and were not rerun.

## Node-local evidence

The complete files remain on the Corelab node (not copied into Git):

- source audit: `/var/cache/users/liuyu-nfs/stage45f_pi0162/audit.json`
- audit SHA256: `b2d74e824da6311a992191fc6e377475791ba8f337efda16c8f913cbf7ded168`
- immutable campaign ledger SHA256: `bd1bd9fff65443e0e378c6d9197da0734c7af042469390f2cbd1b2796c13f834`
- first blocked output: `/var/cache/users/liuyu-nfs/stage45f_pi0162_guarded_sync_20260913_v1`
- second blocked output: `/var/cache/users/liuyu-nfs/stage45f_pi0162_guarded_sync_20260913_v2`
- cap-rejection output: `/var/cache/users/liuyu-nfs/stage45f_pi0162_guarded_sync_20260913_v3`

The v3 output's `pi0160_matched_expert_result.json` SHA256 is
`65fe658913b313077fe0f6a33230b9352ad729ed01c3b80261494a7eb2eb4198`;
its C0/C1 failure files are `66b9a712b98a71be1f9e575be2171d2fde73e721be1cdda9bc99858d04ae350e`
and `392bc27143845f323bf3c8cb1f0b081ad87a97b1eab5b626ab2ce117321391f7`.

## Claim boundary and recommendation

This is an execution/admission result only.  It says nothing about target
progress, synchronization quality, expert tracking, or N2 behavior because no
new q0 was admitted.  The single next intervention is a new PI gate that
authorizes a fresh fixed campaign ledger with enough simulator-constructor
capacity to run both C0 and C1 under the already verified process-start
launcher; do not alter the seed, noise, stats, camera, SafetyGate, or retry the
consumed attempts under PI0162.

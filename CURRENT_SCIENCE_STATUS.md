# Current Science Status

Observed 2026-10-17 11:42 SGT by the existing single local-PC owner.
This is an actual-state summary, not a replacement for immutable run receipts.

## Active Authority

**MP1 is the only new incremental resource family.** The user directly adopted
release `c8cdf60e41e31da5796b44a064ef2f22d594a491`.
The concurrently published E2 documents remain non-additive references; there
is no E2 owner, second budget or 28-day clock.

- MP1 run: `20261016T191445Z-localpc-mp1`.
- T0: 2026-10-17 03:14:45 SGT. New submissions/candidates stop October 23
  03:14:45; new experiments/training blocks stop October 23 19:14:45;
  hard closeout October 24 03:14:45 SGT. Restarts do not extend these times.
- New upper bounds: 48 H200 GPU-hours, 32 A40 GPU-hours, 12 new GPU PBS
  submissions per site, 24,576 optimizer updates. Native site limits win.
- Old E1 and Vanda jobs retain their original capsules, budgets and leases.
  Their reservations are not spendable MP1 balance.
- [Actual adoption and three-host readback](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json).
- [Fixed execution entry](docs/handover/2026-10-17/MAINLINE_ACCELERATION_MP1_EXECUTE.md).

MP1 has completed CPU-only source-loader qualification and one Vanda native
build. There is no new MP1 GPU submission, model invocation or physical
integration. Scientific children must pass the
existing workspace, immutable input, resource, lease and native PBS checks.
Fresh permission and capacity are checked at actual admission, not inferred
from this adoption or an old successful login.

**Campus connectivity is restored and the old E1 native states are reconciled.**
Job 670729 is F/exit 1 after the original owner-heartbeat timeout, not a
model/safety result. Both old jobs are terminal: 16,363 conservative allocated
GPU seconds charged, their old reservation released only after native proof.
Core has no active worker, outstanding physical reservation or source STOP.
The final third E1 job **671655** is Q, with 12,420 GPU seconds reserved;
used plus reserved is 28,783 of the original 28,800 seconds. No fourth E1
submission is permitted. No MP1 GPU has been submitted.
[Restored-network reconciliation and exact final allocation](reports/PI0294/20261016T150228Z-localpc-e1/NETWORK_RESTORED_AND_LAST_NATIVE_ALLOCATION.json).
The preceding interruption remains preserved:
[Actual interruption and resumption order](reports/PI0294/20261016T191445Z-localpc-mp1/LOCAL_NETWORK_INTERRUPTION_01.json).

Offline E1 checks repaired the last-allocation controller's middle-hop cache
path and bounded pipe-write timeout, plus final condition/attempt reporting.
All 228 CPU contracts and a local OS-pipe controller fixture passed; these
are not native pipeline qualification or a measured speedup. The saved M1
T1 audit metadata still reconciles to two seen-TRAIN successes, not new runs.
[Checks and exact native readmission preconditions](reports/PI0294/20261016T150228Z-localpc-e1/LAST_ALLOCATION_ENGINEERING_CHECK_01.json).
No third E1 submission, native source switch, new MP1 GPU or physical action
was made during the disconnected interval. That historical network block
is resolved, not deleted or treated as completion.
[Blocker and explicit completion gaps](reports/PI0294/20261016T150228Z-localpc-e1/OWNER_BLOCKED_NETWORK_AUDIT_01.json).
Original clocks and total caps remain unchanged. The final immutable scope
retains all M1 valid outcomes, reuses completed M2-BC weights without training,
and admits three settled engineering replacements plus the unstarted fourth
BC condition and fixed C0 M2-CORR branch. It does not hot-insert C1 data.

## Source Results

**M1@1024 autonomously completed T1 in both predeclared seen TRAIN layouts.**
Both runs lasted the full 32 seconds with Teacher OFF, original D0/S1/scorer,
no native safety fault, 41/41 terminal hold observations, and independent
raw/action/ACK/contact/score/video audit PASS. T2 was validly rejected by the
original S1 in both layouts. This is not DEV, Go2/WBC or real-robot success.

- [Complete M1 TRAIN4 and independent audits](reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json).
- Terminal E1 H200 job `670729.hopper-m-02` completed M2-BC's actual 512 updates,
  with 537 changed trainable tensors and 512 finite nonzero gradient records.
  Its three attempted evaluations are engineering-incomplete, not task outcomes;
  the first reached 25.6 simulated seconds before the original 900-second
  relay limit. M2-CORR has not yet completed. Both branches have the fixed M1@1024 parent and
  their previously sealed data choices; valid outcomes will not be rerun.
- E1 C0: 833 full H40 windows from two E0 student episodes, T1 only. It is
  state aggregation, not new M1 on-policy DAgger.
- Four real M1 collection episodes are terminal. Two required qualified
  expert takeover after actual student prefixes; two produced no correction.
  Saved-data repair produced 832 new windows, 1665 including C0, with
  independent source and actual Hopper processor/loader checks passing.
- [Repaired C1 fixed revision and actual consumption](reports/PI0294/20261016T150228Z-localpc-e1/C1_INTERVENTION_REPAIR_ACTUAL_DELIVERY.json).
- **E1 M2 still uses C0**, frozen before the repaired C1 became ready.
  The new C1 is available to a later explicitly bound MP1 branch, not hot
  inserted into E1. T2 Recovery remains unqualified (full recovery 0/2).

The original M1@256 precision comparability gap, one disclosed unintended
repeat, invalid initial fork histories, old E0 negatives and all costs remain
preserved. No claim of a flawless full E1 comparison is made.

## Target Platform

Native Vanda refresh: hardware job `1462438` is Q for node-pool availability;
dependent inference job `1462451` remains H. Failed old builds are preserved.
MP1 CPU build `1463204` is F/exit 0: container tests, native compilation,
before/after controller-source checks, and linked-library checks passed.
The simulator and two controller binary hashes are recorded in the
[actual build receipt](reports/PI0294/20261016T191445Z-localpc-mp1/VANDA_CPU_NATIVE_BUILD_ACTUAL.json).
The build used 1,533 native CPU seconds and zero GPU, physics, model queries
or WBC launches. It does not establish runtime safety or task success.
Healthy queued/held jobs were not cancelled. Container/package network
downloads were not pre-metered: the ledger retains an explicit pending
audit and 32 GiB conservative reservation, not a measured zero-transfer claim.

The old TF-R3 safety STOP is unchanged. MP1 permits only evidence-backed,
bounded owned task-interface requalification, not WBC/plant/PD/safety changes
or a cross-host bypass. Source research does not wait for target construction.

## Next Admitted Work

Finish existing E1 M2 and terminal evidence at normal boundaries. Then bind
S-BC2048 on original v3.0 TRAIN126 and the fixed M1@1024 parent; prepare a
matched S-CORR branch from qualified frozen correction data. Preserve 1:1
task exposure, including when only T1 has correction. Actual TRAIN126 has
96 T1 and 30 T2 episodes; the task-first sampler preserves 1:1 exposure.
The real source-reader and exact-cursor-resume CPU checks passed. With only
T1 correction qualified, S-CORR uses 50:50 original/correction within T1,
giving 25% correction overall, not a task-unbalanced 50% overall mixture.
Next Vanda work is runtime qualification, not another compilation-only claim.
No new DEV labels, WBC tuning or BASE padding.

The user additionally permits HF public storage for new owned artifacts.
This does not make existing private repositories public or grant rights to
redistribute private WBC/vendor assets. Credentials stay private; publication
requires sanitization and license review and does not block legal local use.
Historical selected remains NONE; real-robot deployment remains NONE.

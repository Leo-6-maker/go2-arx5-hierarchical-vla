# Current Science Status

Observed 2026-10-17 after E1 and both MP1 recovery-child lease closures by the existing single local-PC owner.
This is an actual-state summary, not a replacement for immutable run receipts.

**Latest actual qualification:** [original M2-BC512 F1/T1 full32s SUCCESS](reports/PI0294/20261016T191445Z-localpc-mp1/M2_BC_COMPLETE_T1_AND_CLOSED_LOOP_QUALIFICATION.json)
has641 camera pairs,41/41 hold,0 S1/native failures,18 independent raw checks,
and80 cross-host observation/generation/actual ACK chains with160 input-image
hashes matched. P0 is qualified for this explicitly metadata-reader-assisted
paused-sim transport, not unassisted/real-time performance. Scientific runtime
remained4d430556. Other M2 conditions remain in progress; no complete T2, DEV,
generalization, correction gain or target result is implied.

**Subsequent actual third-run progress:** [first 29.35s engineering partial and metadata observers](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_FIRST_PARTIAL_AND_DIRECTORY_OBSERVERS.json).
671899 is R; first BC F0-T2 stopped at the original 900s wall bound, not a
complete32s success despite 28/28 observed hold checks. Next BC F1-T1 is running.
Directory-only CPU observers have explicit cache observer effects and separate
sealed sources; no model/action/physics/watchdog code was hot changed. No P0
PASS or unassisted end-to-end performance claim is established by this snapshot.

**Latest direct human scope revision:** [third recovery-family submission](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_ADDITIONAL_GO_AND_NATIVE_SUBMISSION.json)
is explicitly permitted once, at most two hours, without changing MP1 total
budget/T0/deadline or prior attempt counts. Child03 passed both native entries;
671899 is Q, 7,200 GPU seconds reserved, no model loaded yet. BC F0-T1 is
excluded as exhausted; BC3/CORR4 remain eligible. Previous children01/02 stay
CLOSED and their engineering conclusions/costs remain. No fourth submission.

**Later independent native admission:** [sprint adoption and camera binding](reports/PI0294/20261016T191445Z-localpc-mp1/SPRINT_ADOPTION_AND_VANDA_CAMERA_BINDING.json)
records a new camera-only A40 job1463728, currently Q (site node_pool capacity).
This does not reopen the two closed recovery children or TF-R3 physical STOP.
No new model/physics/WBC work has run; 900 GPU seconds and 3,600 CPU seconds
are reserved until native terminal reconciliation. Runtime EGL remains unproven.

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
build, then actually ran recovery jobs 671817 F/1 and 671826 F/0 using the
existing M2 weights with zero optimizer updates. Both recovery children are
now CLOSED. Total conservative H200 cost is 453 seconds; four actual starts,
600 live plus 600 shadow integrations, five generations, three executed ACKs,
two completed draws not installed, pending/reservations zero. All four raw
evidence audits passed, but there are zero valid full policy results:
three engineering-incomplete conditions and five NOT_RUN across eight fixed
conditions. No 32s relay qualification or M2 behavioral gain is established.

The first native failure was a heartbeat-file read in the recovery entry;
the mature bounded control reader was reused in a new immutable child.
The next attempt actually located a 30.301976s stall inside Hopper RPC ledger
accounting, exceeding the unchanged 20s watchdog. Lock wait versus
read/write/fsync is not yet separated. The branch is stopped, with both of
its two PBS opportunities used; no third equivalent recovery submission.
The controller also advanced to an initial-only CORR case before the owner
stopped the cross-model branch; this charged deviation remains in the matrix.
[Actual phase timings](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ACTUAL_RPC_PHASE_TIMING.json),
[eight-condition matrix and four audited attempts](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_E1_M2_RECOVERY_BEHAVIOR_MATRIX.json),
[real qualification failure](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_REAL_RELAY_QUALIFICATION.json),
[closed child resources, not whole-MP1 closure](reports/PI0294/20261016T191445Z-localpc-mp1/E1_RECOVERY_RESOURCE_FINAL.json).
Continue independent saved-T2 diagnostics and separately qualified Vanda
work under the original MP1 clock. Do not train new weights to bypass this
evaluation fault. Fresh permission and capacity remain actual admission checks.

The subsequent registered CPU-only saved-T2 audit actually completed with
zero new physics, model queries or training. Both original M1 T2 trajectories
already entered the unchanged all-corners goal zone (F0 16.70s / F1 16.15s).
Last saved TCP-object distances were 0.04690m / 0.04240m, below the original
0.08m clearance requirement; objects remained in-zone and nearly stationary.
This supports investigating post-push disengagement, not declaring complete
T2 success or a unique collision cause. Original scores recomputed exactly.
That first saved-only receipt left the illegal geom unknown; the subsequent
separately admitted exact-history shadow result below now resolves it.
No new autonomous negative episode was run.
CPU was conservatively charged at its 600s process cap, not advertised as
exact whole-process usage. [Actual saved-state/action audit](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_SAVED_PROGRESS_AND_ACTION_AUDIT.json).

**Exact-history T2 attribution is now complete, not just prepared.** Child
`saved-m1-t2-shadow-01` reconstructed all 18,000 saved historical integrations
with zero maximum difference, including the original static IK history.
The two model prefixes reproduced their original 56/52-tick S1 rejections:
F0 table-left_pad and table-right_pad; F1 table-right_pad. Maximum forbidden
contact overlaps were 0.03726mm / 0.01903mm. These trigger the original
any-arm-table-contact rule, not the separate 4mm penetration threshold.
Both same-state expert and expert-XYZ-only 0.4s prefixes passed; SO3-only
and gripper-only substitutions still rejected. This distinguishes the tested
Cartesian path, not a unique Z bias or permission to calibrate/relax S1.
The audit-only expert remains in approach history; short-prefix PASS is not
qualified full T2 Recovery, a model-intent label or new correction supervision.
[Actual geom/component receipt](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_FIRST_REJECTION_ATTRIBUTION.json)
and [event/contact figure](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_TIMELINE_AND_CONTACT_DIAGNOSTIC.png).

The child is CLOSED with worker terminal, original input hashes unchanged
and reservations zero. New shadow cost: 19,123 integrations (18,000 exact
history + 1,123 diagnostic), Core CPU meter 189.659s including the 180s
management allowance; new GPU/PBS/model/optimizer/live/autonomous-eval zero.
Its two shadow reconstructions were conservatively charged to the MP1 source
start allowance. MP1 source-budget starts are now 6 (4 actual new evaluation
attempts + 2 shadow reconstructions), total integrations 20,323, H200 453s;
the original MP1 clock and target TF-R3 STOP remain unchanged.

**Original T2 TRAIN30 event coverage has also been actually audited.** All
30 admitted demonstrations / 30 distinct scene groups (F0/F1/F2 = 8/11/11,
not 30 OOD categories) contain every original push stage. Their 18,030
overlapping H40 anchors include detach 756 (4.19%), withdraw 1,500 (8.32%),
and 1,926 cross-stage windows containing detach. No window lies entirely in
detach because those segments last only 1.10-1.35s, shorter than H40's 2s;
this is not missing detach supervision. Raw stages exactly match the frozen
manifest buckets. All 640 supervised commands plus the unlabeled terminal
observation were explicitly distinguished and original input hashes rechecked.
[Actual episode/event/window coverage](reports/PI0294/20261016T191445Z-localpc-mp1/T2_TRAIN30_EVENT_AND_H40_COVERAGE.json).

The fixed Source Pair sampler forecast gives 1,324 detach + 2,678 withdraw
visits among 32,768 planned T2 visits per branch. This is a deterministic
future schedule, not actual training consumption or evidence of sufficiency.
No S-EVENT trigger or main recipe change follows automatically from these
counts. One CPU audit failed its initial 640-versus-641 observation check;
the bounded explicit-terminal repair passed. Both workers are terminal,
charged at 600 CPU seconds each as conservative process upper bounds,
reservations zero; new physics/neural/training/GPU/PBS zero.

**The measured RPC accounting hot path has a scoped code repair.** Runtime
`4fda127a` keeps successful-reply telemetry in memory and settles RPC count
plus process CPU once at exit; neural/optimizer/physics charges, journals,
guards and original 20/60/900s contracts stay durable and unchanged.
Poisoned channels now publish a separate engineering stop on both child
roots; native recovery/Source Pair guards reject it before the next model.
An existing marker prevents new panel binding. This is not a native safety
STOP and does not relabel previous policy or engineering outcomes.

Both registered Core/Hopper CPU probes actually held a synthetic ledger lock
in another process while emitting 80 memory replies, then made one final
shared-ledger transaction. All workers and lock holders are terminal, and
uncertain/repeated settlement is rejected. Native probes use no real model,
neural/physics counter or RPC wire; they prove this telemetry-lock isolation,
not actual SSH/NFS/32s qualification or the cross-model model-stop behavior.
Heartbeat and guard reads still do I/O; complete-loop speedup remains unknown.
A killed session requires registered CPU-bound reconciliation, not zero cost.
[Implementation, actual probes and limits](reports/PI0294/20261016T191445Z-localpc-mp1/RPC_REPLY_LOCK_ISOLATION_CPU_RECEIPT.json).

The first CPU probe on each site failed before lock setup on a Path expression;
the explicit role-entry test repair passed on the sole bounded CPU retry.
All four attempts are terminal and conservatively charged 600s/site/attempt
(1,200s per site), reservations zero. Native queue recheck finds no own
PI0294 running/queued GPU job; old recovery leases remain CLOSED, 453 H200s
unchanged, no third recovery PBS. A new legal end-to-end route is still required
before claiming P0 or a completed M2 behavior comparison.

**Campus connectivity is restored and the old E1 native states are reconciled.**
Job 670729 is F/exit 1 after the original owner-heartbeat timeout, not a
model/safety result. Both old jobs are terminal: 16,363 conservative allocated
GPU seconds charged, their old reservation released only after native proof.
The final third E1 job **671655** is now F/exit 0. All three allocations
are natively reconciled: 19,675 conservative GPU seconds charged and zero
GPU reservation. No fourth E1 submission is permitted even with unused
seconds. All source/neural/controller/stdio workers were terminal and
physical reservations zero before the separate CPU artifact-closeout stage.
Both E1 leases and registry entries are now
CLOSED after native/resource/artifact readback. E1 is a bounded closed
execution with explicit scientific comparison gaps, not a successful full
M2 behavioral comparison or a reopened resource window.
[Final resources and closed leases](reports/PI0294/20261016T150228Z-localpc-e1/RESOURCE_FINAL.json).
[Complete execution summary](reports/PI0294/20261016T150228Z-localpc-e1/E1_RUN_SUMMARY.md).
[Actual final allocation, training and consumed exposure](reports/PI0294/20261016T150228Z-localpc-e1/LAST_NATIVE_TERMINAL_AND_ACTUAL_M2_EXPOSURE.json).
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

**BC recovery did not produce a valid comparison.** Its first replacement
stopped after 6 simulated seconds on a 20-second file-RPC timeout; a late
response was then incorrectly reused on the same channel, causing the next
three conditions to fail response-ID checks before physical integration.
All four remain engineering failures, not valid policy negatives. No more
BC replays are made in E1. Both SSH file endpoints subsequently exited.
CORR completed its fixed 512 updates/16,384 samples in the same allocation,
with 537 changed trainable tensors, frozen parameters unchanged, and 512
finite positive gradient records. Its first autonomous case stopped at
10.8 simulated seconds on another original 20-second RPC timeout, with
zero S1 rejection/native fault. The poisoned-channel latch then prevented
the other three physical cases from starting: NOT_RUN, not three failures.
No valid complete M2 comparison exists. Both model weights are preserved.
The single local controller was replaced only at the settled BC/active
CORR-training boundary, using the same owner lock and original deadlines.
An uncertain channel now latches invalid before any later physical case.
The finite direct-open metadata helper actually ran and ended, costing
1.160 CPU seconds, no model queries or physics. It did not establish a
complete pipeline repair; original timeouts and scientific sources stayed
unchanged. Independent main-evidence audit accounts for all 22 actual
attempts, including three zero-integration engineering attempts. Audit
PASS is not task success or complete policy-outcome coverage.
[Actual failure chain, timing caveats and controller continuity](reports/PI0294/20261016T150228Z-localpc-e1/BC_REMAINDER_ENGINEERING_FAILURE_AND_CORR_CONTINUITY.json).

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
  relay limit. M2-CORR training is now complete but its autonomous panel is
  engineering-incomplete. Both branches have the fixed M1@1024 parent and
  their previously sealed data choices; valid outcomes were not rerun.
- Actual consumed traces verify BC T1/T2=8192/8192 and CORR=12288/4096.
  CORR consumed 8192 original and 8192 C0 samples, from six correction
  segments/two independent student parents. Equal updates are not
  task-matched exposure; no causal correction-gain claim is supported.
- All 12 predeclared terminal-model conditions are explicit in the
  [actual behavior matrix](reports/PI0294/20261016T150228Z-localpc-e1/MODEL_BEHAVIOR_MATRIX.json):
  four valid M1 policy outcomes, five engineering-incomplete conditions,
  three NOT_RUN conditions. All 22 actual attempts have accounted evidence,
  but not all have valid task results. The six fixed probe contexts now have
  648 original-prefix S1 shadow checks, without new neural queries.
- On the noninitial fixed student states, CORR T1 prefix8 XYZ error is
  2.19 mm versus M1 6.90 mm, with original S1 prefix execution 6/6 versus
  4/6 across all fixed student states. These overlap C0's same two TRAIN
  parents and do not prove independent generalization or autonomous gain.
  [Actual student-state diagnostics](reports/PI0294/20261016T150228Z-localpc-e1/ON_STUDENT_DIAGNOSTIC_COMPARISON.json).
- [Saved M1 T2 object progress and original final rejection](reports/PI0294/20261016T150228Z-localpc-e1/M1_T2_SAVED_REJECTION_AND_OBJECT_PROGRESS.json):
  actual horizontal motion 13.28/14.61 cm before original S1 rejection,
  without lifting; no complete task success. That historical saved-only
  receipt did not know the specific illegal geom pair and did not inherit it
  from M2048. The separately charged MP1 shadow receipt above now resolves
  those pairs; the old E1 evidence and result remain unchanged.
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

All four fixed inference checkpoints and 38 decoded main/fork clips now
have immutable public HF revisions and READY readback. The private upload
hit quota and its empty prefix was reconciled; the private repository was
not made public. Only two Cosmos metadata paths were sanitized; weights
and statistics stayed unchanged, with complete upstream license notices.
Public payloads exclude optimizer/RNG and private paths/assets. M1@1024
and both M2 full native training states remain on Hopper; M1@256 was
inference-only originally. Public inference/resume qualification is not
claimed. [Actual fixed artifact index](reports/PI0294/20261016T150228Z-localpc-e1/ARTIFACT_DELIVERY.json).

## Target Platform

Native Vanda refresh: hardware job `1462438` and saved-input inference job
`1462451` are F/exit 0, with CLOSED leases. Native allocated GPU walltime is
58 and 137 seconds respectively, charged to their old scope, not MP1.
The former passed actual A40 CUDA checks; the latter loaded old M2048 and
made 17 saved-observation queries (one warmup, 16 timed), median 0.2356s,
peak allocated 13.35 GB. This is neither a live loop nor H200/A40 numerical
equivalence, and does not qualify M1/M2 or native EGL rendering.
[Original native terminal receipts](reports/PI0294/20261016T191445Z-localpc-mp1/PRIOR_VANDA_ENGINEERING_NATIVE_TERMINALS.json).
Failed old builds are preserved.
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

After E1 closure, GitHub CI at d30354e5 failed four fixtures because pytest
loaded a second module alias and string-based mocks patched the other alias.
Tests now patch their actual imported module objects. Remote CI at
0218b645 passed; default local pytest is 827 PASS/14 explicit SKIP, PI0292
108 PASS and PI0294 243 PASS. No native scientific capsule was hot patched.

An earlier MP1 engineering action actually ran on both hosts: 96 same-host
synthetic packet/heartbeat/ledger filesystem cycles each, with no model,
PBS or physical start. The maximum component latency was 15.74ms Core
and 24.71ms Hopper. No 20s stall was reproduced; this is not a cross-node
NFS, SSH or 32s closed-loop test. Direct open is not a proven speedup.
Guard/invoke/accounting/output phase tracing was then used by the two real
recovery children and located the accounting stall described above, without
watchdog or mutation-retry changes.
Initial CPU entry/path errors and a pre-dispatch immutable-request collision
are preserved; failed cost is conservatively charged, not silently zeroed.
All probe processes are terminal, reservations settled, E1 still CLOSED.
[Actual timings, limitations, CI and costs](reports/PI0294/20261016T191445Z-localpc-mp1/ACTUAL_RELAY_CPU_DIAGNOSTIC_01.json).

E1 cannot submit again or reopen its closed leases. Continue within the
already adopted MP1. Both existing-model recovery PBS slots are now spent;
the failed shared relay branch is stopped, not queued for a third equivalent
attempt. A future usable evaluation route still requires actual qualification,
not another same-host smoke or training new weights to bypass the fault.
No valid M1 outcome is replayed.
Then bind S-BC2048 on original v3.0 TRAIN126 and the fixed M1@1024 parent;
use the matched S-CORR branch from qualified frozen correction data. Preserve 1:1
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

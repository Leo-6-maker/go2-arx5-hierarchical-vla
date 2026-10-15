# Task-data-first adoption and current execution boundary

Observed 2026-10-15 14:06:32 SGT. Direction release:
`4a7187e3ac4a1f57a4e696815a1e0d3df74d15c8`, #11 comment6185220209.

## Implemented locally

`scripts/pi0294/task_capture_scope.py` supplies a versioned policy for a NEW
task-capture adapter. Historical `VALID_NATIVE_HOLD_QUALITY_FAIL` is classified
as `DIAGNOSTIC_ONLY_HISTORICAL_WBC_HOLD` only in the new scope. Legacy scope
remains blocked. Safety and actual task success remain independent hard gates.
Nine CPU unit tests passed, including each missing/failed hard check, expired
resource, unsuccessful task, executor mismatch and preservation of old verdicts.

This is a tested policy component, **not a deployed collector or launch receipt**.
It does not grant resources or invoke a process. The future worker must first
validate the actual standard manifest/profile/owner/lease/output and reserve a
ledger ticket through the deployed entry. Readiness cannot substitute for those
checks. The old hold launcher, sentinel and reports are untouched.

R4B D0 is sealed. No D1, additional hold attribution, WBC change or compensation
is scheduled. No controller owner response or RMS PASS is required to proceed.

## Actual resource refresh

Both servers were reachable. Per-user process inspection found no matching
PI0294/native simulator/MPC/controller processes on either inspected host.
Both lease files still say ACTIVE, but their exact deadline is past; a stale
status word does not create a valid window.

| Item | Actual |
|---|---|
| Original new-work cutoff | 2026-10-15 12:09:30 SGT |
| Original closeout deadline | 2026-10-15 13:09:30 SGT |
| Native task processes / CPU reservations | 0 / 0 |
| Main starts charged | 20 / 160 |
| Main integrations charged | 234331 / 2560000 |
| RGB pairs charged | 8974 / 102560 |
| Core CPU | 11962.778267704352 seconds / 96 core-hours |
| Hopper CPU | 0.67567 seconds / 32 core-hours |
| R4 PBS / GPU allocated / formal updates / generation intents | 0 / 0 / 0 / 0 |

The new direction explicitly grants no replacement clock (protocol sections 1,
6 and 7). No expired native or model run was restarted. All existing counters
remain cumulative, including failed engineering attempts.

## Task integration: source-based readiness, no inflated completion claim

| Component | Evidence / next exact work |
|---|---|
| Frozen full robot and native WBC | Actual prior native run; preserve free root, 12 active legs, 6 arm joints, safety and existing clock/receiver revisions |
| EEF13 + optional BASE soft reference | Existing `base_intent.py` / `native_receiver.py`; task travel envelope and actual stop/expiry/reset still need task-level acceptance |
| Finite gripper and task objects | Existing fixed-base task assets are a reusable specification, not an accepted full-body plant; create an isolated additive overlay with explicit gripper mass/inertia/contact identity |
| Moving body / wrist cameras | Must mount to actual moving body/link; fixed-base world camera is not a body camera; save calibrated true image timestamps |
| Teacher | Reuse task logic against measured TCP, object and BASE state; submit EEF/gripper/BASE references, never directly drive arm joints or calibrate a WBC bias |
| Native full-body worker | NOT_IMPLEMENTED; old `scene.make_engine` removes free root/leg joints and changes arm dynamics, so it cannot be relabeled as the new worker |
| Recorder | Preserve actual input versus committed intent versus achieved execution; privileged stages/object truth remain audit/Teacher-only |
| Dataset / FB13 | No accepted full-body pilot, v4.0 dataset or FB13 training yet; M2048 remains fixed-base EEF10 initializer |

The old fixed-base Teacher also uses narrow pose-error conditions to advance
some stages. Do not silently reuse those as a new global WBC gate or delete
them after observing task scores: bind the full-body task Teacher's stage
predicates and actual object success rules before its formal pilot. Functional
grasp/push tolerances remain task requirements, distinct from a hold RMS test.

## One combined continuation package (proposal only)

See `CONTINUATION_CARD.json`. It requests only a replacement execution window,
retaining original cumulative resource ceilings and already charged usage.
It does not reactivate the old lease. No further WBC diagnosis is included.

Sequence: additive task plant/recording acceptance and stop/reset checks → four
declared paired T1/T2 pilots, all attempts counted → qualified 2 then at most 4
isolated workers → frozen v4.0 24TRAIN + 8DEV, TRAIN-only codec/statistics →
conditional FB13 warm-start from M2048, at most 2048 updates under the original
recipe → original fixed task evaluation and archives. Continue independently
qualified tasks while preserving failed scenes and missing-skill denominators.
Do not claim two-task training if one skill has no accepted demonstrations.

Do not expand to v4.1 now: ten historical formal TRAIN attempts already consume
the shared 48-attempt cap. A new 24-TRAIN bootstrap leaves 14, not 24, additional
TRAIN attempts under that cap. Do not reset the count or replace failed seeds.

This publication adds zero remote physics, rendering, neural/gradient queries,
optimizer updates, PBS calls, WBC changes, payload deletions or public raw files.
`selected_checkpoint=NONE`; `deployment=NONE`.

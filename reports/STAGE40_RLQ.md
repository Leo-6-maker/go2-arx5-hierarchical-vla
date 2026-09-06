# Stage40-RLQ: Native Reference Lifecycle Requalification

Status: PRE_CANARY  
Scientific evidence: none  
Stage40 B1/C0/C: locked

## Frozen baseline

- parent HEAD: a8e6a8531f855247a17542744a011e5003595959
- parent branch: stage40/live-one-handoff-tiny1
- RLQ branch: stage40-rlq/reference-lifecycle
- scientific inputs, runtime, controller and SafetyGate unchanged

## RLQ-A audit

The native environment installs the initial reference during construction/reset:

_build_native -> EnvClass -> env.reset -> _load_trajectory_for -> _stage21_load_chunk(initial=True)

This leaves stage21_current_chunk populated before the first policy step. The previous B1 path then called load_reference_chunk(chunk_a) immediately. That API intentionally rejects reset_buf == true, so 603997 stopped before _step_rows and before the first WBC forward.

## Two-phase contract

- initial_bind_reference(chunk): permitted only before the first policy forward; preserves root state, DOF state, episode counter and policy history; never clears reset_buf and never calls simulator reset.
- load_reference_chunk(chunk): permitted only after a policy forward and only while reset_buf == false; classified as RUNNING_UPDATE.

The shared _policy_step records the first real actor forward so the phase boundary is explicit.

## Code changes on this branch

- Added explicit startup/running lifecycle state and counters.
- Added physical payload equivalence check for the already-installed initial chunk.
- Updated future model/expert startup replacement sites to use initial_bind_reference.
- Updated the diagnostic B1 path to consume the native initial chunk instead of duplicating its load.
- Added a bounded, non-GR00T RLQ canary and PBS wrapper.

## Static qualification

- Python compilation: PASS
- PBS shell syntax: PASS
- Stage40 runtime contract unittest: PASS (4/4)
- pytest: unavailable on login node (pytest module not installed)

## Next authorized step

Run exactly one bounded H200 RLQ canary. It must report:

1. initial bind payload equivalence and no-state-change invariants;
2. natural reset_buf transition without manual clearing;
3. one finite native step after initial bind;
4. running-state update with no root/DOF/episode/history change;
5. one finite native step after the update.

A canary PASS only establishes REFERENCE_LIFECYCLE_REQUALIFIED; it does not reopen Stage40 B1.

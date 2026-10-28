# VAP1: Long Allocations By Default

Status: **ADOPTED_USER_SCHEDULING_DIRECTION**, not a native submission receipt.
The user explicitly directs future A40 work to use long allocations, citing
previous successful long-duration A40 training. This changes scheduling, not
the VAP1 scientific comparison, resource allowance or original clock.

Machine policy: [VAP1_LONG_ALLOCATION_SCHEDULING_20261021.json](../../../reports/PI0294/design/VAP1_LONG_ALLOCATION_SCHEDULING_20261021.json).
Parent authority: `43f7fe6f86c683f2e38e5c8802eb8cde208e575c`, section G1.

## Required Scheduling

- Prepare a complete arm's work before requesting its long allocation: model,
  immutable datasets, official readers, source CI, permissions, output space,
  full-state saves, fixed probes, training reentry and closeout. Reuse verified
  assets rather than download them again after allocation.
- Default to one allocation targeting all 4096 updates of an arm. Do not stop
  solely at update 1024 and requeue. Request the useful bounded walltime, up to
  the original eight-hour or stricter site limit, within actual remaining costs.
- Profile the first 32 real A_CONTROL updates inside its 4096-update total.
  Continue the same state after qualification; do not create another short
  GPU job or discard counted updates. Historical A40 feasibility is not a
  reason to skip this recipe's runtime memory/finite-gradient checks, nor may
  H200 timing be relabeled as A40 timing.
- At 1024 save and verify full native state, execute the original 12 TRAIN
  probes, then continue to 4096 inside the same allocation when the remaining
  envelope permits. No update 1025 before the fixed probe receipt. Preserve
  optimizer, scheduler, RNG, sampler cursor and unchanged recipe across probes.
- A and B may run sequentially in one predeclared allocation only if the full
  measured time/resource/save envelope fits. Otherwise use one long allocation
  per arm, not four allocations for two logical phases each. Both arms start
  independently from the common CORR parent; B never inherits A's weights.
- Reserve save/closeout time and stop on original resource, safety, integrity
  or clock bounds. Resume only from the latest complete same-arm state after
  effect reconciliation; do not rerun unknown-effect work.

## Existing Work And Implementation Boundary

Keep job `1471903.stdct-mgmt-02` unchanged: its immutable scope remains the
17-draw, 30-minute saved-TRAIN compatibility check. This decision neither
expands it in flight nor authorizes cancelling a healthy job to replace it.

The existing cold preparation and forecast still describe split phases, and
the native A40 long-training launcher/reentry contract is not yet implemented
or deployed. Those old prepared inputs are not a compliant new long-job
binding. Implement and test the cold contract before the next training
submission; do not merely change `stop_after_step` to 4096 and silently omit
the 1024 probes. Publication is not runtime acceptance or a training result.

No additional GPU/PBS budget, second GPU, owner, clock, training updates,
physics, WBC/S1/scoring change or LOCKBOX permission is created. Original
R0/data/recipe, final reserve and `selected/deployment=NONE` remain unchanged.

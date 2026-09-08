# Stage28 evaluator recovery Phase A / A.1

Status: `CORELAB_SOURCE_RECOVERY_PHASE_A1_COMPLETE`

This recovery is bound to `PI_GATE_STATE_0008` from Issue #11.  It is a
source/provenance-only change based on the historical Stage28 training source
checkout `a361e5cb8a4be33f40a21c9bbf7ad742e25c7fd0`.  The historical H2 training
job `600695.hopper-m-02` was not rerun.

## Provenance bindings

- Stage26 manifest SHA: `eb33c81b68bb8bd49ce0ed9b0b818fdbb29546b49a2d79a3ee62845e172afb7b`
- H2 checkpoint-2000 tree SHA: `3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba`
- H2 checkpoint seal SHA: `1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82`
- Historical training runner SHA: `d61c429a2b8d53c25df63d9b5b9e2464b33c36e9e3c0a610b109b10a6bd63fe4`
- Recovery evaluator SHA: `17827a220cd4a7fd0668b15161a3ac80f5bde6f7cc05d53d044124d362a3c03c`
- Evaluation-only launcher SHA: `172fbed3c5b7897fda2283752beaf875db820f3a00b53a916737e0fb8a2387b2`
- Evaluation-only PBS template SHA: `b34cf9a40cf37b4b82ded4c7320b797b87d41eb8e9f94f38f06bfc5ca3c71329`

## Changes

The Stage26 loader now reads the complete 18-row manifest, filters rows by
`checkpoint == BASE`, requires exactly nine BASE rows, and requires precisely
the fixed probe IDs `c1_seed{2223,2224,2225}_tq{10,50,90}`.  Duplicate, missing,
foreign, malformed, count, and manifest-SHA failures remain fail-closed.  The
historical manifest is read-only and was not rewritten.

The new `scripts/stage28_h2_evaluation_only.py` entry point is independent of
the historical train-coupled wrapper.  It accepts no training mode, binds the
current PI gate and all historical identities, and delegates only to the
fixed-probe evaluator.  `hpc/pbs/stage28_h2_evaluation_only.pbs` is a separate
evaluation-only template; it does not create or modify checkpoint directories
and was not submitted during this Corelab recovery.

The H2 checkpoint tree and seal are checked before the runtime processor/model
construction path.  The evaluation result records the historical training
provenance separately from the evaluation runner provenance.

## Phase A.1 closure

After BASE x9, the evaluator now computes the exact Stage26 hash reproduction
before the H2 checkpoint flow path is invoked.  A mismatch writes
`stage28_h2_base_hash_mismatch.json` and the BASE raw-hash evidence, records all
BASE rows and exact expected/observed differences, and stops with zero H2 flow
evaluations.  The H2 path is reachable only after an exact `MATCH`; successful
results require all 18 paired records.

The evaluation-only launcher now requires and records the actual `PBS_JOBID`
and project `CFP03-CF-004`; real evaluation results report
`pbs_submitted: true`.  The emitted next gate is
`GPT_PI_REVIEW_AFTER_STAGE28_FIXED_PROBE_RESULT`.  The historical training PBS
wrapper remains unchanged and is not reused by the recovery path.

## Validation

- 25 Stage26/Stage28 targeted unit tests: PASS
- `py_compile`: PASS
- `compileall`: PASS
- PBS shell syntax: PASS
- `git diff --check`: PASS
- Stage28 CPU self-test: PASS

No model was constructed, no forward was run, no GPU or PBS job was used, and
no training, backward, optimizer, scheduler, resume, or checkpoint write was
performed in this recovery phase.

Scientific contract: unchanged.

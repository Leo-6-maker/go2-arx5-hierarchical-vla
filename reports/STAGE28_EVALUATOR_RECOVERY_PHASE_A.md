# Stage28 evaluator recovery Phase A

Status: `CORELAB_SOURCE_RECOVERY_COMPLETE`

This recovery is bound to `PI_GATE_STATE_0008` from Issue #11.  It is a
source/provenance-only change based on the historical Stage28 training source
checkout `a361e5cb8a4be33f40a21c9bbf7ad742e25c7fd0`.  The historical H2 training
job `600695.hopper-m-02` was not rerun.

## Provenance bindings

- Stage26 manifest SHA: `eb33c81b68bb8bd49ce0ed9b0b818fdbb29546b49a2d79a3ee62845e172afb7b`
- H2 checkpoint-2000 tree SHA: `3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba`
- H2 checkpoint seal SHA: `1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82`
- Historical training runner SHA: `d61c429a2b8d53c25df63d9b5b9e2464b33c36e9e3c0a610b109b10a6bd63fe4`
- Recovery evaluator SHA: `25707eb545921047cf7df623a66d6b63af4bb146c907f2cfa40f4a855fb1e6dd`
- Evaluation-only launcher SHA: `b70d5999aa823da64892d182880e779c1aee61a4f2e10070d1801496c3d05478`
- Evaluation-only PBS template SHA: `290f36b13f757c567ae0386644a43ef76d94c426ab3c842358ac9717a97ab0fe`

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

## Validation

- 23 Stage26/Stage28 targeted unit tests: PASS
- `py_compile`: PASS
- `compileall`: PASS
- PBS shell syntax: PASS
- `git diff --check`: PASS
- Stage28 CPU self-test: PASS

No model was constructed, no forward was run, no GPU or PBS job was used, and
no training, backward, optimizer, scheduler, resume, or checkpoint write was
performed in this recovery phase.

Scientific contract: unchanged.

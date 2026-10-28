# PI0294 v3.1 Incremental Handoff

This is an engineering transport clarification, not a new scientific GO or an
actual v3.1 export, HF delivery, Hopper qualification, or model execution receipt.
The original clocks, budgets, splits and first-attempt outcomes remain fixed.

The frozen v3.1 plan includes the original new v3.0 80 attempts plus 144 extension
attempts. `scripts.pi0294.hf` packages only extension raw archives for v3.1. It
references the actual v3.0 READY-last revision
`6aa94f4f40d791bedd6b4e75489e4ede6504d10d` through
`PREVIOUS_BANK_DEPENDENCY.json`, including both rejected v3.0 attempts. Original
PI0292 64 TRAIN/16 DEV remain a separate unchanged dependency.

Before READY, the publisher checks the previous delivery's exact remote file
set, sizes and hashes at its fixed revision. A missing archive, changed outcome,
changed split, mutable revision, different run, or changed receipt is rejected.
No old raw archive is repacked or uploaded again. The new native numeric dataset
and its version-specific TRAIN-only statistics remain separate from v3.0.

For v3.1 reception, `scripts.pi0294.hf --mode download` additionally requires
`--previous-native` and `--previous-bindings` from the existing qualified local
v3.0 receipt. Every reused raw archive member is hashed before local binding;
the current new numeric files are independently checked. Missing or changed
local inputs cause failure, not an implicit repeat download. Existing v3.0
reception semantics are unchanged.

Use the existing admitted CPU launcher and fixed source sidecar to invoke the
module. Do not install this change into an active collector or model capsule.
Neither dependency reuse nor CPU fixtures grant Hopper model qualification, add
v3.1 samples to active v3.0 training, open confirmation, or change selection.

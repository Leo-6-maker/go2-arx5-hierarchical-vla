# STAGE22 H2 Run Integrity Contract V1

H2 is the authoritative matched experiment:

```text
same H200
same Stage10/GR00T runtime
same base checkpoint
same train-only dataset
R0 @ 50 steps  vs  R1 @ 50 steps
```

The run-integrity auditor is
`scripts/stage22_audit_h2_run_integrity.py`. It audits the generated
`STAGE22_RECIPE_COMPARISON_V1` manifest before any profile result is cited.
It does not select a checkpoint and does not inspect the locked final-test
split.

## Required invariants

- exactly one full `r0` result and one full `r1` result;
- both runs pass and use simulation-only mode with hardware control disabled;
- `max_steps=50`, matched save interval, learning rate, seed `2222`, and
  `--local-files-only`;
- exact train-only targets `red_target_train_a/b/c/d`;
- identical base-model path and non-empty base index SHA256;
- identical dataset path, deterministic dataset-tree SHA256, split-lock SHA256,
  and GR00T repository SHA;
- `action_horizon=40` and identical test-only `0.05 s` action semantics;
- R0 has projector training with diffusion/visual/LLM frozen;
- R1 has projector and diffusion training with visual/LLM frozen;
- R1 trainable parameter count is `1,419,078,784`;
- matching Python, CUDA, PyTorch, device identity, and total device memory;
- a recorded GPU fingerprint with UUIDs is present for the H2 process.

Failure of any required invariant yields `FAIL` and blocks the H2 scientific
comparison. A successful integrity audit does not imply that either recipe
passes the held-out action-profile gate.

## Usage

```text
python scripts/stage22_audit_h2_run_integrity.py \
  --comparison /path/to/stage22_recipe_comparison.json \
  --output /path/to/h2_integrity_audit.json
```

All paths and manifests are node-local experiment artifacts. They must not be
committed when they contain model weights, private runtime paths, or logs.

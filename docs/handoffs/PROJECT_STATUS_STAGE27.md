# Project Status — Stage27 H2 optimization semantics

Date: 2026-09-04 (Asia/Singapore)

```text
ROLE: CODEX_CORELAB_LONG_HORIZON
STAGE: Stage27
BRANCH: stage27/h2-optimization-semantics-audit
BASE: 735afb7f17f0184ca9eff3e372dfd59c8e008e95
RESULT: PASS
NEXT_GATE: GPT_PI_REVIEW_AFTER_STAGE27_H2_SEMANTICS_AUDIT
```

Stage27 is a CPU/static metadata and pinned-source audit. It did not import
torch/transformers/safetensors, construct a model, read tensor payloads,
invoke forward/inference/training, use GPU/PBS, or read validation `e` or
holdout `f`.

## Findings

The exact Stage23 launcher used `save_only_model=True` and
`resume_from_checkpoint=False`. The sealed `candidate_200` artifact inventory
has no optimizer, scheduler, RNG, scaler, or dataloader/sampler state. It does
have `trainer_state.json` with `global_step=200`, but that file contains
Trainer metadata and LR log history rather than continuation state. BF16 was
enabled and FP16 was disabled, so a GradScaler is not applicable. The pinned
entrypoint rejects the model-only resume combination before model construction.

The audit's schedule implementation matches all 200 sealed O1 learning-rate
records exactly. `warmup_ratio=0.05` gives:

```text
O1@200:       warmup=10,  cosine denominator=190
fresh @2000:  warmup=100, cosine denominator=1900
fixed warmup: warmup=10,  cosine denominator=1990
```

Fresh BASE→2000 with the original ratio differs from O1 at displayed step 2;
only displayed step 1 has the same zero LR. This is an explicit schedule
confound.

## Corelab recommendation for PI review

The least-confounded stock-path candidate is:

```text
H2_FRESH_BASE_2000_FIXED_WARMUP10
```

This avoids pretending that O1 can be exactly resumed, preserves the original
warmup length, and runs one coherent fresh schedule. It still changes the
cosine time scale, so it tests a fresh 2000-step adaptation recipe and cannot
claim a pure “additional 1800 steps” effect. An O1-weight warm-start would
preserve the O1 endpoint but reset optimizer/scheduler/RNG/sampler state and
is secondary. A prefix-preserving custom schedule would need a separate gate.

## Scope and stop

```text
SCIENTIFIC_CONTRACT_CHANGED: false
DOWNSTREAM_AUTHORIZATION_STILL_VALID: true
HOPPER_H2_TRAINING: HOLD / no authorization token granted
GPU / PBS / model / forward / backward / optimizer: false
VALIDATION_E / HOLDOUT_F: false
C2 / WBC / hardware / ROS / ZMQ / P1M: false
```

Corelab stops and returns to GPT/PI review. See
`reports/STAGE27_H2_OPTIMIZATION_SEMANTICS_AUDIT.md` and its JSON companion.

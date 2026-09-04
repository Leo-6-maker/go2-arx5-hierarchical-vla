# Stage27 — H2 optimization-semantics audit

Date: 2026-09-04 (Asia/Singapore)

## Return

```text
ROLE: CODEX_CORELAB_LONG_HORIZON
STAGE: Stage27 H2 optimization-semantics audit
BASE: 735afb7f17f0184ca9eff3e372dfd59c8e008e95
RESULT: PASS
MODEL_CONSTRUCTED: false
TENSOR_PAYLOAD_READ: false
GPU / PBS / FORWARD / TRAINING: false
NEXT_GATE: GPT_PI_REVIEW_AFTER_STAGE27_H2_SEMANTICS_AUDIT
SCIENTIFIC_CONTRACT_CHANGED: false
DOWNSTREAM_AUTHORIZATION_STILL_VALID: true
HOPPER_H2_AUTHORIZATION: HOLD / no token granted
```

The machine-readable companion is
`reports/STAGE27_H2_OPTIMIZATION_SEMANTICS_AUDIT.json`; the reusable audit is
`scripts/stage27_h2_optimization_semantics_audit.py`.

## Decision-ready answers

### 1. Exact resume

`O1@200` cannot be exactly resumed from the sealed checkpoint.

The exact Stage23 wrapper (`stage20s_train_gr00t.py`, SHA256
`f42a1990847cf8f5326c66db37043ffed6e43458dd907fdbd08da5dd905b00a2`) set:

```text
save_only_model = true
resume_from_checkpoint = false
```

The Stage23 `candidate_200` manifest enumerates the checkpoint files and
contains no `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, or scaler state.
The sealed `trainer_state.json` has `global_step=200` and `max_steps=200`, but
only Trainer metadata, log history, and `TrainerControl`; it has no embedded
optimizer, scheduler, RNG, or scaler state. BF16 was enabled and FP16 was
disabled, so an FP16 GradScaler is not applicable.

The pinned `experiment.run` calls `check_resume_compatibility` before pipeline
and model construction. Its source explicitly rejects
`save_only_model=True` together with `resume_from_checkpoint=True` because the
checkpoint lacks optimizer/scheduler/RNG state. Loading O1 weights into a new
run would therefore be a weight warm-start, not an exact continuation.

The custom `Gr00tTrainer` also sets `ignore_data_skip=True` and, on resume,
resets the dataset seed to `dataset.seed + global_step` while logging that the
operation is non-reproducible. No dataloader/sampler state is serialized in the
O1 artifact. This is an additional reason not to label a weight-only restart
as exact continuation.

### 2. State inventory

| State | Evidence | Result |
|---|---|---|
| model weights/index | historical Stage23 seal; shard bytes are not in the Corelab relay | present historically, not read here |
| `trainer_state.json` | SHA256 `2aa506bf605bce28809739970b6d3bc86059dba5490101c7da4fdba94512e4d9` | present; `global_step=200` only |
| optimizer moments | manifest/seal artifact enumeration | absent |
| LR scheduler state | manifest/seal artifact enumeration | absent |
| RNG state | manifest/seal artifact enumeration | absent |
| FP16 scaler | `bf16=true`, `fp16=false`; manifest/seal enumeration | not applicable / absent |
| dataloader/sampler state | pinned custom trainer source and artifact enumeration | absent; resume reseeds instead |

### 3–5. LR schedule semantics

The pinned recipe is AdamW (`adamw_torch`), base LR `1e-4`, cosine scheduler,
`warmup_ratio=0.05`, `warmup_steps=0`, batch 1, and gradient accumulation 1.
With no explicit warmup-step override, the pinned TrainingArguments semantics
are:

```text
warmup_steps = ceil(max_steps * warmup_ratio)
cosine denominator = max_steps - warmup_steps
```

The displayed Stage23 Trainer step `s` matches the cosine lambda at scheduler
step `s-1`. The implementation reproduces all 200 sealed LR log rows exactly
(maximum absolute error `0.0`).

| displayed step | O1@200 LR | fresh BASE→2000 with ratio .05 | fresh BASE→2000 with explicit warmup 10 |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 1.000000e-5 | 1.000000e-6 | 1.000000e-5 |
| 10 | 9.000000e-5 | 9.000000e-6 | 9.000000e-5 |
| 11 | 1.000000e-4 | 1.000000e-5 | 1.000000e-4 |
| 12 | 9.9993165e-5 | 1.100000e-5 | 9.9999938e-5 |
| 100 | 5.4952277e-5 | 9.900000e-5 | 9.9507281e-5 |
| 200 | 6.8347504e-9 | 9.9331605e-5 | 9.7790811e-5 |
| 1000 | — | 5.4211353e-5 | 5.0473600e-5 |
| 1900 | — | 6.9560985e-7 | 6.3424353e-7 |
| 2000 | — | 6.8349045e-11 | 6.2306522e-11 |

Therefore:

```text
O1@200:  warmup=10, cosine denominator=190
ratio .05 at 2000: warmup=100, cosine denominator=1900
explicit warmup=10 at 2000: warmup=10, cosine denominator=1990
```

A fresh BASE→2000 run using the old ratio is not O1-prefix-equivalent: only
the displayed zero-LR step 1 is equal; steps 2–200 differ. This is a schedule
confound, not evidence about model quality.

### 6. Least-confounded H2 candidate

For a single coherent stock-path run, the least-confounded candidate is:

```text
H2_FRESH_BASE_2000_FIXED_WARMUP10
initialization: exact fresh BASE
max_steps: 2000
learning_rate: 1e-4
optimizer: adamw_torch
lr_scheduler_type: cosine
warmup_steps: 10 (explicit; warmup_ratio is bypassed)
```

Keep the already selected TINY1_H2STATS sample, H40, dense 10D/132D ABI,
batch 1, gradient accumulation 1, seed 2222, projector/diffusion trainable
flags, and all locked data/evaluation boundaries. This candidate avoids an
unavailable exact-resume claim, avoids a hidden optimizer/scheduler/RNG reset
at step 200, and preserves the original ten-update warmup length.

It is still not a pure “only add 1800 steps” intervention: the cosine
denominator changes from 190 to 1990, so the first post-warmup LR path changes.
The result may support or reject a fresh 2000-step adaptation recipe, but it
cannot attribute an outcome to extra exposure alone.

The alternatives are less clean under the current source:

```text
exact resume: unavailable;
O1 weights + 1800 fresh updates: preserves the O1 start weights but resets
  optimizer/scheduler/RNG/custom sampler state at the boundary;
custom first-200-prefix-preserving schedule: stronger causal isolation, but new
  scheduler semantics requiring a separate PI decision and implementation gate.
```

## Provenance

The audit verified the immutable relay metadata before deriving schedule
values:

```text
O1 index SHA256:       ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7
O1 core seal SHA256:   7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806
run manifest SHA256:   4dfb69bbd0bb36562c17046735001454c8dc5cfe5e9e0195ae3ca032259a3148
trainer state SHA256:  2aa506bf605bce28809739970b6d3bc86059dba5490101c7da4fdba94512e4d9
train log SHA256:      35cc1f07fff30d413d52f2c11b8b72db7429a8004ae9244a11a798fbfd8f9c07
GR00T revision:        51d4c89f72fda44cbf77285c6a8114b52676b8a1
Stage23 project SHA:   deb920a070141e6a32a88cd0da6719828647e1e6
```

The complete O1 shard bytes remain outside this Corelab relay and were not
read, substituted, or rehashed in Stage27. This report is a checkpoint-state
and optimization-semantics result, not a new model or tensor audit.

## Stop boundary

```text
Hopper H2 training: HOLD
Hopper GPU/PBS: HOLD
model construction / forward / backward / optimizer: false
validation e / holdout f: false
C2 / WBC / hardware / ROS / ZMQ / P1M: false
```

Corelab stops here and requests `GPT_PI_REVIEW_AFTER_STAGE27_H2_SEMANTICS_AUDIT`.

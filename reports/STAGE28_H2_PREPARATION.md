# Stage28 H2 Preparation — Fresh BASE → 2000, Fixed Warmup 10

Date: 2026-09-04 (Asia/Singapore)

```text
ROLE: CODEX_CORELAB_LONG_HORIZON
STAGE: Stage28
BRANCH: stage28/h2-fresh-base-2000-fixed-warmup10
BASE_SHA: 9d0cf7c33b2d89ffee86168befe65978115564aa
PI_GATE: PI_GATE_STATE_0007
```

## Result

The preparation package is ready for the conditional Hopper token.  Corelab
did not construct a model, read checkpoint tensors, use CUDA, train, submit
PBS, or run inference.  The executable script is
`scripts/stage28_h2_fresh_base_2000_fixed_warmup10.py`; its default,
`--plan-only`, `--self-test`, and `--preflight` paths are non-training paths.

```text
H2_EXECUTION_READY: true
SCIENTIFIC_CONTRACT_CHANGED: false
DOWNSTREAM_AUTHORIZATION_STILL_VALID: true
HOPPER: conditional on live-HEAD and Hopper storage preflight PASS
NEXT_GATE: GPT_PI_REVIEW_AFTER_STAGE28_H2_RESULT
```

## Frozen recipe

The selected run is a new fresh recipe, not an O1 resume:

```text
exact BASE only
TINY1_H2STATS / exact training sample
H=40, physical=10D, universal=132D
seed=2222, batch=1, accumulation=1
tune_projector=true, tune_diffusion_model=true
tune_visual=false, tune_llm=false, tune_vlln=false
AdamW (`adamw_torch`), lr=1e-4, weight_decay=1e-5
cosine, max_steps=2000, explicit warmup_steps=10
warmup_ratio=0.0 (disabled)
resume_from_checkpoint=false
save_only_model=false
save_steps=1000, save_total_limit=2
```

The selected cosine denominator is 1990.  The run therefore tests whether a
coherent larger-budget adaptation recipe improves the same local flow field;
it does not isolate “the extra 1800 updates” from schedule changes.

## Identity gates

```text
BASE index SHA256: 407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59
BASE tree SHA256:  8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
TINY1 tree SHA256: 541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157
GR00T SHA:         51d4c89f72fda44cbf77285c6a8114b52676b8a1
Cosmos revision:   9ce19a195e423419c349abfc86fd07178b230561
Cosmos tree SHA:   1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9
```

The runtime `--preflight` hashes the actual files and performs a conservative
full-state storage reserve before model construction.  A failed hash, unsafe
path, symlink, missing cache, or insufficient free space stops the run.
The Hopper wrapper also binds every mode to the exact Stage28 live HEAD
returned by Corelab; a checkout drift fails before model construction.

## Checkpoint state and sealing

The stock Trainer must retain exactly the bounded ladder:

```text
checkpoint-1000
checkpoint-2000
```

Both must contain `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, and
`trainer_state.json`; BF16/FP16 settings make a GradScaler unnecessary.  The
final `checkpoint-2000` inventory and every file SHA are written to
`stage28_h2_checkpoint_2000_seal.json`.  A model-only or partial-state output
cannot produce a PASS handoff.

## Post-training evidence boundary

The PBS template
`hpc/pbs/stage28_h2_fresh_base_2000_fixed_warmup10.pbs` is non-submittable
until normal Hopper project discovery.  Under the PI token it performs one
H200 training run and then one bounded evaluation:

```text
BASE × 9 fixed Stage26 probes
H2@2000 × 9 fixed Stage26 probes
18 direct teacher-forced flow evaluations maximum
```

It records whether the BASE raw-velocity hashes reproduce the immutable
Stage26 control manifest and reports any mismatch without normalizing it.
It does not run a free-running sampler or choose a scientific interpretation.

## Forbidden in Corelab preparation

```text
model construction: false
forward/inference: false
training/backward/optimizer: false
GPU/H200/PBS submission: false
validation e / holdout f: false
sampler/C2/WBC/hardware: false
```

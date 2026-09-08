# ADR-032: Stage28 H2 fresh BASE, 2000 steps, fixed warmup 10

Status: Prepared for the conditional Hopper gate

## Decision

Stage28 uses one fresh exact BASE initialization and one 2000-update GR00T
adaptation with the already selected dense relative-EEF contract.  The
explicit recipe is:

```text
max_steps=2000
warmup_steps=10
warmup_ratio=0.0 (disabled)
lr_scheduler_type=cosine
optimizer=adamw_torch
learning_rate=1e-4
weight_decay=1e-5
seed=2222
batch=1, gradient_accumulation=1
bf16=true, fp16=false
tune_projector=true, tune_diffusion_model=true
tune_visual=false, tune_llm=false, tune_vlln=false
resume_from_checkpoint=false
save_only_model=false
save_steps=1000, save_total_limit=2
```

This is a new larger-budget fine-tune recipe.  It is not O1@200 plus 1800
updates and cannot be described as a pure extra-exposure intervention: the
cosine denominator is 1990 rather than the O1 denominator 190.

## Provenance and state policy

The Stage27-accepted BASE identity remains bound to index SHA256
`407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59` and
canonical tree SHA256
`8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4`.
The Stage28 preflight hashes the actual BASE tree, TINY1 fixture, modality,
pinned GR00T checkout, and Cosmos cache before model construction.

`save_only_model=false` is an artifact-preservation requirement.  The run is
accepted only if stock Trainer output contains `checkpoint-1000` and
`checkpoint-2000`, each with `optimizer.pt`, `scheduler.pt`, `rng_state.pth`,
and `trainer_state.json`.  The final checkpoint-2000 file inventory and bytes
are sealed after training.  The seal establishes artifact identity; it does
not by itself make the run an exact-resume claim.

## Evaluation boundary

After the one training run, the wrapper permits only the existing Stage26
source-mapped teacher-forced local-flow graph:

```text
BASE x 9 fixed probes
H2@2000 x 9 fixed probes
total = 18 direct flow evaluations
```

The same CPU-float32 noise/t pairs, TINY1 sample, H40 horizon, 10D/132D ABI,
and active mask are reused.  No free-running sampler, `get_action`, C2, WBC,
hardware, validation `e`, or holdout `f` is included.  Interpretation and any
next scientific branch remain a GPT/PI decision.

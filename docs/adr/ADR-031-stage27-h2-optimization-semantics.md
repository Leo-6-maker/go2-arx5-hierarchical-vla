# ADR-031: Stage27 H2 optimization semantics

Status: Prepared for GPT/PI review

## Context

Stage26 showed a small but directionally consistent local-flow improvement at
O1@200, while the implied decoded action remained severely out of profile.
The next scientific question is whether the bounded same-sample H40 adaptation
needs more optimization exposure. A direct change from `max_steps=200` to
`max_steps=2000` is not automatically a tenfold exposure-only intervention:
the cosine schedule and warmup are functions of the planned total steps.
Exact continuation would also require optimizer, scheduler, RNG, and sampler
state.

## Evidence

The exact Stage23 wrapper pinned by SHA256
`f42a1990847cf8f5326c66db37043ffed6e43458dd907fdbd08da5dd905b00a2` set
`save_only_model=True` and `resume_from_checkpoint=False`. The immutable O1
manifest contains no optimizer, scheduler, RNG, or scaler artifact. The
trainer state binds `global_step=200` but contains metadata/log history only.
The pinned GR00T source at `51d4c89f72fda44cbf77285c6a8114b52676b8a1` rejects
model-only checkpoint resume before model construction and documents the
custom trainer's resume reseeding behavior.

The pinned LR formula reproduces all 200 sealed O1 log rows exactly. The
original recipe has 10 warmup steps and a 190-step cosine denominator. A
fresh ratio-based 2000-step run has 100 warmup steps and a 1900-step cosine
denominator, diverging at displayed step 2. An explicit 10-step warmup keeps
the warmup length but still changes the cosine denominator to 1990.

## Decision

Stage27 adds a standard-library-only audit and records the following
decision aid; it does not execute or authorize H2:

```text
H2_FRESH_BASE_2000_FIXED_WARMUP10
```

Use a fresh exact BASE initialization and one coherent official training path,
retain the frozen TINY1/H40/ABI/flags, set `max_steps=2000`, keep LR `1e-4`,
AdamW and cosine, and set `warmup_steps=10` explicitly. The protocol must be
described as a fresh 2000-step recipe, not as a pure extra-exposure or exact
resume experiment. Any actual H2 run requires a new GPT/PI gate.

## Consequences

- O1 weights cannot be called an exact resumable training checkpoint.
- The missing optimizer/scheduler/RNG state is treated as a provenance fact,
  not silently reconstructed.
- The warmup choice is explicit, and the remaining cosine coupling is visible.
- No scientific contract is changed by this audit.
- Hopper H2, GPU/PBS, model execution, validation/holdout, C2, WBC, and
  hardware remain locked.

`SCIENTIFIC_CONTRACT_CHANGED=false`.

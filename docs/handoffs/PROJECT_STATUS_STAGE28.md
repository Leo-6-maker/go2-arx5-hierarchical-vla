# Project Status — Stage28 H2 preparation

```text
ROLE: CODEX_CORELAB_LONG_HORIZON
STAGE: Stage28 H2 fresh BASE -> 2000 fixed-warmup10
BRANCH: stage28/h2-fresh-base-2000-fixed-warmup10
BASE_SHA: 9d0cf7c33b2d89ffee86168befe65978115564aa
ACTIVE_PR: Draft stacked on PR #13
PI_GATE: PI_GATE_STATE_0007
```

Stage28 preparation is complete.  The selected H2 recipe is a fresh exact
BASE run with `max_steps=2000`, explicit `warmup_steps=10`, `warmup_ratio=0`,
cosine/AdamW, learning rate `1e-4`, seed `2222`, and the unchanged
TINY1/H40/10D-132D dense relative-EEF contract.  It is not an O1@200 resume or
a pure “additional 1800 steps” intervention.

The implementation is in
`scripts/stage28_h2_fresh_base_2000_fixed_warmup10.py`.  The non-submittable
single-GPU template is in
`hpc/pbs/stage28_h2_fresh_base_2000_fixed_warmup10.pbs`.

The runtime preflight rechecks the Stage27-accepted BASE index/tree bytes,
TINY1 fixture, modality, pinned GR00T source, and Cosmos cache before model
construction.  It also requires a conservative full-state storage reserve.
The Hopper wrapper binds all three modes to the exact returned Stage28 live
HEAD and fails closed on checkout drift.
The training path refuses resume, sets `save_only_model=false`, retains only
checkpoint-1000 and checkpoint-2000, requires stock optimizer/scheduler/RNG/
trainer state in both, and seals the final checkpoint file inventory.

The post-training path reuses the Stage26 direct local-flow graph with the
same nine fixed CPU-float32 probes for BASE and H2@2000, for a maximum of 18
flow evaluations.  It has no free-running sampler, C2, WBC, hardware,
validation-e, or holdout-f path.

```yaml
H2_EXECUTION_READY: true
SCIENTIFIC_CONTRACT_CHANGED: false
DOWNSTREAM_AUTHORIZATION_STILL_VALID: true
STORAGE_PREFLIGHT: HOPPER_RUNTIME_REQUIRED
CORELAB_MODEL_CONSTRUCTED: false
CORELAB_TRAINING: false
CORELAB_GPU: false
CORELAB_PBS_SUBMITTED: false
HOPPER_AUTHORIZATION: conditional PI_GATE_STATE_0007 only
NEXT_GATE: GPT_PI_REVIEW_AFTER_STAGE28_H2_RESULT
```

Corelab stops here.  Hopper must verify the live Stage28 SHA and pass the
runtime storage/identity preflight before executing the one authorized run.

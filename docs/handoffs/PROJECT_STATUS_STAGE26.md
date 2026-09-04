# Project Status — Stage26 H1 C1

ROLE: CODEX_CORELAB_LONG_HORIZON
STAGE: Stage26 H1 C1 preparation plus P1 provenance closure
ACTIVE_EXECUTOR: CORELAB_PREPARATION
HOPPER: CONDITIONAL_FORWARD_ONLY_AFTER_CORELAB_READY
H200_TRAINING: FORBIDDEN
C2: LOCKED
WBC: LOCKED
VALIDATION_E: LOCKED
HOLDOUT_F: LOCKED

Corelab has prepared a source-mapped runner on the
stage26/h1-fixed-noise-local-flow branch, based on the Stage25-closed SHA
4005a4dd1ab6c82807985ef856d340fa50acaa6c. The runner reuses the immutable
Stage23 C1 grid and binds the exact TINY1 H2STATS sample, processor statistics,
modality, source snapshot, pinned GR00T source, and exact checkpoint index
identities.

The preparation path was not a model run. The P1 checkpoint gate now reuses
the Stage25 identity strength before any model construction:

BASE requires the exact index SHA and canonical full-tree SHA
`8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4` under
the `payload/GR00T-N1.7-3B` prefix. O1 requires the exact index SHA, the
immutable Stage23 core seal SHA
`7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806`, and
the exact five supplemental `experiment_cfg/*` files bound to their
historical bytes/SHA. Both gates hash actual shard bytes before model
construction; index-only agreement is insufficient. Any altered BASE/O1
shard or supplemental mismatch fails closed.

MODEL_CONSTRUCTED: false
MODEL_FORWARD: false
TRAINING: false
BACKWARD: false
OPTIMIZER_STEP: false
PBS_SUBMITTED: false
GPU_H200: false
WBC_EXECUTED: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false

The conditional Hopper runner is bounded to:

checkpoints: [BASE, O1@200]
probes_per_checkpoint: 9
maximum_direct_flow_evaluations: 18
iterative_sampler: false
training: false
backward: false
optimizer_step: false
C2: false

The output has no pre-registered scientific PASS threshold. It returns paired
BASE/O1 MSE and ratios, raw predicted-velocity hashes, decoded physical
profiles, and SafetyGate transitions for GPT/PI review.

P1_CHECKPOINT_IDENTITY_STRENGTH: PASS
H1_C1_EXECUTION_READY: true
SCIENTIFIC_CONTRACT_CHANGED: false
DOWNSTREAM_AUTHORIZATION_STILL_VALID: true

The current Corelab handoff SHA and these readiness values are recorded in the
Stage26 PR/Issue execution note. Corelab stops before the conditional Hopper
forward-only gate.

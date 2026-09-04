# Project Status — Stage26 H1 C1

ROLE: CODEX_CORELAB_LONG_HORIZON
STAGE: Stage26 H1 C1 preparation
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

The preparation path was not a model run:

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

The final Corelab handoff SHA and H1_C1_EXECUTION_READY value are filled in
the PR/Issue execution note after compile, diff, and deterministic tests pass.

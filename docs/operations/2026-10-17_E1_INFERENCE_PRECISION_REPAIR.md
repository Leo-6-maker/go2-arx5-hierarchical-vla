# E1 Inference Precision Repair

Scope: same-semantics engineering repair inside the existing E1 authority,
clock, query/start/CPU/GPU budgets and fixed scientific panels. No new training,
data, seed selection, threshold change, WBC change or resource window.

The installed Torch 2.9.0 CPU-only environment inspection reports the original
default `cuda.matmul.allow_tf32=False` and float32 precision `highest`.
E1 M0 probes occur before `TrainingArguments(tf32=True)`; M1 probes and resident
evaluation occur after it in the same process. TrainingArguments changes the
global matmul flag. BF16 autocast alone is not sufficient evidence of identical
inference context: the installed Qwen3-VL rotary implementation explicitly
disables autocast for its FP32 frequency matrix multiplication. Its installed
source SHA256 is `685b10cbc48556343df4e159199c5884d02db9ad227640f0f4fbe75ddf0b2a99`.
This establishes a context discrepancy, not a measured action difference or
an explanation of task failures. No extra model queries were used to diagnose it.

The active first PBS capsule is not modified. Its original probes, autonomous
results, negative outcomes and actual costs remain immutable, clearly marked as
post-training inherited-context evidence rather than canonical matched evidence.

The next native allocation explicitly scopes each E1 inference call to original
FP32 matmul highest precision inside the unchanged BF16 autocast primitive,
then restores the prior training flag even on failure. Training precision stays
unchanged. Before D or M2, both fixed M1 checkpoints receive the same 96+12 probes
and original TRAIN4 under distinct `M1-256-CANONICAL` and
`M1-1024-CANONICAL` namespaces, identical fixed inputs/seeds, no outcome selection.
The M1@1024 parent remains fixed regardless of either result. New D/M2 inference
uses the same explicit original context. All additional work is counted inside
the original caps; an insufficient budget preserves NOT_RUN rather than exceeding
the cap. No extra PBS submission is requested for this repair.

Canonical M1 evidence does not erase or silently replace earlier results. Report
both sets, their numerical contexts and separate resource charges. Source physics
STOP still suppresses all subsequent physical evaluations and collection.

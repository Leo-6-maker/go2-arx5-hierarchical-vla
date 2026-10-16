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

## First Allocation Terminal Engineering Failure

Native job 670561 exited 1 after 5190 seconds allocated walltime, with all 1024
updates complete and frozen hashes unchanged. M1-256 preserved three valid S1
rejections and one engineering-incomplete condition (8.8 s actually executed);
M1-1024 inference/evaluation was not reached. The native traceback is a
FileNotFoundError while reading the shared resource ledger, not a model/physics
safety failure. The file is present on the login node after termination; NFS
rename visibility is a plausible cause, not a uniquely established diagnosis.

New E1 control reads retry only ENOENT/ESTALE at most six read attempts over
0.5 seconds. Persistent absence, malformed JSON, permission and other errors
remain fail-closed. No write, model generation, command installation or physics
operation is retried by this helper. The unknown final generation remains
unknown and is never resent. The existing second allocation performs the
presealed canonical panels from their original initial states under new IDs.
This is an explicit engineering recovery, not a replacement of negative results.

## Binding Scope Correction: Do Not Repeat Valid Failures

The preceding eight-start proposal was too broad: protocol section 9 explicitly
forbids repeating valid negative policy outcomes as engineering recovery. This
section supersedes that proposed physical panel while preserving its history.
Fixed offline probes may still compare numerical contexts, but the three valid
original M1-256 failures are retained with their precision-context caveat and
are NOT rerun. Only original F1/000 T2, which was engineering-incomplete, may
recover. M1-1024's original four conditions were never started and remain due.
Thus the intended remaining physical work is one recovery plus four first runs,
not eight reruns. The 256-step fully canonical autonomous panel remains incomplete;
do not fabricate it from other contexts or claim this caveat has disappeared.

Before the restriction was applied, the finite dispatcher had already started
one M1-256-CANONICAL F0/000 T1 condition. The local dispatcher was stopped, its
current relay was given an owner STOP, and no second repeated valid condition
was started. Preserve this unintended changed-context attempt and every actual
query/integration/cost, but exclude it from the adopted scientific comparison.
The native GPU job was not cancelled or modified. After the Core worker and
physical reservations were terminal, the same owner resumes only the unstarted
engineering-incomplete F1/000 T2 UUID already bound in the immutable registry;
completed/unknown neural and physical intents are never replayed. The new CPU
guard derives that one allowed condition from the four original adjudications,
rejects a valid-result retry, and retains the original 1024 parent and all caps.

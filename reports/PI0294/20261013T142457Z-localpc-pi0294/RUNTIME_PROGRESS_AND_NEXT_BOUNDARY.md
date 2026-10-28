# Runtime Progress and Next Boundary

The single existing local-PC owner is continuing the complete overnight goal.
This is neither policy qualification nor goal completion.

Actual serial, two-worker and four-worker MuJoCo diagnostics passed independent
audits and exact comparison of physical arrays, high-rate contact histories,
submitted actions and all PNG bytes. Short-batch pipeline rates were 17.25,
31.74 and 49.24 admitted episodes/hour. Four-worker speedup was 2.85, not four.
An independent eight-episode F1/F2 pilot is now running before official v3.0
collection. These diagnostic/development episodes are not counted as TRAIN/DEV.

Hopper completed the actual four-update continuous path and two-plus-two full
resume path under its own legal H200 allocations. All data, RNG, native noise,
cursor, scheduler and checkpoint-RNG identity comparisons passed; loss stayed
within its fixed tolerance. Full weights and optimizer arrays did not meet the
original numerical tolerance. This failure is preserved. Main training and
microbatch benchmarks were not admitted. No test checkpoint was deleted.

Forensic reads found identical pristine parameters/configuration and bit-exact
losses over the first 32 microbatches, but slightly different gradient norms.
Backward numerical nondeterminism is a hypothesis to investigate, not a proven
root cause or a justification for relaxing the failed gate.

Three of four PBS calls are consumed and reconciled: 489 allocated GPU seconds,
including idle, with zero outstanding reservation. Eight of sixteen engineering
updates remain. The last PBS call is held. Any next execution must preserve
the first failure, qualify its numerical repair, stay within the remaining
update/time/storage bounds and provide a viable main/evaluation arrangement;
it must not spend the last allocation merely to return another preparation
report. The other healthy GPU job is untouched.

The new source contains tested fresh-observation relay hooks, but CoreLab still
consumes immutable 8810aa29 during collection. The relay has not been deployed
or validated with actual double SSH/physics. No model replay is called closed
loop. WBC, mobile Go2 and hardware remain outside this run.

Official paired data, TRAIN-only statistics, private HF delivery/readback,
shared-model formal learning, real checkpoint behavior, candidate selection,
confirmation/regression, videos and final delivery remain outstanding.
Original cutoff 09:00 SGT and deadline 10:00 SGT remain unchanged.

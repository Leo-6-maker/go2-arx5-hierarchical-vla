# Explicit two-job allocation revision

Human explicitly answered: "两任务并行，按上述边界修订" to the question
authorizing account-wide at most two independent GPU allocations, with PI0292
itself still at most one GPU and eight cumulative allocated GPU hours. This
supersedes the just-prepared serial dependency plan for this S3 request only.
The existing unrelated healthy task remains untouched on CFP04-CF-009;
PI0292 remains CFP04-CF-007. No borrowing its GPU or outputs is allowed.

The admission permits only the exact previously observed same-owner
`pi05_relative` single-GPU peer on CFP04-CF-009, or its absence after completion.
Any third, replaced, wrong-project, foreign-owner or multi-GPU own job fails
closed. PI0292's own job remains separately verified on its own running compute
node and scheduler-provided single H200. The scientific source/recipe/data/stats,
original T0/cutoff/deadline, 3-call submission cap and all PI0292 budgets remain.
The unrelated task's resource cost is not falsely charged as PI0292's cost.

The already deployed serial guard/manifest remains immutable. The revised guard
is separately hash-bound in the same registered retry directory before dispatch.
Actual submission, allocation and model results require separate receipts.

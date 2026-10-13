# Actual S3 to main-capacity binding

The new `smoke_capacity_receipt.build` composes the existing full-checkpoint
inventory and conservative main-capacity planner. It reads actual S3 continuous
and 2+2 run receipts, the hash-bound resume equivalence, all three full model/
optimizer/scheduler/RNG/processor/statistics/cursor checkpoints, actual optimizer
step times, and the terminal scheduler-reconciled resource ledger. Missing or
changing files, clock reset and unresolved allocation reservations are rejected.

It hashes the nine source receipts and pre-main probe seal into its result, uses
the largest actual complete checkpoint for the forecast, and retains the full
4096-step / effective batch32 requirement plus three-hour evaluation reservation
in the last allocation. It does not publish MAIN PASS or submit a job itself.

Three new synthetic CPU test methods / two subtests verify the composition and
rejections; fixtures are explicitly not actual smoke/checkpoints. Latest full
PI0292 CPU suite: 85 methods / 79 subtests pass. This source is not a hot edit of
the sealed c187d957 scientific execution code.

At this publication, S3 is still unsubmitted: its fresh native project query
returned account-manager login required. The failed preflight did not create a
REQUEST or call qsub. The original expired-project failed submission remains
the sole used call (1/3); original deadline and all counters remain unchanged.
Parallel guard release c95ab11c CI38377876859 is actually SUCCESS and all five
changed files were read back exactly. The user must restore account-manager
login without exposing credentials; fresh project/quota/queue gates then resume.

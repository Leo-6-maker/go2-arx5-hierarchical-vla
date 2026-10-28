# Dependency queue adoption

Current direct human instruction: "允许挂载任务". It is conservatively adopted
as permission to submit the already authorized PI0292 smoke behind the exact
observed healthy own job, not permission to overlap GPU allocations or to run
new science. The unrelated job's output is not an input to PI0292.

PBS `afterany` waits for the predecessor to finish, independent of its scientific
success. This preserves the healthy predecessor and does not require cancellation.
The syntax and semantics are documented by the
[official PBS Professional reference](https://help.altair.com/2021.1.1/PBSProfessional/PBS2021.1.1.pdf).

The planner's default no-own-job predicate remains unchanged. A distinct queue-only
path requires this explicit authority, exact observed predecessor IDs, the same
owner, running state, one GPU and a different project. Unknown, extra, foreign,
queued, same-project or multi-GPU predecessors fail admission. The IDs stay in
private manifests. The allocation guard still requires the PI0292 job to be the
only current own job before any model load. It also refreshes project and quota
inside allocation, preserving the original clock, 8-hour GPU budget, 3-call limit
and fixed S3 scientific source/recipe.

Submission and scheduler dependency read-back will be reported separately.
This document and CPU fixtures do not claim a job was accepted, started or passed.
Latest PI0292 CPU suite: 80 methods and 71 subtests pass.

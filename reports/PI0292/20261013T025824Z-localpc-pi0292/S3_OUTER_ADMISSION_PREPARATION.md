# S3 outer admission preparation

This is a distinct guarded S3-smoke-02 retry, not a new scientific recipe or a
hot replacement of c187d957 operations. The failed first request, prepared second
manifest and original resource ledger remain preserved.

The stdlib-only guard verifies exact job/owner/project/GPU count, fresh native
personal quota, fixed clock, migrated-dependency evidence, manifest/adoption
delta and lease hashes before replacing itself with the original sealed wrapper.
Project membership, dates and balance are queried again in the actual allocation;
queued time cannot make an old login-node query fresh. If compute-node account
manager access is unavailable, admission fails closed before model execution.
This route still requires actual allocation testing and is not claimed proven.

The private controller prepares a hashed guard and path-only adoption manifest
inside the already registered retry directory. It generates a request/job only
after fresh live admission and source CI success. An uncertain submission outcome
retains the call and its reservations; no automatic repeat is allowed. Existing
unrelated healthy own jobs are never cancelled, ignored or reused.

Latest CPU suite: 79 PI0292 methods and 65 subtests pass, including 7 guard methods
and 16 subtests. Actual qsub/model smoke/training/live relay are not test fixtures
and remain separate scientific gates.

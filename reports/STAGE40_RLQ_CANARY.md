# Stage40-RLQ Canary Result

Classification: REFERENCE_LIFECYCLE_REQUALIFIED_WITH_TEARDOWN_DEVIATION

The single authorized non-scientific canary was job 604041.hopper-m-02 on
hopper-25 with one NVIDIA H200. It executed at git HEAD
b181267d29ad864ba76f848f201d9e98f469022f.

The canary proved:

- initial chunk physical payload equivalence: true;
- INITIAL_BIND completed before the first policy forward;
- root, DOF, episode and policy-history hashes were unchanged by INITIAL_BIND;
- the first finite native actor step completed with done=false and reset_buf=false;
- RUNNING_UPDATE completed with unchanged root, DOF, episode and policy-history hashes;
- a finite native actor step completed after RUNNING_UPDATE with done=false and reset_buf=false;
- no GR00T, training, hardware control or Stage40 scientific rows.

The scheduler returned exit status 139 because Apptainer/Isaac segfaulted during
teardown after the canary JSON had already been written. This is recorded as a
teardown-only deviation; it does not invalidate the lifecycle evidence.

Sealed output:

- result JSON SHA256:
  f9068f0fbf6f5d735b63a2bfef6944ec47e0e2bf924c58cb4ef2d6c26393c753
- run log SHA256:
  b6593628c1e999aeadb42029cdd788dec470b4c3ae900ba1a11fe4367a473ca3
- PBS output SHA256:
  73985c8a064576d8cc7c110df293a5550fca422f3ca08b5852e1911bd2453b7c
- seal SHA256:
  895eec6decbb9d9f37fd5c4384259926f5d85d1228007046b9233e80f407ec8a

The canary establishes only the two-phase native reference lifecycle. It does
not reopen Stage40 B1, C0, or Stage40 C. Next gate:
GPT_PI_REVIEW_AFTER_STAGE40_RLQ.

# PI0292 S3 Compute Admission Hold

Authentication is restored. Fresh native login queries confirm the authorized
project, membership, dates, positive balance and personal quota headroom. This
hold is not the earlier expired-token hold.

S3-smoke-02 (664294.hopper-m-02) entered its own legal single-GPU allocation but
failed before the model because the compute login shell could not find amgr.
The original failure and 12 seconds of full allocated GPU time were retained.

The user explicitly authorized only one extra engineering smoke submission:
PBS cap 3 -> 4, with the original GPU, concurrency, clock, recipe and science
gates unchanged. Repair release 383da16006f1c13918291c9918db8c05dc2ba9c0 passed
106 CPU tests and 137 subtests, GitHub CI 38408205304, and eight byte-exact GitHub
file readbacks. A minimal-PATH native query on Hopper login actually passed.
None of those tests qualified the compute-node executable.

The authorized S3-smoke-03 (664308.hopper-m-02) was submitted once. Its own compute
guard rejected native amgr executable availability or byte identity before the
project query and before the original model wrapper. The retained log does not
distinguish absence, permissions and hash mismatch. Do not claim a specific one
without evidence. The allocation is terminal and reconciled: another 5 GPU
seconds, 17 cumulative. Model loads, optimizer updates and generation remain zero.
This is an incorrect admission-layer environment assumption, not a user login
failure or scientific model-quality failure. Both original failed capsules stay
immutable; neither has been hot-patched or rerun.

Only one PBS submission remains under the revised cap. It cannot start main
training without actual successful S3. The one extra engineering smoke has been
used; another automatic smoke or a silently repurposed last main allocation is
not authorized. No PI0292 process or allocation remains active; the independent
healthy peer is untouched. No further GPU request is made by this report.

## Next Review

The next engineering design should obtain a fresh native project attestation on
Hopper login, bind it to the exact running PBS job and a compute-generated nonce,
and verify it on compute before loading any model. A bounded shared-filesystem
request/response between Hopper login and its own compute allocation is one
candidate; no credentials should be copied to compute or published. Native
scheduler identity, project, quota, source, lease and original deadline checks
must remain. A stale prequeue receipt or a changed timestamp alone is insufficient.

This attestation mechanism is NOT_IMPLEMENTED and NOT_RUNTIME_QUALIFIED here.
The request, response, replay, expiry, producer identity and compute routing must
be tested before another scientific submission. Any further smoke authorization
and how to preserve main/eval submission capacity require explicit review; the
existing cap, time and scientific quality thresholds are not automatically reset.

S0-S2 assets remain qualified; S3 is blocked before model execution and S4-S6 are
NOT_RUN. The full scientific goal is incomplete. Selected/deployment remain NONE.
Original cutoff: 2026-10-14T01:58:24Z. Deadline: 2026-10-14T02:58:24Z. Blocking does
not pause either clock.

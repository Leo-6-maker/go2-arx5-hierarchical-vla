# PI0294 Same-Allocation Stage Admission

A new PBS submission still requires a native login project/balance query and a
fresh, hash-bound project receipt. This change does not renew an AMS token,
authenticate with a password, or reuse an old receipt to start a new allocation.

The final night allocation's main/resident/checkpoint subrequests may instead
bind the original successful admission to the same still-running native PBS job
and the exact live night parent. The project timestamp remains the original
query time; it is never rewritten to pretend a new query occurred.

The child carries the full parent request SHA, retains the original project
receipt SHA and nonce, and cannot change source, inputs, recipe, submission
count, walltime or prior GPU accounting. The actual parent admission must have
passed the 30-minute project freshness check. The parent must still have the
same UID/PID/start identity and exact night-session command, same job/run,
RUNNING child mode and exact active child request SHA. A dead/reused parent,
different job, missing proof or changed request fails closed.

Every substage still obtains current native PBS owner/project/state/node/GPU,
peer allocation and quota evidence. The original allocation end bounds all
substage ends. The initial entitlement is not called a fresh current balance.
The original science gates, 4 submissions, 12 GPU-hours, 16 engineering updates,
fixed training data/statistics, and 09:00/10:00 cutoff/deadline remain unchanged.

CPU fixtures demonstrate the contract only. No new PBS, authentication,
model execution or actual same-allocation qualification is claimed here.

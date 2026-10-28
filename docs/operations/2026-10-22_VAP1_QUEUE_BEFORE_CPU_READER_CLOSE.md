# VAP1 Queue Before CPU Reader Close

The existing local-PC owner was explicitly instructed to submit the GPU task
without waiting for the fixed original-data CPU reader to finish. This changes
engineering scheduling only, not the frozen source A/B recipe or data population.

An optional `FIXED_MANIFEST_AWAIT_CLOSED_READER_V1` consumer binding records the
exact existing ORIGINAL126/C1 producer root and immutable manifest hash before
submission. It cannot select another producer or substitute another data bank.
The already-closed 139-parent bank retains its exact result and lease hashes.

On an actual allocation, the entry waits at most 300 seconds for the original
reader's successful terminal result and administrative closure, before importing
torch or touching CUDA. Every wait interval rechecks the actual owned running
PBS allocation. A failed reader, changed manifest, stop, or expired wait is a
preserved engineering stop with no model load or optimizer update. Allocation
time is charged, including the wait. An absent result is never a pass.

After closure, all existing full-reader, dataset, raw-binding, lease, registry,
same-job hardware, model compatibility, budget and first-update gates still run.
The premodel receipt seals the actual consumer result hashes; every subsequent
model/training admission revalidates them. The allocation plan must include the
300-second input-wait ceiling and reserve all unfinished CPU output at its full
prepaid bound. No budget, clock, WBC, S1, statistics or scoring changes result.

The existing waiting control process may be replaced only after proving it has
not registered/submitted a GPU allocation and recording its exit. The fixed CPU
reader is not stopped or restarted. There remains one scheduling owner and at
most one new source-A40 submission; unknown submission effects are not retried.

This document is not a submission receipt. Report the actual PBS job identity
and native queue/allocation state separately after execution.

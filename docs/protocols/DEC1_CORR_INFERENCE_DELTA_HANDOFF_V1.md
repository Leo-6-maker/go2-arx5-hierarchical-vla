# DEC1 Fixed CORR Inference Delta

This is an engineering implementation inside fixed DEC1 authority
`a7a7ecd9a6be62a7221af25a5e32a0009b076a84`, not new compute allowance.
Original PV1 T0, deadlines, final reserve and single-GPU limit remain.

## Exact Assets

The only policy is CORR@2048, original complete-manifest SHA
`94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`,
with original R0
`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`.
The existing public M2048 and E1 M2-CORR are different policies and cannot
substitute for it. There is no new model or statistic fitting.

The first 4,986,649,584-byte safetensors shard is byte-identical to an existing
public M2048 shard, SHA
`cbcaea5ee88f1e0f1465043920a2647c67e7de17d24adfd1c477742a6168edec`.
The delta manifest refers to that member at fixed HF revision
`4c3097652bb81e43ea40a2b0b423086ec6ea7091`; it is not reuploaded. Its reuse
does not turn the old model into CORR: the two changed shards are mandatory.

Only the nine inference members are materialized. Two configuration model-name
locators are relocated using the established `public_config` function; all
weights, index and statistics remain exact. Existing fixed upstream license,
Cosmos agreement, notices and attribution accompany the delta. No optimizer,
scheduler, sampler, RNG, Trainer state, private paths, raw observations,
WBC/vendor assets or credentials are published. It is not a full-resume export.

## Native CPU Admission

Each operation requires a new standard manifest, owned registered output,
immutable source/input binding, profile and live DEC1 child lease before
dispatch. Existing runs and closed leases are not reopened. The original
campaign and T0 are explicit even on the alternate site; there is one owner.

Before dispatch the coordinator must actually prepay 4,500 CPU seconds each
for publishing and alternate-site consuming, plus 180 for READY finalization,
in the original campaign CPU ledger and DEC1 classification. Alternate CPU
is conservatively included within the GPU-site CPU allowance, not borrowed
from an expired Vanda experiment. Each worker has a CPU limit, at most two
cores and one-hour wall limit. No CUDA visibility, model load, GPU allocation,
PBS submission, inference, physics or optimizer is involved.

Reserve 16 GiB logical transfer for upload plus independent download and small
metadata, inside DEC1's original 64 GiB total, and at most 8 GiB new alternate
resident data. The existing frozen shard is a local hardlink after a fresh
full byte hash, with source mode/content preserved. It is not an independent
backup. The original protected full checkpoint is untouched. This transfer
reservation must remain in every later full-checkpoint/matrix budget forecast.
No checkpoint retirement is implemented or authorized by this component.

## Commit And Readback

`scripts/pi0294/dec1/inference_assets.py` reuses the mature inference allowlist,
sanitizer, safetensors-header inspection and fixed-HF verifier. It writes an
intent before an effect-bearing Hub commit. An existing prefix, result or
intent refuses blind reuse. A failed or uncertain mutation requires a separate
read-only reconciliation; no automatic retry or overwrite is provided.

Payload publication is **not READY**. The alternate consumer downloads every
new member at the exact payload revision and hashes the actual bytes. It also
rehashes its local reused frozen shard; old metadata alone is insufficient.
The resulting nine-file inference directory is checked again. Only its sealed
real consumption receipt admits a separate READY-last commit. The original
payload and dependency remain fixed and are verified at the final revision.

An asset READY is **not numerical qualification, policy improvement, target
executor delivery, source/target closed loop, or a scientific Goal completion**.
A future bounded A40 compatibility allocation still needs global GPU/PBS/query
accounting, exclusive native queue proof, fresh site authority and its own
standard cold admission. Existing H200-only guards are not weakened here.

The public consumer can explicitly bind `public_anonymous_read=true` with no
credential locator. It then passes `token=False` to the Hub, not an ambient
token or the other site's Home path. Publishers and READY writers still need
their existing local writer file, checked before dispatch. No credential is
copied between sites. A failed pre-download consumer is preserved and closed
only after proving it made no Hub calls; a new cold child, not a hot edit or
blind restart, performs the corrected read. Its CPU must still fit the already
prepaid finite pool and all failed control/source-copy bytes remain accounted.

At source publication this component has CPU fixtures only. A native delivery
must be reported separately with actual revisions, full readback and costs.
`selected/deployment=NONE` throughout.

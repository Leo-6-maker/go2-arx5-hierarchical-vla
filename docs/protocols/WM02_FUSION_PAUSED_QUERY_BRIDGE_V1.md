# Paused capture and one frozen query · V1

This is a future-capsule implementation contract, not an executed experiment or
an online control result. Existing archived inference and observation modules
remain unchanged. Neither inference nor this bridge installs a VLA reference.

`FROZEN_VLA_LIVE_PAUSED_QUERY` uses existing profile
`wm02-fusion-vla-shadow`, site `CORE_FUSION_NEURAL`, the original registry,
FUSION/P1 leases, GPU UUID/headroom checks, job owner and neural owner locks.
There is one model load and one generation attempt; no training or physics.
The immutable CORR2048 manifest is
`94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38`.
The full checkpoint-file inventory must hash to
`f4c9486006de1a58721a1fc723361264e406ac4910ba46f14a31233c7a28fe23`.
Every checkpoint/processor member and declared source file is reread. Loading
retains original local-only backbone alias/revision, full loading-info checks,
FP32 frozen parameters, BF16 autocast/TF32-off inference, four sampler steps and
the saved source category 10. Original `ResidentModel.generate` performs the
single processor decode; physical H40 and normalized H40 are retained unchanged.

There are two explicit snapshot binding modes:

- `SEALED_SNAPSHOT_EXACT_SHA`: the GPU spec pins an already sealed paused
  envelope. This alone does not implement a live two-scope bridge.
- `PRE_REGISTERED_SINGLE_SNAPSHOT_V1`: both fresh finite scopes are registered
  before native startup. The GPU spec pins the producer native spec and a common
  pair file, then waits finitely for its one predeclared snapshot. The GPU spec
  is never edited after registration.

The pair file has exactly these keys:

```text
schema = WM02_FUSION_PAUSED_QUERY_PAIR_V1
pair_uuid, query_uuid
native_root, native_run_id = native-01
native_boundary_step = 3703, capture_index = 0
gpu_root
task_sha256, scene_sha256, frame_map_sha256
seed, checkpoint_manifest_sha256, source_sha
deadline_epoch, max_queries = 1, native_integrations_by_bridge = 0
```

Both specs pin `paused_query_pair` and its file SHA256. The CPU spec declares
`paused_query_gpu_spec`, `paused_target_task_sha256`, and
`paused_frame_map_sha256`. The GPU spec declares `producer_native_spec` and its
SHA256, `snapshot_binding_mode`, `paused_task_sha256`, `query_uuid`, `seed` and
finite `snapshot_wait_limit_s`. Its `recorded_observation` is derived exactly as
`native_root/native-01/PAUSED_OBSERVATIONS/query_uuid/PAUSED_OBSERVATION.private.json`;
`recorded_observation_sha256` is absent or null in pre-registered mode. Its actual
value is bound in generation intent and result after the real capture exists.
Both specs share the actual owner/campaign/P1 identity, source capsule and
registry. GPU deadline equals the pair deadline and cannot exceed the producer
or original scope deadline. The pair contains no spec hashes: the graph is
`task → pair → CPU spec → GPU spec`, without a circular SHA dependency.

Native caller APIs in `fusion_paused_query_bridge` are:

```python
preflight_pair(gpu_spec_path, native_spec)
capture_and_wait_query(adapter, binding=target_shadow_binding,
    task=declared_task, task_sha256=declared_task_hash,
    gpu_spec_path=gpu_spec_path, capture_index=0)
```

The first function is read-only and runs before physics admission/startup. It
requires both original registry entries and exact ACTIVE leases, a fresh pair,
matching producer spec bytes, task/query/boundary/frame/model and original
STOP/deadline checks. It requires neither a snapshot nor a running GPU process.
It grants no execution permission. The second function accepts the actual
already admitted native adapter; it does not register or launch anything and
does not reacquire the campaign lock held by that adapter.

The native caller captures the exact paused packet and real dual-RGB/calibration
members with the existing CPU Mesa recorder. After all five files and the sealed
envelope are verified, an atomic rename of the fsynced completion record publishes
`PAIR_SNAPSHOT_READY.private.json`. The already registered GPU worker waits only
for that one completion record, checks all actual member hashes and the full
sidecar, then performs one query. A capture failure, STOP, elapsed deadline,
producer closeout, invalid envelope or query failure terminates the attempt.
There is no recapture or query retry. Original worker closeout remains in use.
The original single owner and unique query directory provide one writer; an
existing completion marker is rejected. No hard-link filesystem support is used.

During the wait, the native adapter continues its original guard and fresh
pause-ACK/unchanged raw-packet/quota checks. On return it verifies the GPU's exact
spec, original registry entry, CLOSED lease/worker exit zero, query/model/noise
and observation identities, full action SHA256 and finite raw H40, then calls
the complete actual `verify_still_paused` barrier. Returned data are JSON-safe
hashes, paths, counts and pause proof; the full arrays remain in the GPU NPZ.
No motion is authorized by this return, and any later action requires its own
original complete certificate, ACK, live TTL and qualified execution contract.

Cold model-load wall time, query wall time, CUDA event time, CPU time and peak
memory are separate from the unchanged native simulation clock. A paused query
does not renew/rebase reference TTL and its returned action is not a live
reference. One draw cannot establish P50/P95 latency. Target camera equivalence,
embodiment/task qualification, source outcomes, task success and closed-loop
control gains remain unestablished by this capture/query diagnostic. Source
policy-view mapping and the complete actual target provenance remain mandatory.

`QUERY_ATTEMPT`, model-load intent, generation intent, raw arrays and failure
records are retained. A failure after generation intent has an uncertain completed
query count and never permits an automatic redraw. Mock tests exercise these
interfaces without actual GPU, model, renderer, physics or remote execution.

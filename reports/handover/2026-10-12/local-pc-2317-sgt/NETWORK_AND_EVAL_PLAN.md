# Network And Evaluation Plan

Status: DESIGN_ONLY. Owner: local-pc. No model service, tunnel or physics test was opened.

## Verified Versus Assumed

Verified at2026-10-12 23:08-23:18 SGT:
local-pc independently reaches Corelab through existing AMD ->4080S SSH, and Hopper login through alias hopper.
Both endpoints authenticate as leoyuer and read the same fixed model-repository revisions and small manifests.
PBS client/scheduler telemetry is available on Hopper. Historical cache/source/venv identities exist.

User constraint: Corelab and Hopper cannot directly communicate.
No cross-server network probe, scan, direct scp/rsync/ZMQ, or policy bypass was attempted.
SSH reachability to login nodes does not prove access/port forwarding into future allocated compute nodes.
HF read access does not prove write permission or permission to publish restricted data.

## Large Asset Transfer

Corelab is producer for task physics, expert episodes, preprocessing, CPU scores and video.
Hopper H200 is producer for neural predictions, training checkpoints and model-eval results.
local-pc coordinates, pins GitHub protocols and reads small receipts.

Corelab -> fixed HF data revision/prefix -> Hopper -> fixed HF model/eval revision/prefix -> Corelab CPU audit.
Use revision + allow_patterns. Reuse installed caches and existing authentication; never mirror all model history.
TransferManifest binds experiment/task/split, producer/consumer, execution SHA, repo_id/type/revision/prefix, per-file size/SHA256, data/teacher/executor/processor/stats/decode and STAGING/READY/VERIFIED.
Commit READY last and consume the final complete revision. Partial files do not qualify a bundle.
Full Trainer includes optimizer/scheduler/RNG/config/processor/stats; Core CPU scoring needs only numerical prediction evidence, not optimizer weights.
Resolve the visibility/license of new assets before upload; the historical hub is public and private WBC source/models/meshes/logs are excluded.
Write verification and a new restricted-data destination have not been established in this session.

## Offline Evaluation

Corelab packages fixed observations, causal history, labels and noise identities.
Allocated H200 runs neural model eval, returns prediction hashes/numerical outputs under a fixed revision.
Corelab independently recomputes CPU diagnostics.
Teacher-trajectory prediction and prerecorded action replay are offline tests. They cannot establish reactive physical success.
Do not move high-level GR00T eval to the old Corelab neural server.

## Proposed Reactive Relay

local-pc owns two persistent authenticated SSH channels:
1. to the Corelab task environment;
2. to the Hopper node inside a new legal PBS allocation.

A future model service binds compute-node localhost only. Its launch and permitted SSH forwarding need explicit later GO and cluster-policy validation.
For every decision:
Corelab completes the previous committed eight controls -> records new actual RGB/TCP/gripper snapshot -> local-pc relays that snapshot -> H200 returns H40 -> local-pc relays validated action -> Corelab commits the prefix -> records a new snapshot.
No future observation or whole-episode precomputed action sequence may be substituted.

Each request/response binds episode UUID, segment/reset generation, decision index, observed sim time, previous committed action hash, observation/hash, causal-history cursor, model revision, processor/stats/decode identity, request/action hash and expiration.
Use length-prefixed bounded binary payloads with JSON headers and explicit ACKs; chunk images/arrays and avoid shell-interpolated data.
Reject wrong episode/index/model/hash, duplicate/stale reply, missing previous-action ACK or history crossing reset/takeover.
Timeout/disconnect invokes a registered local safe hold/abort; preserve pending/committed/rejected evidence and never silently replay an expired reply.

Independent simulation may advance in lockstep only after a response; label NON_REALTIME.
Measure actual action -> new observation causal ordering. A paused simulator requires coherent physics/renderer/controller time handling.
Native wall/ROS WBC cannot be paused in only one component; future WBC composition requires its own clock qualification.
HF is not a per0.4s queue; GitHub is not control transport.

## Required Later Verification

- Authorized PBS queue, finite allocation, H200 node identity and SSH access/forwarding rules.
- Corelab role-specific independent physics runtime and permission to reserve GPU/render resources.
- Two persistent channels with synthetic payload only first; clock, framing, hashes, reconnect and reset/expiry negatives.
- Actual dual RGB payload size/bandwidth and p50/p95/max relay + model latency, under the real allocation.
- Strict previous-action/new-observation ordering and valid causally indexed history.
- Simulation lockstep clock/reset behavior; deployment latency and WBC real-time behavior are separate qualifications.
- Resource limits, transport buffers, cancellation, no orphaned services and finite shutdown.
- New fixed data/model manifests with supported visibility and permissions.

If the relay cannot be qualified, retain offline evaluation only or propose a separately authorized asynchronous paused-simulation design. Do not label prerecorded replay as closed loop.
Current relay implementation status: NOT_IMPLEMENTED_AND_NOT_TESTED.


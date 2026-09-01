# RecedingHorizonExecutorV1 and ReferenceUpdateV1

## Purpose

This contract defines transport-independent replacement of an external EEF
reference while one simulator episode continues. It is an interface contract,
not a network protocol and not a real-robot safety limit.

## Update contents

Each accepted `ReferenceUpdateV1` carries:

- `chunk_id` and the previous chunk ID;
- observation, receive, and execute timestamps;
- lifecycle status and chunk age;
- frozen prefix length `K`;
- raw previous-prediction and achieved-to-new-first pose discontinuities;
- provenance and explicit negative assertions for reset/teleport/history.

The Stage21 validation set is `K in {4,6,8,10}`. A test chooses and freezes
one value before evaluation; it cannot be tuned from the final test result.

## Invariants

An accepted reference-only update must satisfy:

```text
physics_reset        = false
physics_teleport     = false
policy_history_reset = false
```

The update may replace trajectory buffers and reference-progress state only.
Native controller timing, checkpoints, embodiment semantics, and physics
state remain unchanged.

## Lifecycle

`ChunkLifecycleTracker` classifies a chunk as `FRESH`, `AGING`, or `STALE`
from explicit age thresholds. Non-monotonic receive/observation time and
duplicate or older chunks are `OUT_OF_ORDER`; invalid timestamps are
`INVALID`. A stale/out-of-order chunk is not accepted as a new reference.

The Stage21 synchronous paused-simulation run deliberately had zero age at
execution because inference completed while physics was paused. That fact is
not a deployment latency claim.

## Handoff metrics

Two metrics must remain separate:

```text
previous predicted endpoint -> new chunk first point
current achieved EEF        -> new chunk first point
```

The first detects profile discontinuity. The second detects whether the new
reference is at least anchored near the current physical state. Neither is
silently repaired by smoothing, clipping, or retiming.

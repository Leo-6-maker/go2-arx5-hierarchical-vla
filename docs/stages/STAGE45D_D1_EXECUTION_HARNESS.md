# Stage45D D1 execution harness and runtime admission closure

Status: **RUNTIME_ADMISSION_READY_FOR_H200_GATE; NOT_EXECUTED**

This support artifact implements the CPU/static closure authorized by
PI_GATE_STATE_0107 (Issue #11 comment `5617612655`, Issue #50 comment
`5617614769`). It does not authorize or perform H200, Corelab GPU, PBS, or
model execution. PI_RESOURCE_ROUTING_POLICY_V3 makes Corelab the preferred
future bounded execution environment, but does not expand this current gate.

The committed runner is
`scripts/stage45d_d1_teacher_forced_local_flow.py`.  Its default plan/audit
modes resolve the repository from `__file__`, work from a foreign current
directory with an empty `PYTHONPATH`, and import only the CPU StructuredAction
codec.  Torch, GR00T, CUDA, checkpoint loading, and model construction are not
top-level imports.  `--execute` is gated by a future PI authority token and a
separate runtime-admission flag; the late-bound template keeps both closed.

The Stage45D-specific admission package is split from the scientific harness:

- `manifests/STAGE45D_D1_RUNTIME_ADMISSION_MANIFEST_V1.json` is a static,
  unmaterialized schema for the future observed runtime;
- `manifests/STAGE45D_D1_RUNTIME_COMPATIBILITY_MAP_V1.json` maps each D1
  boundary to the reusable Vanda runtime/admission responsibility at PR #49
  HEAD `7d055069ba1d391c17457f709f7fc9aff544ef02`;
- `scripts/stage45d_d1_runtime_admission.py` performs the CPU-only static
  checks and provides a future `validate_future_admission` entry point.

The immutable protocol-correction HEAD
`1dd5837267cb1af34cd2ae7a978f70d6c8f78bdb` is recorded only as
`protocol_provenance_head`. A later execution checkout must late-bind and
cross-check its own observed branch, exact HEAD, repository root, and clean
worktree; it may never pass by checking the protocol provenance HEAD alone.

## Frozen algebra

For each of the three checkpoints, 24 TRAIN-only windows, three fixed noise
seeds, and three fixed flow times, the harness records the exact source-mapped
law:

```text
x0      = fixed float32 CPU noise
x1      = checkpoint-statistics-normalized StructuredActionV1 expert label
x_t     = (1-t) * x0 + t * x1
v_target = x1 - x0
v_hat   = local velocity at (x_t, t)
x1_hat  = x_t + (1-t) * v_hat
```

Physical interpretation is always derived from `x1_hat`, never by decoding
`v_hat` as an action.  CPU synthetic oracle tests recover `x1` for all frozen
probe times when `v_hat == v_target`.

## Source and decode equivalence

`manifests/STAGE45D_D1_EXECUTION_MANIFEST_V1.json` pins GR00T source commit
`51d4c89f72fda44cbf77285c6a8114b52676b8a1` and names the accepted Stage38
anchors for conditioning, state/action encoders, time discretization, DiT
action-head forward, and velocity extraction.  The physical bridge is the
existing Stage45C `_decode_v2` path, which applies the unchanged Stage40
inverse-normalization and StructuredActionV1 materialization law.

`STRUCTURED@1000` is routed through the historical fixed-statistics path.
`FULL24@1000` and `FULL24@2000` are routed through the Stage45C runtime-V2
statistics bridge with stats SHA256
`3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337` and
payload SHA256
`752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75`.
The verifier rejects a FULL24 request that selects the historical
Structured-only fixed-stat route.

## Evidence lifecycle

The future run writes and hashes `v_hat`, `x0`, `x_t`, `v_target`, `x1`,
`x1_hat`, and the exact mask before any normalized or physical derived metric.
Each raw manifest binds checkpoint/tree/stats/window/seed/t identities.  Durable
flush/fsync and a versioned failure seal are part of the writer contract.  The
execution manifest fixes the later budget at 648 action-head calls plus 72
conditioning passes, with no adaptive extension.

The return token for this static gate is:

`STAGE45D_D1_RUNTIME_ADMISSION_READY_FOR_H200_GATE`

All current counters remain zero: model construction/forward, GPU, PBS,
training, Validation-E, Holdout-F, WBC, simulator, and hardware.

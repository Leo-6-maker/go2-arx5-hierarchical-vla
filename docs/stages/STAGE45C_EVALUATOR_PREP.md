# Stage45C evaluator preparation

This support branch implements the CPU/runtime-independent evaluator lane
authorized by PI0088 (`Issue #11` comment `5596802225`).  It starts from
Golden Training HEAD `8197e4cd591b9c44cf9f43576d3020e9e6ff5f24` and does not
change the StructuredAction, decoder, training, WBC, or SafetyGate contracts.

## Accepted source semantics

The canonical library is an extraction of the accepted Stage44R implementation:

| Stage44R source | Stage45C library | Preserved contract |
| --- | --- | --- |
| `scripts/stage44r_corelab_attribution_repair.py::_trajectory_metrics` | `evaluation.trajectory_metrics.trajectory_metrics` | world-frame metre positions; `d_start - d_end` progress; 1e-9 POSITIVE/ZERO/NEGATIVE sign; best progress; endpoint displacement/target cosine; path length |
| `scripts/stage44r_corelab_attribution_repair.py::_comparison_metrics` | `evaluation.trajectory_metrics.comparison_metrics` | H40/D10 shape; decoded-vs-expert position, endpoint, path, rot6d, gripper, and mixed-unit diagnostics |
| `src/go2arx5_vla/stage40_structured_decode.py::decode_raw_structured_action` | `evaluation.structured_action_eval.decode_raw_prediction` | raw `[1,40,132]` → Stage38/40 inverse normalization → physical `[40,10]`; K16/T16/H40; exact stats path is late-bound |
| Stage43 runtime boundary | `evaluation.checkpoint_eval.validate_safety_gate_decision` | observed SafetyGate must be `ACCEPT` or `WARN`; no clipping, smoothing, retiming, or correction |

The second-difference roughness diagnostic uses the already-maintained
`src/go2arx5_vla/metrics/action_profile.py` convention (`||p[i+2] - 2p[i+1]
+ p[i]||`, metres).  It is descriptive only; no roughness threshold or
trajectory correction is introduced.

Target metrics are continuous diagnostics only.  No task-success threshold is
introduced.  All target values are evaluation metadata and are not passed as
numeric model inputs; the accepted model input keys remain `body_rgb`,
`wrist_rgb`, and `state`.

## Pipeline boundary

```text
Golden Release / checkpoint manifests (Vanda P0/P1, late-bound)
                                  ↓
H200 runtime adapter (future, not implemented here)
                                  ↓
raw prediction artifact [1,40,132]
                                  ↓
Stage40 StructuredAction decoder (CPU authority)
                                  ↓
decoded physical trajectory [40,10]
                                  ↓
per-window target/comparison metrics (CPU authority)
                                  ↓
ordered 24-window aggregate + hash-bound evidence
```

Raw predictions and decoded arrays stay separate from aggregate reports.  The
aggregate carries their array hashes and any externally recorded artifact
path/hash so metric output cannot silently replace the raw evidence.  Every
row also carries one exact checkpoint identity and model/runtime identity;
aggregation rejects identity drift within a checkpoint cohort.

## Current gate and intentional late binding

`manifests/STAGE45C_TRAIN24_EVALUATION_PROTOCOL_V1.json` is
`PREPARED_NOT_EXECUTED`.  It freezes the existing `FULL24_EXISTING_TRAIN`
24-window target-major/snapshot-major order and the evaluator ABI, but leaves
Golden Release, GR00T runtime, FULL24, statistics, checkpoint, sampler/noise,
output-root, and runtime-admission identities null until Vanda completes P0/P1.
The library rejects an unbound runtime for execution.

The following remain explicitly out of scope for this branch:

- model construction, forward, inference, training, or checkpoint substitution;
- Validation-E or Holdout-F reads/materialization;
- H200/PBS submission, WBC/simulator, or hardware execution;
- evaluator thresholds, sampler, normalization, horizon, frame, or SafetyGate changes.

The PBS file is a thin future wrapper only; it is not submitted by Corelab.

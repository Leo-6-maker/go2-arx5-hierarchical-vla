# Stage22 R0/R1 Action-Profile Comparison

## Status

The comparison is implemented but no model-profile result is claimed yet.
The H200 R1 one-step gate has now passed and the authoritative H2 direct
launch is running/dispatched. The earlier 16 GiB and 32 GiB Corelab failures
remain non-authoritative resource evidence; no Corelab full model run was
started. The training input is the locked train-only fixture, not the
historical all-target loader fixture.

## Pre-declared recipes

| Recipe | Projector | Diffusion/action module | LLM | Visual backbone | Purpose |
|---|---:|---:|---:|---:|---|
| R0 | train | frozen | frozen | frozen | Stage20S control |
| R1 | train | train | frozen | frozen | action-generation hypothesis |

Both use the same base checkpoint, materialized dataset, seed policy, H=40,
and test-only action period. The authoritative H2 path invokes the frozen
Stage20S entry point from `deb920a070141e6a32a88cd0da6719828647e1e6` directly;
`scripts/stage22_train_recipes.py` is optional non-authoritative tooling and
must not be used to replace the frozen launch.

The independent `scripts/stage22_r1_preflight.py` additionally refuses a
selected GPU with existing compute apps and records GPU UUID/VRAM, conservative
device-level peak usage, PyTorch peak allocated/reserved memory, trainable
parameter count, one-step elapsed time, cleanup residual, and the
`max(15% total, 2 GiB)` headroom decision.

## Selection metrics

Training loss is not a promotion criterion. Each checkpoint will be evaluated
on held-out same-scene snapshots for:

```text
position error: mean / median / p95 / max
endpoint position and rotation error
path-length ratio
mean-speed ratio and max-speed ratio
trajectory roughness
initial position/rotation jump
TrajectorySafetyGate decision
```

No output will be clipped, smoothed, retimed, or otherwise corrected before
these measurements. A successful re-timing diagnostic would not qualify the
unchanged model output.

The predeclared simulation threshold is encoded in
`scripts/stage22_profile_gate.py`: zero SafetyGate rejects; median path,
mean-speed, and max-speed ratios in `[0.75,1.25]`; median mean-position error
at most 2 cm; median endpoint error at most 3 cm; common-future overlap P95 at
most 25 mm and max at most 50 mm. These are simulation qualification criteria,
not real-robot safety limits.

## Current scientific status

```text
MODEL_PROFILE_COMPARISON: NOT_YET_RUN
DIFFUSION_ACTION_TUNING_DID_NOT_CLOSE_PROFILE_BLOCKER: NOT_ASSESSED
H1_H200: PASS
H2_H200: RUNNING_OR_DISPATCHED
CORELAB_AUTHORITATIVE_GATE: READY_FOR_DIRECT_LAUNCH_ARTIFACT_AUDIT
```

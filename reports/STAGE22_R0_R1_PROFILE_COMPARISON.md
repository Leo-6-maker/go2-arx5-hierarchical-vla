# Stage22 R0/R1 Action-Profile Comparison

## Status

The authoritative H2 direct-launch comparison and Corelab read-only audit are
complete. The immutable relay audit status is
`H2_AUDIT_PASS_EXCEPT_LIVE_MODULE_INVENTORY`; this single strict role-level
inventory exception does not change the independently observed profile result.
The earlier 16 GiB and 32 GiB Corelab failures remain non-authoritative
resource evidence. The training input was the locked train-only fixture, not
the historical all-target loader fixture.

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
MODEL_PROFILE_COMPARISON: AUTHORITATIVE_H2_AUDITED
DIFFUSION_ACTION_TUNING_DID_NOT_CLOSE_PROFILE_BLOCKER: CONFIRMED_ON_TESTED_DISTRIBUTION
H1_H200: PASS
H2_H200: EXECUTED_AND_AUDITED
H2_GPU_PROVENANCE: H2_GPU_PROVENANCE_SUPPORTED_WITH_FORMAT_DEVIATION
H2_PROFILE_EVALUATOR_SEMANTICS: H2_PROFILE_EVALUATOR_SEMANTICS_FROZEN
CORELAB_AUTHORITATIVE_GATE: PASS_EXCEPT_LIVE_MODULE_INVENTORY
R1@100: NOT_AUTHORIZED
R1@200: NOT_AUTHORIZED
HOLDOUT_F: NOT_ACCESSED
WBC: NOT_AUTHORIZED
```

The audit independently reconstructed BASE, R0, and R1 from nine immutable
profile logs: 18 records and 15 overlap transitions per recipe, paired across
all seeds `2223/2224/2225` and the six locked `red_target_holdout_e`
snapshots. R1 reduced median mean-position error to `0.075335 m` and median
endpoint position error to `0.139669 m`, versus R0 `0.081892 m` and
`0.156097 m`; however, R1 retained median path/mean-speed ratio `14.240`,
max-speed ratio `32.036`, overlap p95 position error `0.205834 m`, and
`0 ACCEPT / 0 WARN / 18 REJECT` SafetyGate decisions. No profile-qualification
gate is met, so no checkpoint is selected and no bounded extension is
authorized.

The full summaries and required `BASE → R0`, `BASE → R1`, and `R0 → R1`
paired deltas are in `reports/STAGE22_H2_AUTHORITATIVE_AUDIT.md` and the
node-local machine-readable audit artifact.

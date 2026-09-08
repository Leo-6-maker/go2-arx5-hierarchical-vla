# ADR-027: Fixed-Noise Teacher-Forced Flow Diagnostic

- Status: Accepted for Phase C preparation
- Date: 2026-09-03

## Context

Stage23 Phase B strongly supported a free-running temporal profile failure,
but its sealed Phase A bundle did not expose a direct local flow-field probe.
The next diagnostic must distinguish failure to fit the flow objective from a
failure introduced by free-running sampling, without changing the action ABI,
normalization, sampler, or inactive universal dimensions.

## Decision

Prepare a nine-probe teacher-forced comparison of `BASE` and `O1@200` using
the same first ordered `TINY1_H2STATS` training snapshot, the same full
`[1,40,132]` CPU-seeded noise tensor, and the same three quantiles of the
official transformed Beta time distribution. Use the official equations:

```text
x_t = (1-t) * noise + t * expert_action
v_target = expert_action - noise
action_hat = noise + predicted_velocity
```

Only the first ten dimensions of the implied universal action may enter the
official processor unapply/physical `[1,40,10]` decode. The active slice is
still in model-space normalization until that canonical decode runs. The 122
padded dimensions remain in the noisy model input but are not modified or
treated as physical output.

The Phase C harness is preparation-only by default. It must not construct or
load a model, run inference/training, submit PBS, execute WBC, or read
validation/final-test targets. A future C1 execution requires a separate
authorization and must record raw velocity, hashes, profile metrics, and
SafetyGate results before any sampler comparison.

C2 sampler comparison at 4/8/16/32 steps is locked until C1 passes and an
official-vs-mirrored four-step parity check passes.

## Rationale

Fixed noise and fixed time eliminate diffusion sampling randomness and most
conditioning drift from the BASE-vs-O1 attribution. The teacher-forced local
field is the smallest diagnostic that can show whether the learned model
approximates the supervised velocity target before a four-step Euler rollout.
It prevents a sampler or inactive-noise intervention from being smuggled into
the objective-fit question.

## Claim boundary

`LOCAL_FLOW_OBJECTIVE_FIT_SUPPORTED` would only support local objective fit on
the tested TINY1 sample and probes. It would not establish free-running
trajectory quality, same-scene receding-horizon success, WBC tracking, real
robot readiness, or generalization. A failed C1 returns
`LOCAL_FLOW_OBJECTIVE_FIT_NOT_SUPPORTED` and stops C2 under this ADR.

## Evidence

See:

```text
reports/STAGE23_PHASE_C_PROTOCOL.md
scripts/stage23_phase_c_flow_field_diagnostic.py
reports/STAGE23_PHASE_B_ATTRIBUTION_AUDIT.md
```

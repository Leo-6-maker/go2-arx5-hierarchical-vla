# ADR-025: Action-Diffusion Profile Attribution

- Status: Accepted for Stage22
- Date: 2026-09-01

## Decision

Compare two matched simulation adaptation recipes:

```text
R0: projector/state/action encoder tuning; diffusion, visual, and language frozen
R1: R0 plus officially exposed diffusion/action-module tuning; visual and language frozen
```

Use the same base checkpoint, dataset, timing, seed policy, and held-out
same-scene labels. Select or reject checkpoints using geometric action-profile
metrics rather than training loss. Record full-horizon path length, speed,
roughness, initial continuity, SafetyGate result, and old/new common-future
overlap inconsistency.

## Rationale

Stage20T/21 localized the remaining deviation to a long, aggressive and
future-profile-inconsistent GR00T action trajectory. Changing provider,
binding, WBC thresholds, or adding asynchronous transport would confound that
attribution. Predicted trajectories are audited unchanged; no clipping,
smoothing, retiming, or checkpoint modification is allowed.

## Claim boundary

Even if R1 improves the profile, the result is limited to the tested
simulation task family and does not establish real-robot readiness, universal
visual grounding, or general manipulation performance.

# ADR-020: Qualify Native Timing Sensitivity Before Model Promotion

- Status: Accepted for Stage20T
- Date: 2026-09-01

## Context

Stage20S executed one sealed actual GR00T output through the provider, safety
gate, binding, preview adapter, and delivered native WBC in simulation. The
model path ended at native `TRACKING_ABORT` near the `0.70 * traj_L` timing
threshold. The error decomposition showed controller-side error much larger
than reference reconstruction error, but the single rollout could not say
whether the cause was model trajectory profile, reference timing, or native
reset sensitivity.

## Decision

Use a matched attribution matrix before changing training, provider, adapter,
or WBC semantics:

```text
A actual model output
B same-observation future desired label
C known-good Stage16 control
```

Run the same checkpoint, simulator, asset, provider/binding path, reset seeds,
and native timing semantics. Only timestamp scales may vary in derived
diagnostics. Preserve the native `traj_s`, `traj_s_ref`, timing error,
lateral distance, termination flags, and pre-reset terminal snapshot.

## Evidence

The 33-run matrix produced 14 reference completions and 19 tracking aborts.
At original timing, A completed 0/3 while B and C each completed 2/3. Seeds
13 and 14 isolate the model path: B completes under the same native conditions
where A aborts. Seed 15 aborts A, B, and C at the initial reset boundary and
is recorded as shared reset sensitivity.

The delivered `to_canonical()` preprocessing is timestep-dependent. Its
existing low-pass branch is active at 0.75x and changes the internal canonical
path length, so 0.75x is not treated as a pure temporal counterfactual. The
unfiltered 1.00x–2.00x results carry the primary attribution.

## Consequences

- The actual model-driven visual-reach rollout is not promoted to simulated
  end-to-end qualification.
- The native controller baseline is frozen; no threshold or checkpoint is
  changed to obtain completion.
- A model trajectory-profile limitation and a reset/initial-condition
  sensitivity are separate findings.
- A production action period remains undecided until dataset/inference/
  execution timing semantics are defined by robot-specific data.
- The next experiment may implement same-scene receding-horizon orchestration,
  but it must preserve the four error terms and fixed-reset control.

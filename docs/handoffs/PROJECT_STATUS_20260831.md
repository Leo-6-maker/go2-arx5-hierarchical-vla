# Project status handoff — 2026-08-31

## Current decision

The active question is whether the delivered native WBC can track externally
supplied E1/E2 EEF trajectories in its own native embodiment. P1M is not the
current baseline because Stage13C exposed a material embodiment mismatch.

Stage14 consolidated the project-owned integration code into this repository
and mechanically injected seven E1/E2 references into the native runtime. The
native runtime stayed finite and completed six of seven requested diagnostic
runs; the orientation-only run ended through the native timing termination
rule. These are preliminary injection results, not a qualified tracking
baseline: E1/E2 files declare `SIM_REFERENCE_ONLY`, while the native runtime
adds its own reset-dependent shoulder/reach anchor, and no explicit transform
or TCP/episode-anchor binding between those conventions is yet available.
The current Stage14 gate is therefore `REFERENCE_INJECTION_BLOCKER`.

## Evidence already completed

| Area | Status | Bound evidence |
|---|---|---|
| GR00T N1.7 | validated with deviation | 132D universal processor; 40-step custom 10D path; full gated inference remains auth-dependent |
| D1/D2/D3 | completed with deviations | public-data audit, kinematic Go2+ARX5 data, Isaac RGB pipeline |
| P1/P1M | sim qualified only | lifecycle, writer, physical render, collision approximation; not real calibration |
| E0/E1/E2 | prepared | controller-independent metrics, 78 E1 references, 36 E2 whole-body geometric references |
| Stage13A/B | completed | safe intake, semantic ABI, checkpoint binding, CPU forward, adapter |
| Stage13C | pass with deviation | native WBC runtime coherent; P1M bridge unqualified |

## Immutable artifact anchors

- delivered WBC archive: `b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167`;
  source commit is `UNKNOWN` and provenance is `LAB_DELIVERED_ARTIFACT`;
- P1M canonical USD: `92d64f697d3f5ea6f17bbeaab72142607019f0309b5068c47f37a623499b9a53`;
- Isaac-GR00T repository: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`;
- SONIC reference repository was studied and remains unchanged.

## Delivered native WBC ABI

External reference is variable-length absolute XYZ + rot6d, transformed into a
nine-point SE(3) arc-length preview with offsets `0.010, 0.025, 0.050, 0.090,
0.150, 0.225, 0.325, 0.425, 0.500 m`. Arm observation is 201D × 60 history;
dog observation is 90D × 30 history; arm output is 15D (6D arm plus internal
9D body/gait plan); dog output is 12D; physical action is 18D. Rates are 50 Hz
control, 200 Hz physics, decimation 4. Gripper is separate.

## Claim boundaries

E2 geometric feasibility is not dynamic tracking. Generator expressivity is not
policy generalization. Native WBC tracking is not P1M transfer. Simulation
collision qualification is not real collision/calibration qualification.

## Next experiment

Resolve the explicit `SIM_REFERENCE_ONLY -> native world/episode` frame and
anchor contract, rerun the seven native continuous-reference trajectories,
then—only if the reference gate passes—run the 40-point interface ablation and
a bounded 24–36 reference subset. Do not use the current unaligned numbers as
evidence for or against EEF-only, E2 whole-body resolution, or native WBC
generalization.

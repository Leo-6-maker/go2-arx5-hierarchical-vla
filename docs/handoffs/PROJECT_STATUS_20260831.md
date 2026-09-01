# Project status handoff — 2026-08-31

## Current decision

The active question is whether the delivered native WBC can track externally
supplied, semantically aligned E1/E2 EEF trajectory shapes in its own native
embodiment. P1M is not the current baseline because Stage13C exposed a
material embodiment mismatch.

Stage14 consolidated the project-owned integration code and mechanically
injected seven E1/E2 references, but those runs were unaligned because the
benchmark placement/TCP convention was not bound to the native reset anchor.
Stage15 introduced `TrajectoryShapeV1`, `PlacementProfileV1`, and
`EpisodeReferenceBindingV1`; an independent implementation reproduced 12
native anchor/preview traces to below `1e-6` scale. The binding blocker is
closed. The aligned diagnostic set is still not broad native evidence: five
nominal cases completed, the native static-relocation case terminated at 25
steps under the native timing rule, and the reused E2 continuous case was
reclassified as native geometrically infeasible.

## Evidence already completed

| Area | Status | Bound evidence |
|---|---|---|
| GR00T N1.7 | validated with deviation | 132D universal processor; 40-step custom 10D path; full gated inference remains auth-dependent |
| D1/D2/D3 | completed with deviations | public-data audit, kinematic Go2+ARX5 data, Isaac RGB pipeline |
| P1/P1M | sim qualified only | lifecycle, writer, physical render, collision approximation; not real calibration |
| E0/E1/E2 | prepared | controller-independent metrics, 78 E1 references, 36 E2 whole-body geometric references |
| Stage13A/B | completed | safe intake, semantic ABI, checkpoint binding, CPU forward, adapter |
| Stage13C | pass with deviation | native WBC runtime coherent; P1M bridge unqualified |
| Stage15 | pass with deviation | native binding self-consistent; native classification/qualification still has one reference-class deviation |

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

Native binding details: the reset anchor is constructed from the native base
pose, arm mount, normalized `[0.36, 0, 0.10]` body direction, reach-table
radius, and `rho_star=0.60`; it is frozen at reset. The controlled TCP
semantic is `x5_link6_gripper_center`: position uses the `gripper_center`
offset while orientation uses `x5_link6`.

Stage15 native reclassification of all 114 E1/E2 shapes is: 70
`NATIVE_ARM_EASY`, 17 `NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT`, 3
`NATIVE_BOUNDARY`, 24 `NATIVE_GEOMETRICALLY_INFEASIBLE`, and 0 native
continuous-base cases. The 16 original E2 continuous cases became native
geometrically infeasible under the bounded native reach analysis; this is a
classification change, not a WBC failure result.

## Claim boundaries

E2 geometric feasibility is not dynamic tracking. Generator expressivity is not
policy generalization. Native WBC tracking is not P1M transfer. Simulation
collision qualification is not real collision/calibration qualification.

## Next experiment

Resolve the native static-relocation qualification issue or explicitly design a
native-valid static/continuous placement set, then run a bounded native
benchmark. Only after that may the 40-point interface ablation and GR00T/WBC
integration be reconsidered. Do not use Stage14 unaligned numbers as evidence
for or against EEF-only, E2 whole-body resolution, or native WBC
generalization.

# PI0160 matched-reset expert/model packet

Status: **PARTIAL, evidence complete for all five q0 inputs**.

Authority: Issue #11 comment `5651703057` (`PI_GATE_STATE_0160`) and PR #69
comment `5651704759`.  The fixed target is `red_target_train_a` at world
`[0.30, 0.10, 0.58]`; target XYZ and condition metadata were excluded from
the GR00T input.  The model input was body RGB, wrist RGB, StateV1, and the
fixed language instruction only.

## Execution

The independent native expert ran the fixed order C0, C1, C2, C3, C4.  C0,
C2, C3, and C4 completed all 245 post-S0 ticks.  C1 captured a valid q0 and
q1, then terminated at segment 1 with the recorded
`TRACKING_ABORT` (49 rows, 51 WBC steps, one reference update); it was not
retried.  This is a valid negative/partial episode, not a reason to tune or
discard its q0.  The fixed N2 seed2223@2000 diagnostic used all five q0
snapshots, once each, with one E4 FR query and three matched local-field
probes per condition.

## Counts and boundaries

| side | construction | conditioning | action-head | total calls | other |
| --- | ---: | ---: | ---: | ---: | --- |
| independent expert | 0 | 0 | 0 | 0 | 1,031 WBC steps; 13 reference updates; 4/5 complete |
| fixed N2 model | 1 | 10 | 35 | 45 | 5 FR + 15 teacher probes; raw first |

GR00T output was never sent to WBC.  Model training, backward, optimizer,
ICL, DAgger, PBS, H200, hardware control, Validation-E, Holdout-F, and new
data collection were all zero/false.  No SafetyGate relaxation or closed-loop
learned-policy rollout was performed.

## Interpretation

All five model FR decodes have negative final target progress (mean
`-0.071259916 m`), while the 15 teacher/local-field probes have descriptive
mean progress `+0.004823820 m`.  The late `t=783` probes are positive in all
five conditions; this is a fixed-policy/local-field observation, not a task
success threshold or a winner claim.  FR path length is about 1.856 m with
roughness about 0.0427 m; teacher paths shorten as the flow time increases.

The four complete expert episodes reach the same declared target reference
under the native expert, but the achieved trajectories terminate around
0.719 m from the target in this WBC/reference diagnostic.  C1 has no complete
expert metric file because it terminated early; its q0 is still a valid
matched model input.  `start_residual_m` is zero for every model row.

The q0 body and wrist images are byte-identical across the five resets (body
SHA `8a04fca6a41ae409b5c75d905fe8b2ae95c2aafc1b203850fc9ac43c093595ba`, wrist
SHA `205f0e6b8d03232d009a3d149558ae1b62ba4099cb1d07fdb0b4beb2c51b8334`),
whereas StateV1 and measured EEF differ as intended.  Therefore this packet
does not support a visual-viewpoint effect claim; the reset comparison is
primarily a StateV1/measured-EEF comparison under the native zero-settling
capture boundary.  Normalized clipping is reported separately from model
error: the five FR label bundles contain 7, 7, 8, 5, and 9 clipped active
scalars respectively out of the frozen 190 active T16 scalars.

## Single recommendation

Authorize one fixed-budget visited-state independent-expert/local-field
supervision intervention using the matched q0/after-execution pairs.  Keep
the checkpoint, stats, noise, decoder, and SafetyGate fixed; do not start
ICL or broad retraining automatically.

The raw arrays remain node-local and immutable.  See `matched_summary.csv` and
`evidence_manifest.json` for the exact paths, run heads, and evidence hashes.

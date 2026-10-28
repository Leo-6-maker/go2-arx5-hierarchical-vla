# PI0267 fixed CPU liveness return — PRIMARY_PARTIAL

Authority: `PI0267_EXECUTOR_DECOUPLED_RECOVERY_LEARNING_GO_V1`, release `f21bca4ec5b004e968f285032fc57df904deebef`, canonical 5937193116.
Branch: `pi0267/executor-decoupled-recovery`, parent `48048ae37b76534bbbe8a0ba76a9c84ed793e7ea`.

## Fixed results

| Teacher | Executor | M1 | M3 | Total |
|---|---|---:|---:|---:|
| OLD | E0 exact | 8/8 | 1/8 | 9/16 |
| OLD | E1 lag 0.15 s | 8/8 | 1/8 | 9/16 |
| NEW | E0 exact | 8/8 | 1/8 | 9/16 |
| NEW | E1 lag 0.15 s | 8/8 | 1/8 | 9/16 |

All 64 original conditions completed 40 s with 100 repeated 0.4 s prefixes. First-valid results are retained, including every failure. No parameter/seed/geometry/score search followed the result.

Required NEW E0 16/16 is not met; required NEW E1 >=12/16 and each task >=6/8 is not met. Verdict: **PRIMARY_BLOCKED_TEACHER_LIVENESS / PRIMARY_PARTIAL**. This is a teacher/path-contract result, not a GR00T fine-tuning quality failure.

## What failed

Only family-000 / M3 command-0 passes, in both executors. The other seven M3 conditions fail in both OLD and NEW matrices. All NEW M3 endpoints are reached within 15 nm at 40 s, but endpoint accuracy cannot replace entry/order/direction qualification.

In five failing M3 geometries/commands per NEW backend, V2 first admits entry with directed angle already beyond its -1 mm path deadband; reverse/gap becomes true at that admission. In two others, continued approach before the true arc start produces a later negative directed excursion. The saved actual-state decomposition reports 2.45–3.97 mm maximum backward excursion for failing NEW cases. For this fixed canonical virtual initial pose, entry is admitted during the source curve's approach stage, with current-replan source phase <4 s. It is not evidence of a unique WBC/neural root cause. The original scorer and source geometry are unchanged; no entry rule, pose, path, phase history, or tolerance is revised after observing this result.

Large changes in source-time projection are separately saved. Source time is quintic-paced metadata; its difference is not physical directed arc progress or a measured tracking error. Neither executor uses task geometry or the scorer to modify execution. E0 tracks the linearly interpolated H40 / quaternion shortest interpolation at dt .02; E1 uses exactly the registered first-order law. Neither is robot dynamics or a second embodiment.

## Source and finite S0 closure

The fixed four DEV geometries and source canonical initial TCP are sealed before S1. The canonical initial rotation is identity. This is the declared virtual design; it is not the measured PI0266 WBC reset. The historical post-reset pose remains bound separately in the S0 raw evidence. General nonidentity/noncommuting rotation, anchor, repeated-input identity and terminal contraction are checked in formula fixtures.

S0 inspects 16 historical DEV diagnostics, first/middle/final frame ticks (0/1000/1980), 48 prefixes total. Requested H40 or once-installed CLOCK references, native installed same-tick samples, actual current/next TCP, projection metadata and reference-only install events are recorded separately. Physical state/history is preserved by the retained install receipts. At OBS replanning boundaries the next chunk overwrites the rotation sample: the original previous-installed position is retained, but that boundary rotation is explicitly excluded rather than reconstructed. Source hashes bind every selected proposal, table and frame.

The maximum sampled requested-to-installed position difference is CLOCK 0.000000344 m and OBS 0.000906037 m. These are finite interpolation/binding/timing comparisons, not a blanket installation-defect verdict. Actual-to-installed tracking is calculated directly at matching ticks. The supplied 1-D toy is reexecuted and matches its frozen numerical JSON; its 0.5 mm bias is chosen, not identified empirical WBC dynamics.

Staged reference/scene wrapper byte hashes differ from the pinned Git versions; the actually consumed `reference`, `quintic` and `scene` function ASTs are identical and both byte identities are recorded. The remaining selected source files match the pinned release hashes. A NumPy bool JSON serialization repair occurred before any matrix episode; it changes neither the formula nor parameters. Original and repaired source bindings are preserved.

## CPU reproduction and counters

All 64 raw episode arrays replay through an independent quaternion interpolation and NumPy Rodrigues executor path. NEW labels also replay through an independent piecewise 4D metric projection and scalar yaw formula for the actual canonical/yaw-only matrix. Worst actual position replay difference 1.25e-16 m; worst rotation 1.39e-16 rad; independent new-label position difference 2.8e-08 m. These are numerical reproduction checks, not new episodes. The unchanged PI0266 V2 scorer is shared, and is explicitly not claimed independent. Original negative/positive scorer fixtures plus a missed-entry negative pass.

Completed CPU matrix: 64 episodes / 128000 dt-.02 mathematical updates / 6400 H40 plans / 2560 mathematical seconds. The constructed six toy traces and producer formula fixtures are separate CPU evidence. One serialization failure precedes episodes. Rendering proposal count=0; new policy-frame pairs=0; GR00T queries/heads=0; optimizer/canary updates=0; WBC starts/physics steps=0. Budget is not transferred to a new hypothesis.

## Conditional stop and omitted downstream work

S2 virtual rendering/dataset/statistics: NOT_RUN_FAILED_S1_ADMISSION.
S3/S4 canaries/N4096/R4096/checkpoints: NOT_RUN_FAILED_S1_ADMISSION.
S5 192 virtual model episodes / 768 offline / legacy new panel: NOT_RUN_FAILED_S1_ADMISSION.
S6 eight expert +24 model WBC transfer: NOT_RUN_FAILED_PRIMARY_DEPENDENCY.
No EVAL image, label or score payload is opened. No new donor/model materialization, training submission, camera tuning, replacement family, physical trial, cleanup, mail, monitor, PR77 pointer or merge.

PI0266 remains PARTIAL with its already verified public archive unchanged. Donor L2048 is unchanged/protected. Formal selected_checkpoint=NONE and deployment_selected_checkpoint=NONE. No virtual research candidate is qualified. The protocol-authorized early scientific stop is delivered; the primary learning study is not called COMPLETE. A changed entry/path contract would require new scientific authorization.

# PI0280 — pristine pretrained baseline closeout

Authority: Issue #11 [6037445331](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6037445331), release `9c0172ced33c6052f7cb037b4a6d91910f9fe69e`.

## Verdict

The authorized nonblocked baseline lanes have been executed. **The requested GR00T + Go2/ARX5 object-task closed loop was not executed.** Two independent blockers remain: unsupported target observation semantics and the delivered frozen WBC's original tracking termination. They are not four target-policy quality failures. No training or automatic return to PI0279 is permitted by this result.

| Lane | Execution | Quality / scope |
|---|---|---|
| A0 predecessor pause | Confirmed on Corelab and Hopper; old recoveries and raw accounting preserved | PI0279 paused, not relabelled FAIL |
| A1 pristine original | Original NVIDIA GR00T-N1.7-3B, original processor/statistics/Cosmos and effective tensors verified | No task-adapted or NEW-slot substitution |
| A2 native DROID | 32 observations × stock/wrapper = 64 actual calls; all 32 exact same-runtime comparisons pass | Open-loop reproduction, not task completion |
| A3 source Panda | Four complete 32 s task episodes, plus 1 s real-history pre-roll each | Persistent StackCube success **0/4**; narrow developed OOD source simulation |
| B1 WBC hold | Full 10 s fixture executed | A control fixture, not an object task or universal pose PASS |
| B1 WBC vertical return | Original native `TRACKING_ABORT` at about 0.5 s | Valid executor qualification failure; original threshold retained |
| B1 target task replays | Four expert and four recorded-model conditions not executed | `BLOCKED_NATIVE_EXECUTOR_QUALIFICATION`, quality null |
| B2 fresh target loop | Not executed | `BLOCKED_TARGET_PRETRAINED_INPUT_CONTRACT` |

`selected_checkpoint=NONE`, `deployment=NONE`. Baseline execution and useful task performance are separate. A manifest or source-Panda result does not complete the target closed loop.

## Model identity and native reproduction

- Model revision: `2fc962b973bccdd5d8ce4f67cc63b264d6886495`, bound in `execution/design/BASELINE_BINDING.json`.
- Official source: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
- Cosmos: `9ce19a195e423419c349abfc86fd07178b230561`; verified local file-location rebind only.
- Original BF16 inference, FP32 stock decoding, S4, H40, 132 output dimensions. Effective weights unchanged through native and source-physics execution.
- A2 median actual call latency: 0.5487 s. Source physical-loop median inference latency: 0.4436 s; simulation pauses while waiting, **NON_REALTIME**.

Original descriptive MSE, mean across the fixed 16 anchors per genuine episode:

| Episode | EEF9D MSE | Gripper MSE | Joint MSE |
|---|---:|---:|---:|
| 1 | 0.01451957 | 0.02109526 | 0.01129253 |
| 2 | 0.24403801 | 0.04408368 | 0.07580955 |

EEF9D mixes position and rotation representation coordinates; these are the original descriptive definitions, not a new physical success gate. Joint outputs are diagnostic and never sent to Go2. See `execution/native/RESULT.json` and `execution/evidence/NATIVE_CPU_READBACK.json` for every anchor, actual noise identity and independent saved-array parity.

## Source-physics result

| Condition | Task duration | Queries | Native IK holds | First success | Persistent success |
|---|---:|---:|---:|---|---|
| 10000 | 32 s | 60 | 9 | none | false |
| 10001 | 32 s | 60 | 191 | none | false |
| 10002 | 32 s | 60 | 0 | none | false |
| 10003 | 32 s | 60 | 122 | none | false |

Independent command-to-achieved endpoint position RMS was respectively 7.765, 20.489, 3.629 and 19.541 mm. This measures local reference tracking, while the object predicate remained false throughout every episode.

All conditions retain their full denominators. The unchanged gate produced 239 ACCEPT and 1 WARN; no REJECT was installed. Original source IK holds remain visible, without teacher takeover or a scripted finish. Independent CPU computation uses the original object predicate at all 41 task-time samples 28–30 s, after subtracting the measured 1 s history pre-roll.

The native DROID converter's Euler conventions and fixed Panda TCP transform were reconstructed from pinned source and 677 genuine source frames. Static-transform variation was below 5.7e-8 m and 2.8e-7 rad. Input reconstruction and timestamped first-eight-sample installation audits also passed. This does not remove the camera, gripper and task distribution shift: DROID's real tool and the source Panda simulator are not identical. Four developed OOD failures cannot establish that the pretrained model lacks manipulation knowledge in general.

## WBC findings and transfer boundary

The actual delivered older controller was used: bound actor/config/asset hashes, full Go2/ARX5 dynamics, native 0.005 s simulation and decimation 4. WBC weights, gains, spatial preview, safety and termination limits were retained.

The hold's final/max same-tick position error was 7.049/16.239 mm. Its actual orientation differed from the raw requested quaternion by 1.074/1.210 rad final/max. Installed orientation was not recorded in this original hold attempt and remains **UNOBSERVED**; this cannot alone identify a rotation-conversion defect. URDF hard joint-limit penetration was absent in saved rows, but inherited soft-limit margins were negative for arm joint x5_joint4 and the gripper. These observations must not be erased by reporting position alone.

The vertical-and-return fixture's spatial progress reached 0.100000 m at approximately 0.5 s while its time-law progress was only 0.001224 m. Timing error 0.098776 m exceeded the unchanged 0.7 × path-length limit, 0.070000 m. The original `TRACKING_ABORT` fired; it was not a fall or hard joint-limit termination. This is evidence about the delivered executor's progress handling on this returning path, not proof of a unique rotation root cause.

In the replacement attempt, the newly recorded installed quaternion matches the requested raw quaternion within 2.98e-8 rad. Actual-to-installed orientation error is still 0.988597 rad at termination. That separates correct recorded orientation installation from achieved physical tracking; it does not make the executor qualified.

One earlier same-condition attempt hit an index error during automatic reset after the native stop. Its raw evidence and counts are retained under `vertical-engineering-000`. The minimal wrapper fix captures the original termination before automatic reset; the one engineering replacement reproduced the native stop. No further quality retry or threshold/path/gain repair was performed.

The target scene transform was frozen from geometry before source policy scores; no target task scene was instantiated after failed executor qualification. Its four expert and four recorded-model slots remain blocked, not 0/4 or 0/8.

All eight serialized pretrained input routes were inspected. DROID requires actual source-robot joints and real history; the other routes require unavailable embodiment-specific fields. ARX5 angles cannot honestly populate Panda joints. The B2 input block therefore remains independent of the WBC failure. No fabricated joints, duplicate history, shadow Panda state or placeholder NEW slot was used.

## Accounting and artifacts

| Counter | Actual |
|---|---:|
| Policy calls / S4 heads | 304 / 1216 |
| A2 / A3 calls | 64 / 240 |
| Physical starts | 7: four source + two nominal WBC fixtures + one engineering replacement |
| Source control / integration / RGB pairs | 2640 / 13200 / 2640 |
| WBC reserved control / integration / actor / RGB pairs | 553 / 2212 / 1100 / 110 |
| New backward / optimizer / training updates / statistics fits | 0 / 0 / 0 / 0 |
| New task adaptation dataset production / hardware commands | 0 / 0 |

WBC ledger reservations include partial engineering execution and are not all completed saved control rows. Source A3 elapsed wall time was 891.356 s. Final OS rusage was not retained by its original launcher; measured intermediate CPU snapshots remain, and missing final CPU/GPU utilization is not estimated. Model work used Corelab GPU0; no new PBS or H200 training was started. Raw source output size was 1,356,404,250 bytes before final audit additions.

Full RGB arrays and original traces remain on Corelab. Compact evidence and clearly labelled actual videos are archived separately; the fixed revision/full-file readback receipt is `execution/evidence/COMPACT_ARCHIVE.json`. Local source videos are under `execution/source/videos/`; WBC attribution and real videos are under `attribution/`. No paused recovery or failure evidence was deleted.

## Next decision, not new authorization

The preparation worker has produced a bounded EEF/gripper-interface draft, marked `launch_allowed=false` and `NOT_AUTHORIZED`. It keeps object tasks central, separates source/interface-valid reference learning from target execution, and makes task-data admission and retention independent of target-WBC success. It does not initialize a slot, train an adapter, collect new data or reopen this baseline. The current result stops for review of the input and executor boundaries.

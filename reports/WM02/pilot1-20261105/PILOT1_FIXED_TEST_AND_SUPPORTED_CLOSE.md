# Fixed TEST result and a real supported close–hold–release

The frozen free-space Actor did not generalize reliably: V0 and V1 each succeeded on one of four TEST goals, versus three of four for frozen ARX. Separately, the new supported-contact curriculum completed a real two-sided native grasp, **1.2 seconds of continuous supported hold and 1.836 seconds of terminal release**. This is a supported pinch/release microtask, not full grasp/lift/carry/place.

## Frozen P23/P24 TEST

The original goals, success threshold, order, WBC, native binary and V0/V1 checkpoints remained frozen. There were two single-scene qualifications and 20 method evaluations on two TEST physical parents, using two isolated CPU-native slots. P23/P24 data were not fitted or relabeled TRAIN. All 20 conditions physically executed exactly once; no failed condition was rerun to obtain PASS.

| Method | Recorded task successes / attempts | Native PASS | Recorded mean XY error |
|---|---:|---:|---:|
| Frozen ARX | 3/4 | 4/4 | 0.1293mm |
| TRAIN mean response | 3/4* | 3/4 | 0.1185mm* |
| Geometric rule | 0/4 | 4/4 | 0.7600mm |
| Actor V0 | 1/4 | 4/4 | 0.7293mm |
| Actor V1 | 1/4 | 4/4 | 0.5523mm |

*One TRAIN-mean P23 trial finished all physical steps and recorded a successful task score, then failed while reading the final pause acknowledgement. Its native FAILED report remains FAILED. Strictly requiring both native PASS and task success gives **2/4 for this method, with one recording failure**, rather than promoting that failure to a valid runtime PASS. The table's recorded physical error includes all four observed endpoints and explicitly retains this distinction.

The independent sealed-record check found all 64 executed task-atom consumption records and complete pause-sealed traces for every trial, including the finalization failure. It did not issue a new native qualification or rewrite any result. Both coordinators and all child/batch leases are CLOSED; an independent process check found no matching owned native process alive.

The complete comparison, all 22 trial identities, original data-file hashes and worker evidence are in [FREE_TEST_FINAL_SUMMARY.json](FREE_TEST_FINAL_SUMMARY.json). Its canonical source-data identity is:

`53d327fd1741d6002912619832bcfc82125feb4742c20a5a26c3c55441df60e6`

The first segment used source `5f329b9b3128e26f5393592f02585918e794e98d`; the untouched P24 continuation used snapshot-read repair `e13344f7eb95453ee93a8870f58f8f15cf42cfba`. New code only bounded live-snapshot reads; it did not change the task or authorize another native tick. The 22 native jobs used 90,266 ticks and 1,478.60 CPU seconds; accumulated native worker wall was 491.84s, counting each two-slot batch once. Coordinators added 1.40 CPU seconds. The first unit measured peak RAM of 2,792,656,896 bytes. Main WM/Actor updates for this evaluation were zero.

The practical conclusion is a student generalization gap, not a new WBC failure or a proof of its cause. ARX exceeded the geometric rule in this tiny panel, but no learned Policy advantage over the strong response rule is established. The earlier V1 DEV result 3/4 remains true; it did not reproduce on these TEST parents. Future recipe selection must use TRAIN/DEV and a new unexposed final TEST, not reuse P23/P24 as a blind panel.

## Supported contact: new geometry and real native qualification

Native TRAIN geometry showed that the old side-contact object lay outside the right finger. Closing there would leave the object. A proposed centered C07 layout was statically rejected because the wrist and table intersected the original startup trajectory. No physics was launched for this rejected layout.

A finite TRAIN-trajectory grid checked X=0.280…0.305m and 2/5mm support half-thicknesses, preserving all robot dynamics and native contact rules. Four of twelve layouts were statically feasible; the first feasible layout was frozen as SC07B: cube `[0.300,-0.01725,0.704]`, support center `[0.301,-0.01725,0.682]`, half-size `[0.022,0.022,0.002]`. The cube genuinely settles on this fixed support. Geometry selection used the C02 TRAIN trajectory and no simulated contact-force labels.

Scene compiler: `26a35bf69f57d5ce21e7157cfab4acabb0df375e`.
Scene SHA256: `dca8d261e525175ba3c5526a1bb101a5a126ea64402de586bd6ed7f2b77c0ab4`.
Exact frozen robot/dynamics/safety contract: `5495b40fd499cdce97c78724cce49fa2eec9809b16e7aefc6559b06d6b5f6898`.
Native C08 binary: `29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2`.

SC07B then passed a separate original native single-environment open-prefix qualification: 4,103 ticks, exit 0, no STOP and clean shutdown. This qualification did not claim grasp success.

## Closed-loop native close, hold and release

The [protocol](../../../experiments/wm02/pilot1_supported_close_v1.json) was frozen before physical closure. It holds the same initial EEF reference; only the original gripper working target changes. It retains the native 0.04m/s target-speed rule, gains, forces, mechanical limits, robot parameters, 2ms physics and existing safety. After 64 open/steady atoms it commands logical opening 0.4, attempts 32 close atoms, 24 supported-hold atoms, then releases for 48 atoms. A loss of current supported grasp during hold triggers an open command and preserves task failure.

Every native tick is read from the original journal. The hold predicate is the original native grasp definition—both pad forces, opposing normals, relative motion and valid finger state—plus actual table support on every tick. Required continuous hold is 500 native ticks; release requires at least 100 terminal noncontact, open-finger, table-supported ticks. No geometry overlap is substituted for a force event.

The first commissioning attempt, source `a69ecdd96134f0de28f291a1fec026582aeee074`, reached 4,453 ticks and 78 actual task atoms. Real two-sided grasp appeared, but a renewal failed the cloned reference certificate: the planner held the already grasped object static while solving the nominal EEF pose, predicting 4.294mm pad penetration beyond the unchanged 4mm bound. No native STOP or outer rejection occurred; native shutdown was clean. The unsubmitted/rejected reference is not an executed training label. This failure and its cost remain preserved.

On that exact sealed state, the existing `ACTUAL_GRASP_ATTACHMENT` **planning-copy** hypothesis passed all 1,051 certificate points with zero integrations, unchanged Cartesian references, and unchanged contact/rate limits. Repair `acec4861fdf8ddf18e755e6fea74a303e3529749` routes this existing hypothesis only after the actual original native grasp predicate is true. It changes neither the live object nor physics constraints. Actual object dynamics remain free; no weld, teleport, fabricated grasp or WBC compensation was introduced.

A fresh single-environment commissioning worker completed:

| Evidence | Actual result |
|---|---:|
| Native ticks / actual task atoms | 6,703 / 168 |
| Bilateral native grasp ticks | 1,128 |
| Continuous supported hold during HOLD stage | 600 ticks = 1.2s |
| Terminal actual release | 918 ticks = 1.836s |
| Pad contact-positive ticks | 1,156 |
| Native status / worker exit | PASS / 0 |
| Native STOP / outer rejection | 0 / 0 |
| Resets / teleports / actual object attachments | 0 / 0 / 0 |
| Full manipulation | Not established |

The successful worker used 147.78 CPU seconds and 77.23 wall seconds. The prior failed commissioning used 77.19 CPU seconds / 44.67 wall seconds; the open qualification used 62.25 CPU seconds / 33.30 wall seconds. This is a hand-programmed microtask with measured state guards, not a newly trained supported-grasp Policy. WM/Actor updates in this subtask are zero. The earlier trained exit Policy and its separate DEV/TEST 2/2 result remain separate evidence.

The sealed supported-contact data identity is `99f033e33ee0218ccc836ad1a77ff60fc7c7f23aafbec43234cc954b713e251f`.

[SUPPORTED_CLOSE_SUMMARY.json](SUPPORTED_CLOSE_SUMMARY.json) preserves the positive and negative outcomes, exact sources and costs. The source regression set now passes 192 tests; five supported-close tests and five snapshot tests also pass under `-O`. Software tests do not substitute for the native qualifications above.

## Next scientific work and actual continuation

The original P18/P19/P20 TRAIN parents had not yet been collected. The [coverage experiment](../../../experiments/wm02/pilot1_train_coverage_v2.json) completes their already frozen goals and method order, adding V1 beside V0: three scene qualifications, then 30 actual TRAIN feedback conditions in two native CPU slots. It does not rerun completed P17/P23/P24 conditions. The original task, goal generator and 0.1858445664mm success threshold remain pinned; the Actor trainer now supports a frozen task file instead of recomputing the task from expanded coverage.

The first TRAIN coordinator completed its three real qualifications, then its spec constructor omitted the per-scene qualification reference for `TRAIN_FEEDBACK`. Admission stopped before any feedback registry entry, lease, worker or physics. Independent reconciliation sealed the original intent as ABORTED with zero additional target effects; it did not turn it into a successful job. A fresh corrected coordinator reuses only the CLOSED qualification files as evidence and receives new leases for the untouched 30 conditions. No active capsule is patched.

Matched follow-up recipe: same frozen ARX, fresh identical seed/architecture/optimizer and 2,000 updates per arm, old versus expanded TRAIN coverage, TRAIN-only normalization and fixed DEV. Those model fits have not started yet. P23/P24 TEST remains excluded from fitting and future blind selection. Supported-contact WM/Policy learning from the new bilateral data is also pending; the successful physical sequence is not labeled as learned control gain.

Actual worker entries use the existing registered private specs, not these examples as grants:

```bash
PYTHONPATH="$WM02_SOURCE/src:$WM02_SOURCE" "$WM02_TARGET_PYTHON" -B \
  -m scripts.wm02.pilot1_native_worker --spec "$WM02_SUPPORTED_RUN/EXECUTION_SPEC.json"
PYTHONPATH="$WM02_SOURCE/src:$WM02_SOURCE" "$WM02_TARGET_PYTHON" -B \
  -m scripts.wm02.pilot1_parallel_worker --spec "$WM02_NEW_TRAIN_BATCH/EXECUTION_SPEC.json"
```

Do not rerun CLOSED job specs. Current work remains local, with bulk data/cache/model output on the qualified non-root disk. No H200, GPU physics, distributed backend or replacement of frozen WBC was started.

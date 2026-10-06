# PI0277 R3 final scientific verdict and local handoff

**Execution complete; quality not qualified. FOCUS4096: TRAIN4 3/4, DEV8 3/8. Status: NOT_USABLE. STOP_FOR_NEXT_TARGETED_TASK_DECISION.**

Authority: Issue #11 canonical 6021627390, release `a5a7e569684d52170f02b0819ae006a7587800dd`, main R3 protocol and conditional gripper-bound addendum. This is the existing `pi0267/executor-decoupled-recovery` lineage. `selected_checkpoint=NONE`, `deployment=NONE`, `sprint_candidate=NONE`.

## Actual physical results

Every final TRAIN and DEV episode completed full 32 seconds / 640 controls. Counted success requires the original native on-top/static/not-grasped predicate at all 41 control samples from 28 through 30 seconds. First success, lift and on-top observations are separate diagnostic events.

| Panel | Seed | Sustained 2 cm lift | On-top observed | First native success (s) | Persistent success | IK holds |
| --- | ---: | --- | --- | ---: | --- | ---: |
| TRAIN | 10000 | yes | yes | 7.90 | PASS | 0 |
| TRAIN | 10001 | yes | yes | 5.20 | PASS | 0 |
| TRAIN | 10002 | yes | yes | 5.20 | FAIL | 83 |
| TRAIN | 10003 | yes | yes | 4.70 | PASS | 0 |
| DEV | 30000 | yes | yes | NONE | FAIL | 0 |
| DEV | 30001 | yes | yes | 5.35 | PASS | 0 |
| DEV | 30002 | yes | yes | 4.70 | FAIL | 0 |
| DEV | 30003 | no | no | NONE | FAIL | 455 |
| DEV | 30004 | yes | yes | 27.75 | PASS | 0 |
| DEV | 30005 | yes | yes | 25.80 | PASS | 0 |
| DEV | 30006 | no | no | NONE | FAIL | 0 |
| DEV | 30007 | no | no | NONE | FAIL | 536 |

TRAIN met its >=3/4 admission gate; the first FOCUS DEV8 was therefore executed. DEV failed its >=6/8 gate. No candidate was frozen and no CONFIRM16 condition was instantiated or executed. All eight DEV conditions and all five failures remain in the denominator. No replacement, additional checkpoint selection, teacher retry or third lineage followed.

## Learning and mechanism evidence

All earlier measured TRAIN panels remain: GRIP512/2048/4096 and FOCUS512/1024/2048 each 0/4. GRIP1024 remains unmeasured. Old failures are not promoted by the final TRAIN improvement.

The earliest authorized suffix-only trigger was FOCUS1024: 3/4 full-H40 envelope failures with no first-eight-command violation. The original job stopped at a complete optimizer boundary, update 1802. Its complete state was resumed with the once-only objective phase:

`L_old + 0.1 L_generated_gripper + 0.1 L_box`

The uniform valid H40 box term used the fixed [0,1] command interval and 0.05 scale only for updates 1803–4096: 2294 updates / 4588 microbatches. Optimizer, cosine/global step, RNG, remaining visit order, teacher, gate, S4 and controller were preserved. The first continuing normal update recorded finite nonzero box-fraction gradients and finite four-step total-objective gradients. Separately isolated box gradients at every Euler state were not queried or claimed.

FOCUS2048 and FOCUS4096 had no envelope terminations in their fixed TRAIN4 panels. FOCUS2048 reached 3/4 sustained lifts and 2/4 on-top, but 0/4 persistent success; FOCUS4096 reached 4/4 lifts/on-top and 3/4 persistent success. This same-lineage observation combines additional training and the declared objective phase; it is not an isolated causal estimate of the box term.

The final TRAIN10002 failure is specific: on-top began at 4.73 s and ended at 17.74 s; no qualification-window success remained. Its first IK hold was later, at 27.6 s. Late IK holds do not explain the initial loss of on-top by temporal precedence alone. Contacts, joint limits and command/installed/achieved errors are retained; their correlations do not establish stiffness, grasp geometry or a unique controller cause.

A joined **1800-row TRAIN query/installation table** retains targets at 0/7/8/15/39, requested increments, commanded and actual opening, actual TCP before/after, object lift/grasp/drop, raw-to-installed and installed-to-achieved errors, IK spans, saved contacts and original actuator transport. The joined **112-row FIT table** uses actual stored noise hashes across all seven measured points. No teacher phase or object truth entered policy inputs. Object truth was used only for saved-array reconstruction.

Original actuator transport clipping/saturation remains present and measured. No new raw-model output clamp or repair was introduced. The earlier milestone shorthand that output was not clamped refers to added model-output intervention, not removal of the pre-existing actuator transport.

## Verification and cumulative budget

The terminal FOCUS audit covers all 8192 microbatches: frozen input/label identities, observed actual noise, FP32 training queries/Euler states with native BF16 epsilon, full S4, restored modes and 471 trainable / 559 frozen scope. Raw-noise regeneration was not performed. Separate final tensor-payload comparison proves all 559 frozen payloads byte-identical to GRIP4096.

Independent saved S4/decode/fresh-input/native CPU replay passed for all **336 final TRAIN/FIT outputs** and **640 DEV outputs**, with zero new model or physics calls. Receipt hashes are indexed in `CPU_FOCUS4096_TRAIN_RECEIPT_INDEX.json` and `CPU_FOCUS4096_DEV_RECEIPT_INDEX.json`. Earlier noncandidate FOCUS512/1024/2048 full S4 replays remain explicitly deferred; their native/input/noise decision proofs and failures remain valid.

Current R2 evaluation and R3 execution/delivery code seals match. A separate predecessor-source inventory check found two stale helper hashes in the preserved PI0276 manifest: `run.py` and `source_worker.py`. All other 466 native/project files match. The actual source worker is exactly the version already bound by the R2 seal and committed at `78d42f1a`; protected gate, actuator, initialization, step and finish functions are AST-identical to the predecessor version. The differences are the prior same-control observation cache/return and a source-teacher horizon exception; the latter is not used by these learner episodes. R2 installs its own minimal `run` module. This is recorded in `PREDECESSOR_SOURCE_MANIFEST_RECONCILIATION.json`; the old manifest is not relabelled all-files PASS or rewritten.

| Shared PI0277 ledger | Actual | Cap |
| --- | ---: | ---: |
| Formal updates | 8192 | 8192 |
| All completed updates, including qualification | 8232 | 9048 |
| Training heads | 148263 | 172032 |
| HOP CPU input attempts, including prefetch/engineering accesses | 35840 | 65536 |
| Conservative combined CPU attempts, adding two read-only handoff example attempts | 35842 | 65536 |
| Quality requests, including 128 inherited donor FIT requests | 2680 | 14336 |
| Quality heads | 10720 | 57344 |
| Physical starts | 36 | 234 |
| Controls | 19464 | 150000 |
| Physics steps | 97320 | 750000 |
| RGB pairs | 19471 | 155000 |
| New teacher controls / WBC / hardware | 0 / 0 / 0 | unchanged |

Counters and raw hashes are in `FINAL_HOP_RAW_BUDGET.json` and `FINAL_CORE_RAW_BUDGET.json`. PI0276 historical source experts/donor outcomes remain reused external evidence; these counts describe the shared PI0277 execution ledger, not all prior projects. The budget was not reset.

## Timing and performance

FOCUS job 651332 ran 01:17:27–02:45:28 SGT and ended at the exact update-1802 boundary. The same-lineage bound continuation 651442 ran 02:51:07–04:09:09 SGT, terminal F / exit 0. Both were before the 07:00 no-training boundary.

Final transfer plus target readback took about 839 s from first target shard-lock metadata to atomic READY; this is not pure network time. Recorded phase wall times were TRAIN/FIT panel 356 s, TRAIN decision 5.5 s, TRAIN full CPU replay 12.1 s, DEV8 panel 693 s, DEV decision 10.0 s and DEV full CPU replay 19.2 s. These phase intervals are boundary-file metadata measurements. Actual native event spans are also recorded. Separate model-load time was **UNOBSERVED**; it was not estimated from panel duration.

The delivered cached-output example was separately exercised by a read-only loading observer: the current model load, BF16 CUDA placement and synchronization took **2.680 s**. It reused the original FIT output, with zero parameterized neural forward calls and unchanged quality ledgers/output hashes. This is a current handoff reload measurement, not a reconstruction of historical panel startup. Two engineering load attempts are retained: the first overly broad guard blocked a parameterless CPU image transform; the corrected guard allowed image preprocessing while forbidding parameterized neural forwards. Neither attempt generated a prediction or consumed a training update.

Production median update time was 1.733 s and loader wait 0.000224 s. The qualified FAST_CACHE2 input benchmark improved input throughput from 8.86 to 139.67 items/s; it is not an overall H200 speedup estimate. Peak allocation was 34,384,967,680 bytes, not a utilization percentage. No additional worker sweep, batch/precision change, compile, optimizer change or fewer S4 steps was introduced. Final S4 query latency median was 0.498 s across 976 existing outputs. Execution pauses simulation for inference and is not claimed real-time.

## Actual local handoff and retention

Corelab handoff directory:

`/home/liuyu-nfs/Projects/go2-arx5/PI0277/R3/handoff/FOCUS4096`

It contains the final handoff index, policy contract, measured input example, cached-output inference command, selection trail, TRAIN query table and FIT comparison. The active policy view is **model-only**, at `/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/R2/models/FOCUS4096`; it is not an optimizer recovery source.

Hopper retains the latest two complete FOCUS Trainers, full-file readback PASS:

- `R2_FOCUS/FOCUS/main/training/checkpoint-3072`: tree `9aaeaca7a15a2ed21258e487a6c809dab220f774d49036f4ad61d5d1125bbcd9`.
- `R2_FOCUS/FOCUS/main/training/checkpoint-4096`: tree `93e500e8ecb6a719d4a53be09d23b35bdeeb83fcc61e75416247a2c0380f4604`.

Both paths are under `/scratch/e1192548/Projects/go2-arx5/experiments/PI0277`; the sealed bound runtime is `R3_BOUND`. GRIP4096 parent and small evidence remain protected. Current asset inventories distinguish complete Trainer, model-only and retired metadata. Listed assets occupy 121,655,394,304 unique-inode allocated bytes on Hopper and 66,706,468,864 on Corelab; hard-linked logical sizes are not added as physical usage. Current personal quota headroom is unobserved/unknown and is not inferred from shared free space.

All **12 final saved-RGB videos** (4 TRAIN, 8 DEV), each 640 frames, exist and have full SHA256 readback. First TRAIN success/failure and first DEV success/failure are explicitly indexed; failures were not omitted. They are actual source-Panda simulation recordings, not synthetic success or target-WBC footage. Video paths and checksums are in the handoff index and video receipts.

HF bulk, full matched CONTROL and large extra panels remain **DEFERRED_BY_R3**, with local originals retained. No unique recovery source was deleted and no automatic post-closeout archive/training restart is scheduled. Public claims remain limited to this single fixed-language ManiSkill StackCube source-Panda experiment; no Go2/WBC, hardware, language generalization or deployment success is claimed.

The terminal decision follows the original gates: **TRAIN admission passed; DEV qualification failed; confirmation not admitted.** The scientific dependency chain is stopped, while actual learning progress, failures, complete local models and evidence are preserved for the next targeted decision.

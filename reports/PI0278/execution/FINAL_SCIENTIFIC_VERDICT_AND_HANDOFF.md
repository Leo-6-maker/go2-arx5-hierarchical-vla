# PI0278 final scientific verdict and local handoff

**Execution COMPLETE; quality VALID_FAIL / NOT_USABLE. COVER93@4096 TRAIN4 = 2/4, below the registered 3/4 gate. STOP_FOR_NEXT_TARGETED_TASK_DECISION.**

Authority: [Issue #11 canonical 6027913340](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6027913340), release `83af9099f3bab9a4ac08eaa85380c2cfd6351f34`. Existing branch `pi0267/executor-decoupled-recovery`. `selected_checkpoint=NONE`, `deployment=NONE`, `sprint_candidate=NONE`.

## Actual object-task results

All twelve new episodes completed the original 32 seconds, 640 controls and 3200 physics steps. Success requires on-top AND static AND not-grasped at all 41 control samples from 28 through 30 seconds, plus interface-valid full execution. A first success or visually stacked cubes do not satisfy that gate alone.

| Point | TRAIN persistent success | Sustained lift | On-top observed | Admission |
| --- | ---: | ---: | ---: | --- |
| FOCUS4096 parent, reused | 3/4 | 4/4 | 4/4 | Historical DEV3/8; NOT_USABLE retained |
| COVER93@512 | 2/4 | 4/4 | 4/4 | Valid failure |
| COVER93@1024 | UNMEASURED | UNMEASURED | UNMEASURED | Explicit superseded-backlog skip; no result removed |
| COVER93@2048 | 0/4 | 3/4 | 3/4 | Valid failure |
| COVER93@4096 | 2/4 | 4/4 | 4/4 | Valid failure; terminal |

No new checkpoint met TRAIN admission. Therefore DEV was not opened, no candidate was frozen, and CONFIRM30032..30047 and locked30048..30063 were not opened. The final4096 screen was completed. No failed condition was replaced, no teacher rerun was used, and no additional training followed.

| Point | TRAIN seed | First success at control sample (s) | Persistent gate | IK holds |
| --- | ---: | ---: | --- | ---: |
| 512 | 10000 | 15.35 | FAIL | 1 |
| 512 | 10001 | 6.15 | PASS | 0 |
| 512 | 10002 | 5.20 | PASS | 0 |
| 512 | 10003 | 6.25 | FAIL | 0 |
| 2048 | 10000 | 15.15 | FAIL | 70 |
| 2048 | 10001 | 8.85 | FAIL | 0 |
| 2048 | 10002 | 8.00 | FAIL | 0 |
| 2048 | 10003 | NONE | FAIL | 422 |
| 4096 | 10000 | 7.35 | FAIL | 0 |
| 4096 | 10001 | 5.80 | FAIL | 0 |
| 4096 | 10002 | 5.90 | PASS | 0 |
| 4096 | 10003 | 4.65 | PASS | 0 |

The saved native100Hz event intervals are separately retained. Their first transition can precede the 20Hz control-sample first-success value above.

### The final failures are post-placement stability failures

| Final TRAIN seed | On-top samples /41 | Static /41 | Not-grasped /41 | Registered verdict |
| --- | ---: | ---: | ---: | --- |
| 10000 | 41 | 12 | 25 | FAIL: static and not-grasped |
| 10001 | 41 | 25 | 41 | FAIL: static |
| 10002 | 41 | 41 | 41 | PASS |
| 10003 | 41 | 41 | 41 | PASS |

All four final episodes lifted, reached on-top, released after grasp, and had an early native success. They also had zero IK holds and no envelope termination. Final commanded-to-installed XYZ median errors were approximately 0.04–0.06mm; installed-to-achieved control-end median errors were approximately 1mm, with the recorded physical-response interval included. These observations distinguish reference installation, tracking and object completion; they do not establish a unique contact, stiffness or model cause.

10003 has static flicker after30s. Its registered28..30s gate still passes; full32s interface-valid execution was completed. No retroactive success definition was introduced. 2048 failures were different: all four lacked on-top in the qualification window, and two failed despite zero IK holds. Do not collapse all milestones into one mechanism.

## Actual learning, coverage and integrity

Job `651626.hopper-m-02` ended F / exit0 at 2026-10-07 11:12:09 SGT. Completed: 4096 formal updates, 8192 formal microbatches, plus16 actual qualification updates. The continuous8 versus4-save-fresh-process-restore4 comparison was exact for model, optimizer, scheduler/RNG, consumed inputs, losses, outputs and gradients.

All8192 formal inputs/labels match the frozen plan and proofs. Actual observed epsilon/tau/S4 identities match; raw-noise regeneration was not performed. Training uses FP32 features/query/Euler with native BF16 epsilon; evaluation uses the original BF16 query and FP32 physical decode. Four full differentiable S4 steps, mode restoration8192/8192, generated-gripper and box terms from the first microbatch, and471 trainable /559 frozen scope are audited. All559 final frozen tensor payloads are byte-identical to the parent.

The completed consumption audit verifies all93 admitted TRAIN episodes ×4 groups: first and last eligible anchors and all nonempty eight-way temporal strata. Each group has2048 visits; every focused episode has64 visits/group, and each background episode13/14. Prefetch reservations are not reported as consumed updates. This is the authorized50/50 retention-weighted plan, not uniform episode sampling.

The saved20..30s expert hold observations have median200 duplicate StateV1 rows among201 and zero median adjacent RGB change. Added time coverage does not establish coverage of learner-induced post-placement states.

The unchanged48-row source panel gives the following per-group median full-H40 XYZ RMS, in millimetres. It is TRAIN fitting evidence, not independent evaluation or a controller tolerance.

| Group | Parent | 512 | 2048 | 4096 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 38.51 | 41.20 | 45.44 | 36.84 |
| 1 | 8.61 | 17.23 | 17.15 | 8.97 |
| 2 | 13.85 | 22.64 | 23.07 | 17.14 |
| 3 | 4.65 | 8.45 | 13.52 | 3.62 |

Final release/hold source fitting improved on this measure; physical persistent success did not reach admission. Additional optimization and changed visits are not separated by a matched control. Neither universal GR00T failure nor a sole missing-data cause follows from these small developmental panels.

Independent CPU replay passed for all48 parent diagnostic outputs and1104 new outputs (three48-row diagnostics plus960 closed-loop queries). It recomputes saved four-step Euler, original decoder, fresh same-tick RGB/state, actual noise and native success with zero new neural/physics calls. Sixteen exact parent outputs were reused; parent fresh queries were32. Early immutable FAST receipts retain their pre-full-replay `PENDING` metadata; the later complete `CPU_REPLAY_*` receipts are the full-replay authority.

## Actual budgets and performance

| New PI0278 ledger | Actual | Cap |
| --- | ---: | ---: |
| Formal updates | 4096 | 4096 |
| All completed updates | 4112 | 4176 |
| Training heads, including actual canary/parity/recompute | 74046 | 76032 |
| HOP CPU input attempts, including prefetch/cold encoding/readback | 13144 | 65536 |
| Fresh quality requests | 1136 | 4608 |
| Quality heads | 4544 | 18432 |
| Physical starts | 12 | 52 |
| Physics steps | 38400 | 170000 |
| Controls | 7680 | 34000 |
| RGB pairs | 7680 | 34500 |
| Technical episode replacements | 0 | 4 |
| New teacher collection / WBC / hardware | 0 / 0 / 0 | 0 / 0 / 0 |

HOP CPU items comprise8236 FAST_CACHE2 accesses,2454 cold reference encodings and2454 cache readbacks. The separate Core inference/replay workflows have their actual query/output counts above; the HOP item ledger is not presented as a whole-machine CPU telemetry measure.

The sealed PI0277 ledger remains8192 formal /8232 total updates,148263 training heads,2680 quality requests /10720 quality heads and36 physical starts /97320 physics steps. Combined healthy formal updates are12288, all completed updates12344 and training heads222309. No old budget was reset.

Qualified FAST_CACHE2 with ordered CPU workers/pinned memory was used. Actual median update1.685s, loader wait0.000142s, peak allocated GPU memory34,377,103,872 bytes. These are measured timing/allocation values, not SM utilization or a controlled whole-GPU speedup estimate. More workers were not swept when measured data wait was negligible.

Final complete-Trainer save took12.415s. Final replacement payload was5,676,315,136 bytes over the existing parent, enumerating all1030 tensor keys with471 changed; all reconstructed full files matched the immutable9.53GB source snapshot. Final transfer invocation1050.244s includes packaging/transport/readback, reconstruction23.883s. Model load2.619s, final FIT/TRAIN panel371.578s, CPU replay14.404s; median saved final S4 query0.463s. Simulation pauses for inference: NON_REALTIME.

Engineering receipts preserve the inherited512 READY creator-release reconciliation, true interrupted2048 relay prefix-resume, and video receipt path repair. The latter reused the already-created clips; it did not rerun physics or training. Exact-source original hashes and first-valid outputs remain intact.

## Local delivery, recovery and stop

Complete local Trainers3072/4096 and the complete parent remain on Hopper. Core holds model-only views and exact replacement transport payloads, not optimizer-resume packages. Config directories are not labelled resumable. Terminal resume beyond the budget was not run; the actual short canary qualifies the recovery mechanism. Final snapshots retain their own source-file SHA256 inventories.

Current measured Hopper quota headroom is about198.60GiB (admin report can lag); Core NVMe free space about168.51GiB. Shared `df` is not personal quota. No further cleanup was executed during terminal evaluation. HF bulk/readback and old CONTROL remain **DEFERRED_USER_PRIORITY**, explicitly allowed by this GO; original local recovery/evidence has not been retired.

All12 actual RGB clips and quantitative tables are on Core under `/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2/delivery`. Each clip is source-Panda/NON_REALTIME and retains its real PASS/FAIL. Saved-frame512/2048/4096 contact sheets have visual QA; judgments come from native arrays. No generated replacement image or physics render was used.

Final local inference package: `/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2/handoff/COVER934096/POLICY_HANDOFF.json`, status **NOT_USABLE**. `RUN_INFERENCE.sh` points to the actual model loader/inference entry; its accompanying input example deliberately reuses one existing first-valid FIT output and makes zero new queries. It is a cached example, not another fresh performance result. The measured evaluation itself used fresh model predictions from fresh observations with original gate/controller installation.

**One recommended next decision, NOT_AUTHORIZED:** a bounded matched investigation of learner-state feedback coverage for post-placement stable holding. The final failures specifically concern late static/not-grasped predicates despite all41 on-top samples; expert late-time exposure is actually complete but often constant. This supports testing that gap, not claiming it as proven cause. Freeze one intervention/control and budget before a new GO. No new loss, controller threshold, seed, scene, task data or training run is launched here.

Stop at the registered finite hypothesis endpoint. Execution completion and quality failure are separate. Formal selected/deployment remain NONE.

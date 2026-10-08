# PI0285 R3 terminal report

## Verdict

**Execution closed; valid confirmation quality FAIL.** The original single DOSE2048 phase actually completed and its first development candidate was frozen. Unique CONFIRM_A completed all32 original conditions and independently recomputed at20/32, below26/32. The source evaluator closed with `STOP_CONFIRMATION_VALID_QUALITY_FAIL_NO_SECOND_CANDIDATE`. No second candidate, shared training, new neural probe or physical trial follows this result.

| Model point | TRAIN | DEV | Unique CONFIRM_A | Status |
|---|---:|---:|---:|---|
| PLACE1024 | 0/8 | Not admitted | Not admitted | Valid quality FAIL |
| PLACE2048 | 2/8 | Not admitted | Not admitted | Valid quality FAIL |
| PLACE4096 | 4/8 | Not admitted | Not admitted | Valid quality FAIL |
| DOSE2048 / REPAIR_A2048-2048 | 7/8 | 13/16 | **20/32** | Development gates passed; final qualification FAIL |

Thresholds remained7/8,13/16,26/32. Independent source and CPU panel hashes, frozen candidate and final route are in `WP2/REPAIR_A2048-2048` and `WP8/REPAIR_A2048-2048`. The final versioned delivery bundle is `WP8/terminal`; earlier READY/pending snapshots remain historical. `selected/deployment=NONE`; source PANDA/native PD, NON_REALTIME, is not target WBC or hardware qualification.

## Actual learning and integrity

PLACE completed4096 formal updates; the sole admitted DOSE phase completed2048. The non-null-parent continuous8 versus4+fresh4 proof passed actual model/Adam/scheduler/RNG/input/noise/generated-output parity. Both phases kept native loss, row10 EEF/gripper interface, shared DiT scope, frozen VLM/other31 rows, processor/statistics, K4 and precision. Final full Trainer audits passed; lower saved loss or DIAG error did not substitute for object-task qualification.

Job655273 completed its original0->2048 segment, then exited1 on report finalization; readonly reconciliation retained the actual completed updates. Exact resume655301 finished4096 with exit0. DOSE655670 finished2048 with exit0. No training was repeated for reporting or HF issues. Original655259's one attempted four-chunk microbatch remains raw accounting.

## Confirmation failures

All32 actual confirmation conditions completed and were CPU scored. The12 failures are285302,285303,285304,285305,285306,285312,285321,285322,285323,285325,285327,285329. All12 have zero task hold samples and no first full task hold; none is a gate REJECT.

- Six lack at least one required grasp/lift/transfer history condition.
- Six have all three history conditions but fail the later complete task/hold requirement.
- Seven failing episodes record native IK holds; five record none. This association does not prove a unique learning or controller cause.

The saved funnel separates reference installation, actual tracking, object events and persistent task completion. Original late failures remain. There is no inferred successful transfer to Go2, altered WBC, new representation or claim that failure is solely insufficient data.

## Storage engineering deviation

Confirmation285316 completed80 actual queries but saving its observations hit SSD ENOSPC. Its incomplete arrays, computed score and all80 queries remain preserved/charged. Sixteen already valid results were reused. One exact same-condition engineering replacement was admitted inside the original32-replacement allowance.

Independent CPU comparison verifies all80 scientific input/output payloads and gates, initial state, full physics arrays and computed score are identical across the two attempts. Measured wall latency and NPZ container timestamps are separately recorded. This was not a quality retry or noise/seed change.

A closed PLACE1024 inference view was migrated with full-byte readback and original path preserved. Only the exact two obsolete PLACE2048 duplicate BF16 weight shards were deleted after pinned full-Trainer HF readback and link/consumer checks:6,288,169,448logical /6,288,183,296allocated bytes. Its local view is explicitly metadata-only and not loadable. Raw evidence, statistics, latest full Trainer and other projects were preserved. No broader cleanup was performed.

## Budget actually observed

| Item | Actual | Registered cap |
|---|---:|---:|
| Formal updates | 6144 | 12288, with A limited to6144 |
| Discarded canary updates | 32 | 64 |
| All completed updates | 6176 | 12512 |
| Quality requests, both hosts | 6901 | 32768 |
| K4 heads, both hosts | 27600 | 131072 |
| Physical starts | 257 | 840 |
| Technical physical replacements | 1 | 32 |
| Control calls / RGB pairs | 169620 each | 580000 each |
| Physics integrations | 848100 | 2900000 |
| Dispatches | 339240 | 1160000 |
| Teacher controls | 112640 | 400000 |

The257 starts comprise176 original teacher episodes,80 unique model conditions and one engineering replacement. One logged quality request did not enter K4 heads; it is not erased to force a requests-times-four equality. Unused shared/canary/query allowances are not permission to continue after confirmation FAIL. Fine-grained historical GPU utilization and exhaustive aggregate CPU core-hours are not inferred from scheduler walltime.

## Eight bounded workpackages

| Package | Actual terminal state |
|---|---|
| WP1 native training/input/restore | COMPLETE: admitted data reused, actual canaries and6144 formal updates, CPU inventories |
| WP2 snapshot/physics/CPU feedback | COMPLETE: four eligible points, original task panels and unique confirmation |
| WP3 temporal/source attribution | Saved-array reports delivered; no extra neural/physics calls |
| WP4 bounded response probes | Earliest1024:32 calls/128heads and CPU report; final/DROID/shared probes NOT_EXECUTED; no probes after failed confirmation |
| WP5 alternatives/history coverage | CPU-only draft delivered; no new adapter/history experiment authorized |
| WP6 sole branch/recovery | Original DOSE2048 executed; correction collection NOT_TRIGGERED, no second branch |
| WP7 shared two-task stage | Schema/paired-stream READY only; T2 acquisition and SHARED NOT_LAUNCHED because T1 confirmation failed |
| WP8 delivery | Complete load identities, CPU qualification, full-Trainer HF readbacks and actual videos; no qualified/protected T1 receipt fabricated |

## Recoverable research models

PUBLIC repo: `leoyuer/pi0246-gate-pass-campaign`. Each listed complete Trainer passed fixed-revision14-file readback.

| Point | Prefix | Fixed revision |
|---|---|---|
| PLACE1024 | arms/PI0285/PLACE4096/checkpoint-1024 | 0810fb97bd3fa8624ecb7494f5fa01aae55accbc |
| PLACE2048 | arms/PI0285/PLACE4096/checkpoint-2048 | 43ee74595d710c3f581ef61c19104620b0bdfdf0 |
| PLACE4096 | arms/PI0285/PLACE4096/checkpoint-4096 | 086709ebbc6b2fb38b1c11e8b64b8094f2efd9c3 |
| DOSE2048 | arms/PI0285/REPAIR_A2048/checkpoint-2048 | 75432eb91a2c602930a9e1e979aaebbeaebff02b |

DOSE full Trainer is20,468,184,976bytes, tree `fb17b65effc3ae8dcbefa784adc8f7e6fab916821d29e4b2539e5be685ed8469`. Hopper retains complete recovery; Corelab retains only its immutable native BF16 inference view. Full Trainers were not copied to Corelab or the PC. These are recoverable research artifacts with a valid negative quality verdict, not qualified deployment policies.

Actual saved-frame videos include fixed TRAIN success285100/failure285106, and unique-confirmation first success285300/first failure285302. All show both real source views, full task time and explicit actual-model/NON_REALTIME/not-target-WBC labels. Video packaging creates zero new render, physics or model calls. Final compact evidence archives are in PUBLIC `evidence/PI0285/common-scene-first-policy-20261008/terminal-R3/{core,hop}`. Their upload and full-file readback remain in progress as of06:30SGT; no PASS is inferred from upload progress. Full RGB remains on Corelab, and complete Trainers have separate completed pinned receipts. Final archive completion receipts will be added without new scientific calls.

Scientific execution stops here. Mail/monitor/merge, target WBC/physics, hardware and protected TEST/DEV-F remain untouched.

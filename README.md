# Go2 + ARX5 Hierarchical Whole-Body VLA

This repository is the source of truth for the Go2 + ARX5 hierarchical VLA
contracts, adapters, experiment runners, tests, and provenance documents.
Large or private model, WBC, dataset, simulator, and hardware inputs remain
outside Git and are referenced through manifests and reports.

## Current execution priority — PI0285 R3, source manipulation policy delivery

Read the [fixed R3 protocol](docs/protocols/PI0285_R3_PARALLEL_WORKPACKAGES_AND_POLICY_DELIVERY_V1.md), [authority #6066233135](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6066233135), and [actual execution receipts](reports/PI0285/r3/PLACE4096_COMPLETE_AND_SOLE_DOSE_CONTINUATION.md). Release `d6bdb8678e28ca5331d335b2ccd4d5753aef17d6` continues the original R2 recipe and cumulative budgets on `pi0267/executor-decoupled-recovery`.

PLACE4096 training and fixed evaluation are complete; TRAIN1024/2048/4096 success is0/8,2/8,4/8, all below7/8. The original sole DOSE2048 branch completed2048 updates, actual integrity and full14-file HF readback. Fixed TRAIN7/8 and DEV13/16 passed independent CPU scoring; the first development candidate is frozen and its unique CONFIRM_A32 is in progress. After an ENOSPC save failure,16 complete results are reused and only the same-condition engineering-incomplete attempt is replaced within the original allowance. These are source PANDA/native-PD results, not target WBC or hardware qualification. Confirmation and independent full-Trainer protection precede shared-task training; `selected/deployment=NONE`.

**2026-10-09 Asia/Singapore stops:** optional research05:30; new recovery collection06:00; new optional/shared optimization06:30; optimizer08:30; neural/physics09:45; handoff10:00. No date rollover, new search, recurring monitor or merge. Actual task feedback and recovery integrity precede optional analysis and bulk HF archival.

## Historical execution priority — PI0282 R2, deadline-gated policy delivery

**Hard deadline: 2026-10-08 09:00 Asia/Singapore.** Read the [R2 gated pretrained-policy sprint](docs/protocols/PI0282_R2_0900_GATED_PRETRAINED_POLICY_SPRINT_V1.md), [cumulative limits](reports/PI0282/design/R2_DEADLINE_AUTHORITY_AND_LIMITS.json), and newest PI0282 R2 canonical in [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11). This is a continuation of the current owner goal, not a competing run. Adopt actual completed R1 data/canaries/training and preserve counters. Publication is not a remote start receipt.

Prioritize genuine-pretrained ACTION native-loss adaptation, immutable local snapshots and real source task feedback. Optional INTERFACE comparison and HF bulk are not prerequisites for delivery. The protocol permits bounded same-data continuation, concentrated fit, and one validated recovery-supervision branch only when their respective gates hold. WBC, target simulation/model calls and hardware remain zero; VLM and all undeclared parameters remain frozen. No old task-weight lineage, custom losses, fabricated observations, relaxed task thresholds or scripted online finish.

Formal PI0282 R1+R2 updates are capped at8192 cumulatively, not8192 additional. Stop new updates no later than07:30 and earlier if measured evaluation time requires; no new recovery collection after06:00. Freeze one development-qualified candidate for a single CONFIRM16; all neural/physical work ends by08:50 and handoff by09:00. No automatic date rollover. If no qualified policy is obtained, deliver the best explicitly unconfirmed result and actual failure, not a claimed PASS. `selected/deployment=NONE`; source prototype qualification is not Go2/WBC deployment.

## Parent protocol — PI0282 R1 (R2 supersedes conflicting execution rules)

The [PI0282 pristine native-loss EEF interface protocol](docs/protocols/PI0282_PRISTINE_NATIVE_EEF_INTERFACE_ADAPTATION_V1.md) and [original review/limits](reports/PI0282/design/REVIEW_AND_LIMITS.json) preserve the initial design. R2 changes priority, bounded continuations, intermediate selection, deadline and cumulative caps; untouched semantic requirements remain.

PI0280/PI0281 native baseline work is sealed: original model calls reproduced; direct EEF, joint closed loop and commanded-FK replay are0/4 on the developed source task; expert bridge4/4. The delivered WBC progress repair prevents the immediate return abort but orientation qualification still fails. The original model has no honest target EEF-only input route. These source/control findings are not a completed GR00T+Go2 high-level closed loop.

The parent comparison used INTERFACE2048 and ACTION2048 from the same pristine GR00T-N1.7-3B and newly declared EEF10 interface, native flow-matching only. Compare row10 state/action encoders+decoder with that same scope plus shared DiT; freeze VLM and protect other rows. The placeholder is not a pretrained zero-shot target route. Under R2, practical source-policy delivery precedes completion of this matched comparison; omitted or shortened arms are explicitly reported.

PI0279 CLEAR4096 and all old automatic fine-tuning/recovery-data continuations remain `PAUSED_BY_USER_PRETRAINED_BASELINE_PRIORITY`; their pause and recoveries are preserved. R2's permission to reuse named validated recovery examples is not permission to restart those old drivers. New target WBC/physics/model calls and hardware remain0. Source data admission remains independent of target-WBC success. H200 training and Corelab evaluation use actual legal allocations, local checkpoints and verified caching; HF bulk stays deferred. `selected/deployment=NONE`; source-policy qualification is separate from target deployment.

Historical priorities: [PI0280 no-training baseline](docs/protocols/PI0280_GENUINE_PRETRAINED_GROOT_FROZEN_WBC_BASELINE_NO_TRAIN_V1.md), [PI0281 readout/WBC reference review](docs/protocols/PI0281_CORELAB_NATIVE_READOUT_AND_WBC_REFERENCE_REPAIR_NO_TRAIN_V1.md), and [original pause decision #6037170888](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6037170888). Their zero-update execution records remain unchanged.

## Current research direction — read first (2026-10-05)

**后续GR00T方案的统一前提：** [任务空间物体操作研究章程 V1](docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md)
(`GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1`).

> 我们让VLA只学习task-space manipulation behavior，而不学习目标机器人的joint action；低层WBC独立学习广泛的task-space tracking。因此任务数据可以不依赖目标机器人遥操作，而同一冻结WBC能承接多个新任务。

**从现在起，真正的物体操作任务数据、GR00T学习与闭环执行是主线；简单轨迹生成/跟随降为基础设施、执行侧qualification与有界诊断。** 上层团队继续冻结师兄提供的WBC，不将低层训练计为上层新增成果。上述claim是设计前提与待验证目标，不是既成实验结论。

每份后续protocol须引用章程并明确物体任务、独立任务数据来源、EEF/夹爪接口、固定WBC身份、对照、任务成功标准与预算。不要把降低轨迹RMSE当作物体任务成功，不要按目标WBC执行成功筛选高层任务数据。

The current working branch is `pi0267/executor-decoupled-recovery`; use the latest explicit execution authority and actual receipts in [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11). This direction does not restart or change PI0274, authorize new runs, or reset budgets. `selected/deployment=NONE` remains. The accepted-baseline and Stage45A references below are historical navigation, not the current experiment authorization.

Related preparation: [sim-first object-task roadmap](docs/coordination/2026-10-05_SIM_FIRST_OBJECT_MANIPULATION_ROADMAP.md) and [GR00T/frozen-WBC meeting handoff](docs/coordination/2026-10-07_GROOT_WBC_MEETING_PREPARATION.md).

## Accepted repository baseline

The current accepted research tip is the normal-merge tip on `main`:

```text
branch: main
commit: bfa14bf1b74eae908c0922575a4db671c903b9b8
parents: 237be6fc37c20cf44ccccf5f0cfbd3cce629760f + 9852090f66759c8ddc7ce974b31e57350ab536c1
accepted source: stage44r/corelab-attribution-repair @ bd0f02934e680789cf38adacb838e1ea555c707d
diagnosis: STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED
```

The normal consolidation merge preserves the forward-only Stage19--Stage44R
history and adds navigation and audit material; it does not rewrite, squash,
or delete scientific history. Stage45A now has a CPU/read-only design branch
and protocol ([protocol](docs/stages/STAGE45A_FULL24_ADAPTATION_PROTOCOL.md),
[manifest](manifests/STAGE45A_FULL24_ADAPTATION_DESIGN_V1.json)); it does not
authorize model construction, training, data generation, H200 fine-tuning,
Vanda A40 execution, or new model/WBC runs.

## Current architecture

```text
Vision + Language + Proprioception
        -> GR00T N1.7
        -> StructuredActionV1 / EEF trajectory
        -> SafetyGate
        -> reference-only receding-horizon handoff
        -> general trajectory-tracking WBC
        -> Go2 + ARX5
```

The external model/controller waist is EEF-only: XYZ + SO(3) trajectory with
an independent parallel-gripper channel. The delivered WBC performs internal
whole-body redundancy resolution; its structured body/gait signal is not a
GR00T label. Stage40--Stage43 closed-loop records and Stage44R attribution are
engineering and diagnostic evidence, not claims of task success, general
robustness, real-time operation, or hardware readiness.

## Navigate the repository

- [Research timeline](docs/RESEARCH_TIMELINE.md) — Stage19 through Stage44R,
  with canonical branches, narrow outcomes, and evidence pointers.
- [Accepted baselines](docs/ACCEPTED_BASELINES.md) — what each accepted gate
  establishes and what remains explicitly unproven.
- [Repository guide](docs/REPOSITORY_GUIDE.md) — source, runner, test, report,
  manifest, and HPC directory conventions.
- [Consolidation audit](reports/REPOSITORY_CONSOLIDATION_AUDIT.md) — the
  read-only RC0 inventory and provenance findings.
- [Historical archival recommendation](docs/HISTORICAL_ARCHIVAL_RECOMMENDATION.md)
  — RC8 PR-closure record and branch-preservation recommendation.
- [Stage45A FULL24 adaptation protocol](docs/stages/STAGE45A_FULL24_ADAPTATION_PROTOCOL.md)
  — design-only, pre-registered data-expansion gate; no training is authorized.

## CPU validation

The maintained contract suite is CPU-only:

```bash
python -m pytest -q
```

When the optional test dependency is unavailable, the standard-library tests
and the static checks described in [the repository guide](docs/REPOSITORY_GUIDE.md)
remain the required fallback. No command in this consolidation performs a
model forward, simulator/WBC rollout, training, data generation, or hardware
action.

## Provenance and private inputs

Historical stage runners and reports stay at their original paths because
Issue #11 evidence and sealed manifests refer to them. Node-local caches and
evidence roots (for example `/var/cache/...` or `/scratch/...`) are references,
not repository contents. Do not commit credentials, checkpoints, videos,
datasets, USD assets, or private WBC sources.

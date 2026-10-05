# Go2 + ARX5 Hierarchical Whole-Body VLA

This repository is the source of truth for the Go2 + ARX5 hierarchical VLA
contracts, adapters, experiment runners, tests, and provenance documents.
Large or private model, WBC, dataset, simulator, and hardware inputs remain
outside Git and are referenced through manifests and reports.

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

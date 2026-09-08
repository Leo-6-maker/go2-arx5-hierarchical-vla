# Go2 + ARX5 Hierarchical Whole-Body VLA

This repository is the source of truth for the Go2 + ARX5 hierarchical VLA
contracts, adapters, experiment runners, tests, and provenance documents.
Large or private model, WBC, dataset, simulator, and hardware inputs remain
outside Git and are referenced through manifests and reports.

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

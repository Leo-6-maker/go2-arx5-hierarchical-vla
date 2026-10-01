# Go2 + ARX5 Hierarchical EEF / VLA Research

> **Start here.** Repository navigation was consolidated on 2026-10-01. Historical
> experiments are retained, not reclassified as passing. `main` contains the
> accepted baseline plus current navigation; the latest experimental runtime is
> **not** silently merged into that baseline.

## Current entry points

| Need | Canonical location |
|---|---|
| Current scientific status and claim limits | [Project status](docs/PROJECT_STATUS.md) |
| Scientific authorization and experiment returns | [Issue #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11) |
| Current experimental implementation | [`pi0266/state-conditioned-feedback`](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/pi0266/state-conditioned-feedback) |
| Protocols and recent result pointers | [Research index](docs/RESEARCH_INDEX.md) |
| Branches, PRs, archival and source recovery | [Repository workflow](docs/REPOSITORY_WORKFLOW.md) |
| Actual consolidation inventory and receipt | [2026-10-01 consolidation](docs/maintenance/2026-10-01/RESULT.md) |
| Tested reusable-code integration still to do | [Issue #47](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/47) |

**Do not use historical PR #77 as the current implementation or as a mandatory
pointer destination.** It is an archived Stage45H provenance discussion; its
original code and comments remain accessible.

## Research objective and implementation boundary

```text
Visual observations + language + task-space feedback
             -> GR00T / high-level EEF trajectory policy
             -> explicit decode, frame/TCP conversion and SafetyGate
             -> reference-only handoff
             -> embodiment-specific execution / WBC
             -> actual task feedback
```

The research objective is to make task data and high-level policy weights
reusable across execution backends, rather than assume arbitrary EEF references
are executable. Current work on Go2 + ARX5 does **not** yet establish general
cross-embodiment transfer. Knowing actual TCP state is different from conditioning
on a particular robot's joints or model ID.

The established external interface is task-space XYZ + SO(3), with an independent
gripper channel. Robot/body/gait realization belongs to the execution layer.
A usable policy identity includes weights, processor, statistics, action/state
mapping and decoder. Read the applicable immutable protocol: the historical
StructuredAction and newer training lineages must not be mixed by name alone.

Repository administration does not authorize a new experiment, change a gate,
open protected data, or grant deployment qualification. Historical FAIL/PARTIAL
and `selected_checkpoint=NONE` / `deployment_selected_checkpoint=NONE` remain.

## Code baseline versus research evidence

Before this housekeeping merge, `main` was at
`607c7b734ba88ee0215a432aa8d2f576da9e77b0`, following the accepted Stage44R
consolidation. This update does not bulk-merge experimental implementations or
rewrite their commit ancestry. Use a protocol's pinned execution SHA for a run,
not the current `main` tip or an old PR number.

The earlier documentation remains available as historical context:
[research timeline](docs/RESEARCH_TIMELINE.md),
[accepted baselines](docs/ACCEPTED_BASELINES.md),
[repository guide](docs/REPOSITORY_GUIDE.md), and
[original README at the pre-cleanup commit](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/607c7b734ba88ee0215a432aa8d2f576da9e77b0/README.md).
They must not be read as the latest PI0266 status.

## Tests and artifacts

The accepted baseline's CPU contract suite is run with `python -m pytest -q`.
Administrative safety tests are independent:
`python scripts/maintenance/test_consolidate_repo.py`.
A passing CPU test is not a model or controller quality PASS.

Large model checkpoints, raw datasets and videos stay outside Git and are bound
by manifests and fixed archive revisions. Private WBC, simulator/vendor assets,
credentials and machine-specific caches must not be published as public evidence.
Existing remote training jobs, local worktrees and artifact stores are not changed
by this repository cleanup.

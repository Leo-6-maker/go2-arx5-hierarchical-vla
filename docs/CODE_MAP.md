# Code map · current route and retained history

Updated 2026-10-28. `main` is the integration entry. Repository presence does not imply native qualification. Reuse existing implementations; do not rename historical run paths just to make the tree smaller.

## Current reusable layers

| Location | Responsibility | Boundary |
|---|---|---|
| `src/go2arx5_vla/contracts/` and `trajectory/` | Task-space trajectory, frame, TCP, time and preview semantics | Scientific numerical behavior is versioned; no silent substitutions |
| `src/go2arx5_vla/adapters/` | GR00T-to-controller reference conversion and execution-progress interfaces | An offline adapter is not a real WBC task PASS |
| `src/go2arx5_vla/target_mujoco/` | Current frozen target MuJoCo/WBC reference and recording interfaces | Do not confuse with a different historical executor ABI |
| `src/go2arx5_vla/workspace/`, `scripts/workspace/`, `infrastructure/` | Existing profiles, source/manifest checks, host-aware paths | No new scheduler or private environment copy |
| `scripts/pi0292/train_adapter.py` | Reused official GR00T modality/Processor and actual submitted-command dataset bridge | Keep R0 and existing label semantics for current arms |
| `scripts/pi0294/generalization/` | GEN1 scene/data preparation and minimal-adaptation model/dataset/training/resume | CPU fixtures are not full GR00T qualification |
| `scripts/pi0294/pv1/` | Source data, trained candidate delivery, C/D and original panel evaluation/runtime | CoreLab 4080S owns closed-loop evaluation |
| `scripts/pi0294/dec1/` | Original native training entry and frozen-executor comparison support | Preserve exact scopes; do not mix target training into DEC1 |
| `scripts/pi0294/replay_viewer/` | 官方Rerun薄适配器：PC离线读取原Teacher／Eval记录与双RGB、状态、参考、审计 | 三条真实稀疏RRD已核；不推进物理／策略／重评分，旧HTML仅历史兼容 |
| `tests/`, `scripts/pi0292/test_*.py`, `scripts/pi0294/test_*.py` | CPU contracts and implementation regressions | Model/physics execution needs its own valid runtime scope |
| `scripts/maintenance/` | Bounded repository administration | No scientific imports, remote job mutations or data cleanup |

## Document hierarchy

`START_HERE.md` is the short entry, `CURRENT_SCIENCE_STATUS.md` is a dated evidence snapshot, and `docs/EXPERIMENT_ROADMAP.md` is the consolidated research route. Detailed protocols remain under `docs/protocols/`; execution handoffs under `docs/handover/`; actual sanitized receipts under `reports/`.

Do not prepend every full runtime log to the root entry. Publish a dated receipt and link it from current status. Keep fixed-SHA citations for historical claims. When evidence contradicts an older snapshot, update the current summary, not the original receipt.

## Retained but not default execution routes

Old Vanda cohost Eval, 5080 migration, Hopper runs, stageNN tools, and former plan versions remain because historical capsules/tests refer to them. They are not active alternatives to Vanda-training/CoreLab-Eval. Their presence does not authorize dispatch. No runtime Python module is deleted or moved in this consolidation.

Legacy exact branches are integrated through normal merge commits. The October28 maintenance manifest records which heads may be retired; same-name tags are created only by the explicit guarded retirement command, not assumed to exist. The active former pi0267 ref remains during publisher transition.

## Validation

Existing CPU workflow includes root, PI0292 and PI0294 checks; its richer experimental version is restored unchanged. This consolidation did not execute those suites and did not obtain a new CI run. Run existing CPU tests in an appropriate management/CI environment before deploying a new source capsule. Never use a source merge as a model, simulator, WBC or production release certificate.

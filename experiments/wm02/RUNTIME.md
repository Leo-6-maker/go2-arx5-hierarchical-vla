# WM02 implemented P0/P1 commands

The pinned handoff is `ddc801f30e288365925c61c37f7ee445ddc29712` (unmerged PR344). The concurrent local-first resource addendum through `19f686c9c006bffce22d3819b2019e1e45ebbec2` was read and merged normally before publication; it updates execution order without changing the original scientific design or the already-pinned execution sources. Actual implementation and effects are in [the first return](../../reports/WM02/wm02-20261104-p0-p1/FINAL_REPORT.md), with [partial local readiness](../../reports/WM02/local/LOCAL_4080S_READINESS.md).

Implemented, import/help without simulator/model/network effects:

```bash
python -m scripts.wm02.resolve --help
python -m scripts.wm02.qualify --help
python -m scripts.wm02.profile_source --help
```

`resolve` records supplied actual owner observations; it never creates an execution grant. `qualify --domain S` uses the reused native driver/S1, full serialized integration/controller restore, real consumed commands and two branches from each of eight fixed diagnostic states. `qualify --domain T` currently returns78 with TARGET_BLOCKED and no target construction, including when a supplied metadata flag claims permission. Full MPC/WBC state export and native target qualification are not implemented.

`profile_source` uses actual1/4/16 process levels only within the bound CPU/RAM allowance, measuring spawn/construction/recording overhead. It does not change dt/solver/contact/safety or qualify target WBC. Both commands require an independent immutable capsule, existing host-aware registry/lease and owner lock. Effects use the existing Journal/onceIO; no scheduler/allocator/service is added.

The world-model directory explicitly reuses six WM01 source adapter modules at pinned file SHA, with new WM02 action-time/state-role/recorder wrappers. No WM01 trainer/checkpoint, third-party OSC implementation or private WBC asset is copied. Future recurrent fitting/PPO/imagined Actor/visual CLIs are still unimplemented; no empty executable stubs are supplied.

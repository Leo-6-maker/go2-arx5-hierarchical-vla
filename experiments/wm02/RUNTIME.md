# WM02 implemented runtime entry points

The pinned handoff is `ddc801f30e288365925c61c37f7ee445ddc29712` (unmerged PR344). The concurrent local-first resource addendum through `19f686c9c006bffce22d3819b2019e1e45ebbec2` was read and merged normally before publication; it updates execution order without changing the original scientific design or the already-pinned execution sources. Actual implementation and effects are in [the first return](../../reports/WM02/wm02-20261104-p0-p1/FINAL_REPORT.md), with [partial local readiness](../../reports/WM02/local/LOCAL_4080S_READINESS.md).

Implemented, import/help without simulator/model/network effects:

```bash
python -m scripts.wm02.resolve --help
python -m scripts.wm02.qualify --help
python -m scripts.wm02.profile_source --help
python -m scripts.wm02.env_preflight --help
python -m scripts.wm02.neural_smoke --help
```

`resolve` records supplied actual owner observations; it never creates an execution grant. `qualify --domain S` uses the reused native driver/S1, full serialized integration/controller restore, real consumed commands and two branches from each of eight fixed diagnostic states. `qualify --domain T` currently returns78 with TARGET_BLOCKED and no target construction, including when a supplied metadata flag claims permission. Full MPC/WBC state export and native target qualification are not implemented.

`profile_source` uses actual1/4/16 process levels only within the bound CPU/RAM allowance, measuring spawn/construction/recording overhead. It does not change dt/solver/contact/safety or qualify target WBC. Both commands require an independent immutable capsule, existing host-aware registry/lease and owner lock. Effects use the existing Journal/onceIO; no scheduler/allocator/service is added.

The world-model directory explicitly reuses six WM01 source adapter modules at pinned file SHA, with new WM02 action-time/state-role/recorder wrappers. No WM01 trainer/checkpoint, third-party OSC implementation or private WBC asset is copied.

The environment-first handoff is `90c0507333d2a9c6b26ee9c22278b050a9249fad`. `env_preflight` actually ran on Core, verifying nine core target members; fourteen initialization members were independently checked at their complete original pinned paths. Neither preflight nor the typed offline `control_snapshot` container grants target execution or implements a native exporter.

`neural_smoke` is implemented and actually completed on GPU0 at `21f2811f00aedb37b47e7b52d827d4ba1e7cb38e`: one 5×2048/GRU256 model, 200 main updates plus one reload continuation probe, and one H16 Actor/Critic update each, with saved/reloaded checkpoints. It deliberately accepts only synthetic engineering fixtures and finite existing-registry Core GPU0 bindings. It is not a real TRAIN replay loader, native reward training or a target learning loop. Actor updates freeze WM weights while retaining action-input gradients; decisions occur at atoms0/8.

Actual environments, commands, costs, memory, input/checkpoint hashes and remaining gaps are in [the E0–E2 report](../../reports/WM02/environment/EXECUTION_REPORT.md). Formal real-replay fit/collection, PPO, calibrated native counterfactuals, end-to-end real interaction and visual CLIs remain unimplemented. No empty stubs or new scheduler are supplied.

Real S follow-up implements `python -m scripts.wm02.replay --help`, `python -m scripts.wm02.check_compiler --help`, and finite `python -m scripts.wm02.train_real --help`. Numeric replay has an explicit 50-D S schema and scene/parent/model-connected holdout; no target BASE padding. GPU1 actually completed 200 main updates at `67721ff0bda5dee189bf4a73b6f8d43a454bd787`, then the metadata repair/evaluation at `d5a87ae6eb7eb9fa48c0265944355357aeebfe1e` used zero repeated main steps and one continuation probe. Full original checkpoint identities and negative prediction results: [real S report](../../reports/WM02/real-replay/EXECUTION_REPORT.md). This is prediction fitting, not formal native reward/Actor training. At that revision fresh source collection/real-Actor collect-loop CLI remained unimplemented.

The contact-identification successor now implements and actually runs `scripts.wm02.collect_contact`, `scripts.wm02.build_paired_replay`, `scripts.wm02.train_contact`, `scripts.wm02.analyze_contact`, and a separate `scripts.wm02.probe_contact_onset`. Their `--help` is side-effect-free. V1 freezes24 whole-scene16/4/4 partitions and five paired candidates per four states; both matched GPU1 arms complete200 updates before once-heldout evaluation. Future interventions preserve real history/hidden/current state; second decision uses predicted TCP. Analysis is numeric read-only and never opens a runtime lease or fits a model. The onset probe admits exactly the four diagnostic DEV parents and SHA-bound1.6s original complete snapshots under a new registered source-only identity; it is not a fresh TRAIN collector or a target permission. Actual15,616 new S transitions, 400 main＋2 recovery updates, model/data identities, negative results, costs and all closed runs: [contact report](../../reports/WM02/contact-identification/EXECUTION_REPORT.md). Formal reward, Actor proposal execution/learning and target-native loop CLIs remain unqualified or unimplemented.

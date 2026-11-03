# WM02 — target-native world-model policy learning

**Status: design only; runtime NOT_IMPLEMENTED.** No training, simulator or remote job was started by publication.

Start with [Codex handoff](../../docs/handover/2026-11-04/WM02_CODEX_EXECUTE.md), [full protocol](../../docs/protocols/WM02_GO2_ARX5_WORLD_MODEL_ACTOR_V1.md), [adapter/source map](../../docs/protocols/WM02_UPSTREAM_ADAPTER_MAP_V1.md) and [plan.json](plan.json).

Primary experiment: the actual Go2+ARX5 frozen-WBC backend, state Actor, native interaction, recurrent structured world model and short imagined policy optimization. PPO is the matched baseline. GR00T is not required. Fixed-base ARX5 and upstream OSC results are diagnostic domains, not target-system success.

This folder is independent of the unmerged WM01 PR342. It contains no train/collect/evaluate executable. Do not invent CLI execution claims. P0/P1 resolve backend, authority, action timing, recursive state, native restore/contact and throughput before long runs. Existing runs, safety and private assets remain unchanged.

Planning quantities, not results: 64 then 256 cumulative initial attempts; WM 200-update smoke and independent 5000-update pretraining; 100k pilot then 1M real atomic transitions per arm/seed; 2 arms x 3 seeds; 100 R0 TEST layouts giving 600 planned final episodes. Actual incomplete coverage must be reported separately.

Local CPU/programmatic checks and CI were NOT_RUN in this publication because execution tools were unavailable. Readback validates the Git publication only. Implementation must add real tests for the native interface, temporal equivalence, split isolation, gradients and restart semantics.

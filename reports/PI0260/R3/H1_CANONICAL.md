## PI0260-R3 H1 admitted — actual measured temporal ambiguity

Authority: `PI0260_R3_ALIAS_AWARE_LEARNING_MAINLINE_GO_V1`, commit `fd8e69b59a16aa90b1761d23b3a8f5a78cee8763`.
Execution source and receipts: [6912312](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/6912312/reports/PI0260/R3).

**New status: `H1_R3_ADMITTED_WITH_MEASURED_TEMPORAL_AMBIGUITY`.** R2 `H1_OBSERVABILITY_BLOCKED_EXACT_INPUT_FUTURE_ALIAS` and old DEV4/20 FAIL remain intact. `full_observability_proven=false`.

- Reused fixed COMPACT_30_10, one TRAIN-only statistics fit, full geometry/codec replay, and R2 HF PUBLIC fixed revision `3d248c1e54f8eb2caf771b067ef48cd995367a12`.
- Actual full processor pass: TRAIN 13,003 windows / 7,776 conditions / 2,198 aliased groups; KIN-VAL 4,374 / 2,512 / 800. No extra processor merges, mixed task/command identities, cross-split identical conditions or selected numeric geometry overlap.
- Uniform-window empirical deterministic XYZ floor (sqrt MSE): TRAIN 1.466 mm; KIN-VAL 1.426 mm. Frozen 6000-visit weighted floor is separately reported. These are mean-square data diagnostics, not model scores or median/p95 limits. Labels and duplicate-input rows are unchanged.
- Visually inspected 56 source RGB body/wrist pairs across both tasks/splits/commands, fixed phases and lowest required-cue margins. Scope and contact hashes are in `natural_image_review.json`. Prior full saved-array raster/geometry checks remain the population-wide numerical evidence.
- Eight actual supervised/free processor boundary cases passed: same visible condition, supervised action/mask only on loss path; forbidden fields rejected. Eight actual physical-loss/inference-decoder bindings passed, including values outside the q01/q99 interval; no clamp was applied.
- Frozen 6000 training visits (3000 M1/3000 M3, balanced commands), 156 Motion2 evaluation cases, original 64 O64 cases, and two global label-independent CPU Torch noise packets B0/B1 (seeds26031/26032). Across Base/G step0/final and both packets this is 1760 logical score rows. Twelve paired VAL semantic observations have actual distinct opposite-command futures; original rows remain scoring targets.

H1 receipt SHA256 `93432e44326bbb7f05455f6ece257288bf65e35506dc8c0ca61eb06c7b25e7b3`. No neural model loads, canary/main updates, quality queries, physical/WBC or TEST work at this milestone. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

**Next under existing GO:** bind pinned Base/G initializers and run real continuous8 versus 4+complete-Trainer-restore+4 canaries. Passing integrity directly releases both6000 main arms and fixed evaluation. This is a progress ledger, not the experiment terminal or a request for routine GO.

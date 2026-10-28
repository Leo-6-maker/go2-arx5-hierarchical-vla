# PI0262 decision: MIXED; prioritize instruction-assignment decoupling

**Diagnosis: MIXED.** The strongest controlled observation is sensitivity to wording and its original target/family exposure combinations. Some fixed-template TRAIN/KIN-VAL differences remain. Neither a language-only explanation nor a demonstrated pure visual/geometry failure explains the entire panel. The experiment is a completed development diagnosis, not a qualification PASS or a physical success.

## Evidence

The independent CPU replay covers all 1920 rows, all raw/decode/state/reference/noise identities, original metrics and gate results. There are zero ties, zero invalid rows and zero REJECT. Historical PI0261 qualification FAIL and all candidate/selection/deployment NONE fields remain unchanged.

P2000's four secondary pooled cells are:

| Noise | TRAIN T0/T1 | TRAIN V0/V1 | KIN-VAL T0/T1 | KIN-VAL V0/V1 |
|---|---:|---:|---:|---:|
| B0 | 99/120 | 86/120 | 29/40 | 25/40 |
| B1 | 106/120 | 88/120 | 28/40 | 28/40 |

The primary template results prevent these pools from hiding a failure. On TRAIN, P's T0 is 55/60 and 57/60; T1 is 44/60 and 49/60; V0 is 53/60 and 52/60; V1 is 33/60 and 36/60. On KIN-VAL the corresponding results are T0 17/20 and 14/20; T1 12/20 and 14/20; V0 14/20 and 16/20; V1 11/20 and 12/20.

Even the two already-supervised sentence forms fail on their newly crossed target/family combinations. P's original TRAIN combinations retain 59/60 and 60/60, while the other T0/T1 form applied to the same target/observation reaches only 40/60 and 46/60. C shows 50/60 and 51/60 versus 31/60 and 35/60; Donor shows 41/60 and 41/60 versus 22/60 and 27/60. Thus the original TRAIN entry count did not establish interchangeable use of even T0/T1.

The same-target, same-observation, same-noise wording comparison is directly controlled. For P TRAIN, T0→V1 turns 22/60 and 21/60 correct cases into wrong cases, with zero wrong→correct cases in either noise. V0→V1 produces 23 correct→wrong versus 3 wrong→correct at B0, and 24 versus 8 at B1. T0→T1 produces 16 versus 5 at B0 and 11 versus 3 at B1. These are correlated family measurements, not independent trial significance tests.

The direction/bias decomposition shows a boundary problem is common even when instruction ordering is positive. P TRAIN V1 has positive ordering with bias dominating in 26/30 and 23/30 families, nonpositive ordering in only 1/30 per noise, and both commands correct in 3/30 and 6/30. Across all four templates, P TRAIN has 140 both-correct, 90 positive-ordering/bias-dominating and 10 nonpositive-ordering measurements out of 240 correlated family/template/noise measurements. KIN-VAL has 34, 30 and 16 out of 80. This supports wording-conditioned target-choice offsets as an observable failure description; it does not locate a neural mechanism or justify a new loss.

Noise matters: P KIN-VAL both-command correctness crosses between B0/B1 for 4/10 families under T0, 4/10 under T1, and 1/10 each under V0/V1. B0/B1 are only two fixed actual packets.

The complete decomposition also retains the other models:

| Point / split | Both correct | Positive ordering, bias dominates | Nonpositive ordering | Correlated measurements |
|---|---:|---:|---:|---:|
| Donor TRAIN | 13 | 207 | 20 | 240 |
| C TRAIN | 58 | 146 | 36 | 240 |
| P TRAIN | 140 | 90 | 10 | 240 |
| Donor KIN-VAL | 11 | 48 | 21 | 80 |
| C KIN-VAL | 13 | 43 | 24 | 80 |
| P KIN-VAL | 34 | 30 | 16 | 80 |

P's large increase in both-correct choices coexists with a much smaller change in ordering-sign failures. The descriptive improvement is largely consistent with reduced bias domination, not simply creation of an instruction response where none existed. This motivates testing exposure-associated offsets before any new loss or backbone change; it does not prove a neural cause.

## Counterevidence and limits

- Unseen project wording is not uniformly bad: V0 transfers well on TRAIN and is better than T1 there. V0 also exceeds T0 on KIN-VAL at B1. Do not describe all held-out language as failed or claim the backbone cannot understand these words.
- Familiar wording does not fully close the split gap: T1 is only 12/20 and 14/20 on KIN-VAL, and T0 falls to 14/20 at B1. A language-only explanation is insufficient.
- All six color/command strata exist in both splits, but their frequencies differ. An explicitly secondary, post-hoc equal-stratum description gives P T0 B0 TRAIN 85.48% versus KIN-VAL 85.56%; the raw 91.67% versus 85% difference should not be called pure geometry loss. T0 B1 remains 90% versus 73.33%; T1 remains 69.26% versus 54.44% at B0 and 76.19% versus 71.11% at B1. No rows are deleted and no new gate is defined.
- Family split is not randomized pure geometry; color, position and appearance differ. There are 30 TRAIN and 10 development-validation families, with repeated commands/templates/noises. No independent-row significance claim is made.
- Original numeric proximity can coexist with the wrong target: P TRAIN V1 median XYZ is 6.27 mm and 4.34 mm despite 33/60 and 36/60 target choices. H40 entry discrimination is not full task completion, physics or deployment.

## One next intervention, for a new admission only

**Prioritize a TRAIN-only controlled change to the instruction assignment schedule: give each existing same-observation target pair both T0 and T1, balanced across target color, command and family.** Compare against the original instruction assignment while holding frames, states, expert labels, processor/statistics, initialization, S4 loss, trainable scope, optimizer/LR, update budget and paired noise plan fixed under the next separately approved protocol.

The sole changed factor is template assignment/exposure. Keep V0/V1 out of optimization and normalization, and keep original held-out evaluation available as held-out evaluation. This tests whether the large same-target recombination loss and wording-conditioned bias respond to removal of the existing template–target/family association. It does not assume this will solve residual scene transfer. Do not simultaneously change cameras, scenes, loss, adapters or backbone scope.

This recommendation has **not been executed**. PI0262 stops at `STOP_FOR_SINGLE_INTERVENTION_ADMISSION`; optimizer/backward/new data/render/physics/WBC/TEST/DEV-F/hardware remain zero, with all selection fields NONE.

Evidence: `FOUR_CELL_REPORT.md`, `CROSSED_RESULTS.csv`, `WORDING_FLIPS.csv`, `TARGET_CHOICE_DECOMPOSITION.csv`, `NOISE_BOUNDARY_CROSSINGS.csv`, `COLOR_COMMAND_GROUPS.csv`, `COMMON_SUPPORT.json`, `DIAGNOSTIC_SUMMARY.json`, `CPU_REPLAY.json` and the frozen `CROSSED_EVAL_CARD.json`.

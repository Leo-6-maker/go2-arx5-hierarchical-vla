# PI0262 COMPLETE — crossed diagnosis MIXED, next intervention not executed

`PI0262_CROSSED_GENERALIZATION_DIAGNOSIS_COMPLETE — STOP_FOR_SINGLE_INTERVENTION_ADMISSION`

Authority is the user's explicit PI0262 execution goal plus the fully supplied protocol, preserved at `docs/protocols/PI0262_CROSSED_LANGUAGE_GEOMETRY_DIAGNOSIS_V1.md`. PI0261 terminal evidence ref is `3e3983c8345d50eb322405084efd4f3c9381d97f`, canonical [5909947990](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5909947990). Frozen-card canonical: [5911463706](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5911463706). Original PI0261 COMPLETE, C/P complete qualification FAIL, historical reports/hashes/comments and all NONE fields remain unchanged.

## Actual scientific result

All 1920 fixed logical rows are complete and independently CPU replayed. The table pools two templates only as secondary description; primary per-template counts and XYZ/SO3/gate values are in `FOUR_CELL_REPORT.md` and `TEMPLATE_GROUPS.csv`.

| Point | Noise | TRAIN T0/T1 | TRAIN V0/V1 | KIN-VAL T0/T1 | KIN-VAL V0/V1 |
|---|---|---:|---:|---:|---:|
| Donor | B0 | 63/120 | 61/120 | 25/40 | 21/40 |
| Donor | B1 | 68/120 | 60/120 | 20/40 | 23/40 |
| C2000 | B0 | 81/120 | 64/120 | 22/40 | 19/40 |
| C2000 | B1 | 86/120 | 67/120 | 25/40 | 21/40 |
| P2000 | B0 | 99/120 | 86/120 | 29/40 | 25/40 |
| P2000 | B1 | 106/120 | 88/120 | 28/40 | 28/40 |

**Diagnosis: MIXED, with a strong wording/exposure-combination effect.** P's original TRAIN combinations remain 59/60 and 60/60; applying the other already-supervised T0/T1 template to the same target and observation gives only 40/60 and 46/60. TRAIN V0 gives 53/60 and 52/60, while V1 gives 33/60 and 36/60. Thus “unseen wording” is not a uniformly failing class, and the original TRAIN count did not establish interchangeable wording.

Controlled P TRAIN T0→V1 flips 22 and 21 correct cases to wrong, with zero wrong→correct in either noise. P TRAIN V1 has positive target ordering but bias domination in 26/30 and 23/30 families; only 1/30 per noise has nonpositive ordering. Across all templates, P TRAIN has 140 both-correct / 90 bias-dominating / 10 nonpositive-ordering family-template-noise measurements; KIN-VAL has 34 / 30 / 16. These are correlated measurements, not independent trials.

Some scene-family differences remain under fixed templates, but color/command frequencies differ. All six strata are represented in both splits, no rows are removed, and the supplementary equal-stratum description is explicitly post-hoc. P T0 B0's raw scene gap nearly disappears under this description, while T0 B1 and T1 retain gaps. Do not label the whole result pure geometry loss, a language-only mechanism, backbone non-generalization or physical success. There are zero ties, zero invalid rows and zero REJECT across the full panel.

**One next priority:** a separately admitted TRAIN-only controlled instruction-assignment intervention giving each original same-observation target pair both T0/T1, balanced by color/command/family. Hold data, initialization, processor/statistics, S4 loss, trainable scope, optimizer/LR, update budget and paired noise fixed. Keep V0/V1 outside optimization and normalization. This recommendation is unexecuted; no new loss/backbone/camera/data change is launched. Full evidence and counterevidence are in `NEXT_INTERVENTION_DECISION.md`.

## Completion and raw accounting

| Requirement | Actual evidence |
|---|---|
| Full protocol / historical source binding | Exact supplied protocol; full PI0261 ref and canonical; `SOURCE_BINDING_AUDIT.json` |
| Original 30 TRAIN / 10 KIN-VAL families and true colors | `SOURCE_TEMPLATE_SPLIT_MANIFEST.json`, original registry colors/sequence agreement, common index-0 observation checks |
| CPU own/opposite signs and strict ties | `S0_COMPLETION.json`: 80 expert positive/negative/tie controls; no model call |
| Actual exposure kept separate | R3 6000 visits, C/P 4000 actual supervised microbatches each; M1 T0/T1 counts R3 1474/1526, C/P each 1000/1000; V0/V1 project supervision 0 |
| Frozen card before first new query | `CROSSED_EVAL_CARD.json`, SHA256 `a6182faea209aa960ae82d06037b6f0e3cfb0ad87e38821458e62af54fc9987e`, pre-query release commit `93cf739742cba1a588725659ee2d548195f46535` |
| All fixed inference | 640 rows per point; 504 exact old-row reuses, 1416 new valid first queries; all raw/decode/reference/state/noise/S4 traces saved |
| Exact raw query/head accounting | `attempts.jsonl`: 1416/1512; `head_forwards.jsonl`: 5664/6048; every attempt has exactly four heads; invalid-only reserve used 0/96 |
| Independent replay | `CPU_REPLAY.json`: PASS all 1920, actual processed conditions, re-decoding, independent metrics/gate and original reused-array joins |
| Required analyses | `CROSSED_RESULTS.csv`, primary template/color/command strata, all 2880 wording comparisons, 960 direction/bias rows and 480 noise-boundary comparisons; no independent-row significance |
| Original source archives retained | PI0261/R3 fixed public receipts retained; reused outputs are also copied into this new self-contained result archive |
| Public recoverability | Fixed revision and 3875-member anonymous complete readback below |

Donor, C2000 and P2000 each used 472 new queries and 1888 heads. Only one model was loaded at a time on Corelab GPU0. No other project was stopped. Model inference, CPU replay and archive processes are terminal. No model weights or optimizer were copied to this result archive.

Optimizer updates/backward/canary/new scene/IK/render/statistics fitting/new O64 or M3 queries/1000-step queries/WBC/physics/TEST/DEV-F/hardware are all **0**. No monitor, mail, merge or next intervention was enabled.

## Fixed public evidence

Repository: `leoyuer/pi0246-gate-pass-campaign`.

- Revision: `266d1fb7e7c49909fa6b1efb6d900500c000efe4`.
- Prefix: `evidence/PI0262/crossed-M1-fixed-20260930`.
- Tree: `07b5ed73e363fcc8b9499631e82c7143ca9154d79f7010c50898c0991d59f0cc`.
- Members: 3875; source bytes: 239265370; tar bytes: 241899520.
- Tar SHA256: `6899c3633547bbe739e7562cba991249b80193cd8fe79ac2d0ebcdf5a8fad3c6`.
- `PUBLIC_RECEIPT.json`: `PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK` for manifest, entire tar and every member.

[Complete raw result archive](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/266d1fb7e7c49909fa6b1efb6d900500c000efe4/evidence/PI0262/crossed-M1-fixed-20260930/PI0262_crossed_evidence.tar), [member manifest](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/266d1fb7e7c49909fa6b1efb6d900500c000efe4/evidence/PI0262/crossed-M1-fixed-20260930/MEMBER_MANIFEST.json).

Scientific report/ledger evidence commit: [093bef1](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/commit/093bef1). The final receipt/index commit is appended when this canonical is posted. PR #77 remains pointer-only and unmerged.

`synthetic_research_candidate=NONE`; `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`. Stop at the single-intervention admission boundary.

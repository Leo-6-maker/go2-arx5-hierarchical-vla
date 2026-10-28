# PI0261 COMPLETE — both endpoints valid, original full qualification FAIL

`PI0261_INSTRUCTION_PAIRING_PILOT_COMPLETE — STOP_FOR_GENERALIZATION_OR_PHYSICAL_ADMISSION`

Authority: `6f2353a718cc877dddb3124029de76007eaa7df3`, canonical authorization [5903309366](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5903309366). PI0260-R3 COMPLETE and its two full-qualification FAIL results remain preserved.

C and P both completed the fixed 2000-update continuation from the same R3 Base6000 donor. Both complete final Trainers, all 2040 logical evaluation rows, independent CPU replays and public evidence/video are sealed. **Execution is COMPLETE; both endpoints fail the original complete research qualification.** `synthetic_research_candidate=NONE`, `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`.

## Scientific result

| Endpoint | KIN-VAL numeric groups | M1 instruction B0 / B1 | M1 paraphrase B0 / B1 | M3 instruction B0 / B1 | M3 paraphrase B0 / B1 | M1 TRAIN B0 / B1 |
|---|---|---|---|---|---|---|
| Donor | 4/4 PASS | 7/12 / 8/12 | 6/12 / 7/12 | 12/12 / 11/12 | 12/12 / 11/12 | 41/60 / 41/60 |
| C2000 | 4/4 PASS | 6/12 / 7/12 | 6/12 / 6/12 | 12/12 / 11/12 | 12/12 / 12/12 | 50/60 / 51/60 |
| P2000 | 4/4 PASS | 8/12 / 8/12 | 8/12 / 10/12 | 11/12 / 11/12 | 11/12 / 12/12 | 59/60 / 60/60 |

Every original language group requires at least 10/12 independently. P passes five of eight groups; C passes four. There are no semantic ties. Both retain zero REJECT and the four original geometric gates. P median XYZ ranges 6.2–8.7 mm, p95 13.7–15.0 mm, median SO3 .0020–.0116 rad. M3 TRAIN remains 60/60 in both noises for donor/C/P.

Balanced same-observation pairing strongly improves M1 learning on its TRAIN support relative to C, while the original M1 development-validation instruction/paraphrase gap remains. This controlled result supports an exposure effect on TRAIN choices; it does not establish adequate validation generalization. The validation families informed this study, so these are DEVELOPMENT VALIDATION results. O64 is disclosed under the synthetic processor: endpoint median XYZ remains about .183–.186 m and p95 .416–.421 m. It is not renewed historical qualification. Full own/opposite margins, per-command counts, families, rotation, empty-language comparisons and instruction-swap effects are in `MODEL_QUALITY_REPORT.md`, `SEMANTIC_BY_FAMILY.csv`, `MATCHED_INPUT_EFFECT.csv` and the three CPU receipts.

## Completion evidence and accounting

| Requirement | Authoritative evidence |
|---|---|
| Limited source/exposure audit; no new data generation | `S0_REPORT.md`, `S0_DIAGNOSIS.json`, `S0_EXPOSURE.json`; inherited R3 data/processor/labels |
| Frozen 60 TRAIN common-start pairs and fixed schedules | `PAIRING_PLAN.json` SHA256 `208f0bfbc97095369ce84329899bbbde67eb5f694ca8c9022f0566dc61f28477`; `PROCESSED_PAIR_BANK_AUDIT.json`: 60 nonlanguage byte equalities |
| Real paired accumulation and exact resume | C/P `*_CANARY_COMPARE.json`: continuous8 versus split4+restore+4, model/optimizer/LR delta 0, packets equal and modes restored; original sample/RNG/S4/loss/parity/gradient logs preserved under `canary/` |
| Both actual main endpoints | C PBS 639370→639399; P 639525→639815, all exit 0; each final Trainer global_step=max_steps=2000; current PBS history excerpt in `PBS_TERMINAL_STATES.txt` |
| Original S4 loss, gradients, modes and paired random consumption | `MAIN_TRAINING_AUDIT.json` PASS; 4000 sample/RNG/S4/loss rows per arm, 2000 exact cross-arm replay examples; four positive finite first-update Euler gradient norms per arm |
| Final-only fixed evaluation and independent replay | Three 680-row CPU receipts PASS; original 440 donor rows reused; C/P each 680 valid saved cells; fixed `FIXED_EVAL_CARD.json` |
| Complete 1000/2000 Trainers durable | Four fixed HF versions below, each 16-file full readback PASS; no intermediate quality screen |
| Video from saved arrays | `VIDEO_RECEIPT.json`: original fixed selection [48,50,126,128], 240 decoded frames at 20 fps, SHA256 `742366a683f27dfd5c02ba02219799cc4593085987745b05235be39615979afe`; visible KINEMATIC/OFFLINE/NO PHYSICS watermark |
| Public result recovery | `PI0261_PUBLIC_RECEIPT.json`: anonymous full-file and 3238-member readback PASS; reused R3 original arrays recover through `REUSED_R3_PUBLIC_RECEIPT.json` |

Main updates: 4000; discarded canary updates: 32; extra completed recovery updates: 0; raw completed updates: **4032/4288**. Main supervised microbatches: 8000. The six actual S4 logs contain 8064 supervised forwards (4000+16+16 per arm); six first-use parity receipts each have 4 reference+4 candidate heads. With native gradient checkpointing disabled, raw training head accounting is **72624/100000** = 8064 native + 32256 S4 forward + 32256 Euler backward recompute + 48 parity. The failed initial prefetch consumed zero model forward/update. P canary's quota error occurred during the redundant final model-view save after complete checkpoint8; independent CPU exact-resume comparison passed, and no canary was repeated. Its raw exit1 is retained.

New quality attempts: **1516/1728**; inference heads: **6064/6912**. All 2040 logical denominators remain; exact cache reuse provides no extra scientific cases. No intermediate checkpoint selection, recipe adjustment from C's result or quality retry occurred. WBC/physics/TEST/DEV-F/hardware counts are all 0. Scheduler/process checks show the PI0261 producers are terminal. PR #77 remains OPEN and unmerged; no monitor was created.

## Recoverable models and result

Public repository: `leoyuer/pi0246-gate-pass-campaign`.

| Trainer | Fixed revision | Prefix |
|---|---|---|
| C1000 | `407d0ca0c7518c54ed68b041dc4fda2c6ed7cdf9` | `arms/PI0261/C/checkpoint-1000` |
| C2000 | `ce5122232b3577005ecabb4cdd95f252cde5b2a6` | `arms/PI0261/C/checkpoint-2000` |
| P1000 | `c04c41909153d3db319f4c232f6fbcb5fd07f794` | `arms/PI0261/P/checkpoint-1000` |
| P2000 | `875abdde2cc7c9e6858fac008e53f2f4b3375255` | `arms/PI0261/P/checkpoint-2000` |

C2000 tree: `c476ded1519fe792372c5de50ca19bed1e71e60a1eb484efa5b1a5da71626f33`; P2000 tree: `882931d754cb25653105a9213fe0dbeb4d2b14e463efb27e49c6101fd5fc5e4b`.

Result revision **`923b0e4239db9ad3292b644cec74f7c9aa648ccc`**, prefix `evidence/PI0261/controlled-pairing-fixed-20260930`, tree `c05adba5591f1e66f45f6f85ba7bb6f17eb9a8bc97c19d30cf48d740cce650c2`, 3238 members. [Direct quality report](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/923b0e4239db9ad3292b644cec74f7c9aa648ccc/evidence/PI0261/controlled-pairing-fixed-20260930/MODEL_QUALITY_REPORT.md), [video](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/923b0e4239db9ad3292b644cec74f7c9aa648ccc/evidence/PI0261/controlled-pairing-fixed-20260930/PI0261_fixed_semantic_Donor_C_P.mp4), [complete evidence tar](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/923b0e4239db9ad3292b644cec74f7c9aa648ccc/evidence/PI0261/controlled-pairing-fixed-20260930/PI0261_fixed_offline_evidence.tar).

The donor's original 440 outputs remain at R3 result revision `45965a66d854bfcdc3e31dcb6ac8df87e21f4cb1`, prefix `evidence/PI0260/R3-fixed-eval-20260930`, with its original anonymous complete readback receipt preserved. The evidence uploader needed one path-check repair to compare resolved paths on both sides of Corelab's home symlink; scientific computation was unchanged.

Stop at the authorized generalization/physical-admission boundary with both quality FAIL results and all evidence preserved.

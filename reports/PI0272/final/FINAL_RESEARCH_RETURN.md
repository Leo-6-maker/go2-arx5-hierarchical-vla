# PI0272 R1 — matched direct readout fitting: complete fixed results

## Conclusion

Both registered readouts learn the consumed corrective-reference support. **The predeclared local DIRECT advantage over the matched trained S4 control is not established.** On the primary 24 M3 terminal conditions, both fits remove all opposite-direction outputs under the unchanged reporting guard. DIRECT's median per-condition prefix MSE minus mean S4FIT MSE is **+4.59272529e-7 m2**, with lower MSE on only10/24 conditions. DIRECT is better than old AGG on23/24, but S4FIT also improves strongly. This is counterevidence to attributing the residual repair uniquely to a direct head.

DIRECT has lower marginal terminal prefix/full SO3 and full XYZ medians, while the primary registered paired MSE criterion is worse. Its slightly lower marginal prefix XYZ median must not be substituted for that paired criterion. The primary direction rates tie at zero, rather than favoring DIRECT. Early-support XYZ worsens relative to AGG in both tasks for both fits; DIRECT additionally worsens M1 guard XYZ/SO3 relative to S4FIT. The registered combination of primary position, direction, rotation, gate and guard conditions therefore does not establish superiority.

Execution completion and scientific advantage are separate. These are developed TRAIN-support fitting results, not task completion, blind TEST, generalization, WBC or physical transfer. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

## Authority, source reuse and actual execution

- Base release `e62e6a469dce1f4d3df5d49b38bad4a3c627e969`, Issue11 canonical5979236384; R1 release `8a2ffbefe0c57536c0a7e340da0616c7987c74f1`, canonical5980876835. Same branch `pi0267/executor-decoupled-recovery`, no new science goal or budget reset.
- Closed PI0271 terminal5978753654 / closeout90489ca reused; AGG4096 donor HF revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`, prefix `arms/PI0270/AGG/checkpoint-4096`, tree `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`. Old valid queries and archive receipts were reused.
- Exactly296 designated consumed conditions: M1 Q2/Q3 48/48, terminal32; M3 Q2/Q3 48/48, entry/midarc/terminal24 each. 24 TRAIN families overall; components only two families. Actual per-arm2048 microbatches match the CPU plan:272 conditions seven times and24 six times. NEWFIT192 has zero exact condition/label overlap. No EVAL label entered training.
- Both real continuous8 versus4+full-save/fresh-resume4 canaries PASS:1030 model tensors and1413 optimizer tensors have maximum difference0.0; cursor, scheduler and all saved RNG states match. All4096 main microbatches match source inputs, labels and paired native epsilon/tau packets. Both complete1024 updates from the same donor with fresh optimizer; full512/1024 Trainers retained and read back.
- Training generated queries/Euler states FP32, native epsilon BF16, original mixed parameter storage/autocast; quality model-only BF16 and BF16 queries. DIRECT zero query follows explicit phase feature dtype. Physical loss/decode remain FP32. `.eval()` does not define the phase. Same-phase parity maximum raw/decoded difference0 with identical original gate issues; actual generated gradients cover4 S4 nodes or1 DIRECT node.
- Same471 trainable/559 frozen tensor identities, LR1e-5, warmup10/cosine1024, accumulation2, native loss plus .4 position+.1 increment+.1 rotation+.1 peak-speed. Same PREFIX temporal weights. DIRECT uses the same modules at zero latent/bucket0 once, decoded directly; no Euler, added head, target leakage or label input. S4 remains the original four-step noise generator. The intervention jointly changes query initialization and readout interpretation; it does not isolate stochasticity from depth.
- Actual per-arm exposure is in MAIN_TRAINING_AUDIT.json: M1 each command Q2/Q3 168/168 and terminal108 microbatches; M3 each command Q2/Q3 168/168, entry82, midarc82, terminal80. The full small-source export contains166 exact files with source hashes, including actual logs and kernel binding. Separate per-phase CPU kernel receipts were NOT_RECORDED; startup binding and every actual processed-input hash were checked against the qualified Prescott CPU admission.

## Primary result: consumed M3 terminal

Marginal descriptive medians below use original j0/j1 for stochastic policies and one output per condition for DIRECT. The /48 noisy outputs are not48 independent scenes. All directions are defined under the original1e-9m reporting guard.

| Policy | Outputs | Prefix XYZ mm | Prefix SO3 rad | Full XYZ mm | Full SO3 rad | Wrong direction | REJECT |
|---|---|---|---|---|---|---|---|
| AGG4096 | 48 | 8.954 | 0.006020 | 24.059 | 0.011424 | 35/48 | 0 |
| DIRECT0 | 24 | 11.077 | 0.027799 | 32.969 | 0.038328 | 19/24 | 0 |
| S4FIT1024 | 48 | 2.668 | 0.004510 | 5.882 | 0.009035 | 0/48 | 0 |
| DIRECT1024 | 24 | 2.573 | 0.004227 | 5.680 | 0.007631 | 0/24 | 0 |

Teacher endpoint displacement median13.077mm, range0.00000326–22.642mm. Some values are almost zero: signs alone are fragile and do not prove a path verdict. Median signed teacher-direction progress is AGG -1.839mm, DIRECT0 -5.362mm, S4FIT +9.218mm, DIRECT1024 +8.493mm. Full continuous vectors/magnitudes and first-point/.4s endpoint/suffix/gripper/gate details remain in ALL_POINTS3504.csv.

Against S4FIT, median paired MSE deltas for family004/command0,005/0,004/1,005/1 are +9.43210e-7,+2.01992e-6,-2.63242e-7,+2.14849e-6 m2. Only one of four family-command groups favors DIRECT. Both commands and both families have zero final wrong-direction outputs in both fits.

| Family | Command | Policy | Outputs | Prefix XYZ mm | Prefix SO3 rad | Full XYZ mm | Full SO3 rad | Wrong direction | REJECT |
|---|---|---|---|---|---|---|---|---|---|
| pi0265-family-004 | 0 | AGG4096 | 12 | 8.985 | 0.005735 | 22.759 | 0.011202 | 12/12 | 0 |
| pi0265-family-004 | 0 | DIRECT0 | 6 | 10.565 | 0.023878 | 32.969 | 0.033202 | 6/6 | 0 |
| pi0265-family-004 | 0 | S4FIT1024 | 12 | 2.238 | 0.004272 | 5.778 | 0.009673 | 0/12 | 0 |
| pi0265-family-004 | 0 | DIRECT1024 | 6 | 2.407 | 0.003982 | 5.250 | 0.007631 | 0/6 | 0 |
| pi0265-family-004 | 1 | AGG4096 | 12 | 9.478 | 0.012138 | 24.307 | 0.022036 | 7/12 | 0 |
| pi0265-family-004 | 1 | DIRECT0 | 6 | 11.060 | 0.027457 | 34.885 | 0.039107 | 5/6 | 0 |
| pi0265-family-004 | 1 | S4FIT1024 | 12 | 2.643 | 0.005282 | 7.453 | 0.009653 | 0/12 | 0 |
| pi0265-family-004 | 1 | DIRECT1024 | 6 | 2.631 | 0.005462 | 5.667 | 0.010068 | 0/6 | 0 |
| pi0265-family-005 | 0 | AGG4096 | 12 | 8.448 | 0.004720 | 22.017 | 0.008596 | 8/12 | 0 |
| pi0265-family-005 | 0 | DIRECT0 | 6 | 9.467 | 0.026913 | 31.198 | 0.038582 | 4/6 | 0 |
| pi0265-family-005 | 0 | S4FIT1024 | 12 | 2.799 | 0.004124 | 5.829 | 0.008351 | 0/12 | 0 |
| pi0265-family-005 | 0 | DIRECT1024 | 6 | 3.085 | 0.004325 | 7.052 | 0.008965 | 0/6 | 0 |
| pi0265-family-005 | 1 | AGG4096 | 12 | 11.438 | 0.009050 | 26.991 | 0.015749 | 8/12 | 0 |
| pi0265-family-005 | 1 | DIRECT0 | 6 | 13.449 | 0.031547 | 35.309 | 0.040811 | 4/6 | 0 |
| pi0265-family-005 | 1 | S4FIT1024 | 12 | 4.158 | 0.004772 | 7.567 | 0.008170 | 0/12 | 0 |
| pi0265-family-005 | 1 | DIRECT1024 | 6 | 4.087 | 0.003211 | 7.095 | 0.006687 | 0/6 | 0 |

## All consumed strata and secondary M1 target

Each table entry is the median of DIRECT MSE minus that SAME condition's mean over j0/j1. Finite REJECT remains included; no marginal-median subtraction or noise-as-scene replication.

| S4 comparator | Task | Bucket | Conditions | Median per-condition DIRECT minus mean S4 MSE (m2) | DIRECT lower MSE conditions |
|---|---|---|---|---|---|
| AGG4096 | M1 | M1_TERMINAL | 32 | -7.73595202e-07 | 20 |
| AGG4096 | M1 | ORIGINAL_Q2 | 48 | -9.59077883e-06 | 33 |
| AGG4096 | M1 | ORIGINAL_Q3 | 48 | -1.14717974e-05 | 37 |
| AGG4096 | M3 | M3_ENTRY | 24 | -6.87005783e-06 | 21 |
| AGG4096 | M3 | M3_MIDARC | 24 | -9.700646e-07 | 15 |
| AGG4096 | M3 | M3_TERMINAL | 24 | -6.67133945e-05 | 23 |
| AGG4096 | M3 | ORIGINAL_Q2 | 48 | -1.31105031e-06 | 38 |
| AGG4096 | M3 | ORIGINAL_Q3 | 48 | -1.01483114e-06 | 39 |
| S4FIT1024 | M1 | M1_TERMINAL | 32 | 5.010654e-07 | 11 |
| S4FIT1024 | M1 | ORIGINAL_Q2 | 48 | 5.88859575e-07 | 22 |
| S4FIT1024 | M1 | ORIGINAL_Q3 | 48 | -3.90794307e-07 | 25 |
| S4FIT1024 | M3 | M3_ENTRY | 24 | 2.87972858e-09 | 12 |
| S4FIT1024 | M3 | M3_MIDARC | 24 | 1.67445209e-07 | 7 |
| S4FIT1024 | M3 | M3_TERMINAL | 24 | 4.59272529e-07 | 10 |
| S4FIT1024 | M3 | ORIGINAL_Q2 | 48 | -9.55094519e-08 | 28 |
| S4FIT1024 | M3 | ORIGINAL_Q3 | 48 | -1.9673695e-07 | 32 |

Across the96 secondary M1 Q2/Q3 conditions, DIRECT-minus-S4FIT median paired MSE is +1.78785107e-7m2; command0 +9.21381681e-7, command1 -6.20793418e-8. Against AGG it is -1.04804909e-5m2. M1 Q2 prefix XYZ medians are AGG5.121mm, S4FIT3.794mm, DIRECT3.964mm; Q3 j0/j1 are7.401,5.579,5.760mm. These support improved fitting in both arms with mixed direct-versus-S4 differences. M1 terminal, M3 entry and midarc tradeoffs are retained above and in all family/command tables.

DIRECT0 is a distinct, initially poorly calibrated effective policy:18/296 finite original-gate rejections, including M1 Q2 command1 four, Q3 command0 one, Q3 command1 thirteen. After fitting, S4FIT has0/1360 rejects and DIRECT1024 has0/488 distinct evaluated outcomes (plus4 identical repeats). Reused AGG retains its1/1360 finite Q3 j6 angular-speed rejection. No nonfinite/invalid-rotation outputs were removed or retried. DIRECT0 improvement alone is not evidence against S4.

## Early-support spillover guard: all192 retained

These original Q0/Q1 inputs are not a blind holdout or LEGACY qualification. Guard prefix XYZ medians AGG→S4FIT→DIRECT are M1 2.701→4.264→5.026mm, M3 2.120→3.525→3.264mm. M1 wrong directions0→3→1/96; M3 1→1→1/96. All guard policies have zero rejects. Rotation has mixed tradeoffs, explicitly retained below.

Paired guard deltas (after minus before) avoid differences of marginal medians:

| Task | Before | After | Conditions | Paired prefix XYZ delta mm | Paired prefix SO3 delta rad | XYZ worse conditions |
|---|---|---|---|---|---|---|
| M1 | AGG4096 | S4FIT1024 | 96 | 1.8157 | 0.0043973 | 76 |
| M1 | AGG4096 | DIRECT1024 | 96 | 2.3689 | 0.0032547 | 84 |
| M1 | S4FIT1024 | DIRECT1024 | 96 | 0.4084 | 0.0003777 | 62 |
| M3 | AGG4096 | S4FIT1024 | 96 | 0.9530 | -0.0005614 | 70 |
| M3 | AGG4096 | DIRECT1024 | 96 | 0.7608 | -0.0018623 | 72 |
| M3 | S4FIT1024 | DIRECT1024 | 96 | -0.0366 | -0.0004230 | 48 |

| Task | Command | Bucket | Policy | Outputs | Prefix XYZ mm | Prefix SO3 rad | Full XYZ mm | Full SO3 rad | Wrong direction | REJECT |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 0 | ORIGINAL_Q0 | AGG4096 | 24 | 2.130 | 0.010140 | 10.750 | 0.031126 | 0/24 | 0 |
| M1 | 0 | ORIGINAL_Q0 | S4FIT1024 | 24 | 4.310 | 0.004579 | 16.819 | 0.011486 | 3/24 | 0 |
| M1 | 0 | ORIGINAL_Q0 | DIRECT1024 | 24 | 4.747 | 0.005074 | 21.858 | 0.014922 | 1/24 | 0 |
| M1 | 0 | ORIGINAL_Q1 | AGG4096 | 24 | 4.117 | 0.008915 | 9.705 | 0.017905 | 0/24 | 0 |
| M1 | 0 | ORIGINAL_Q1 | S4FIT1024 | 24 | 6.025 | 0.011686 | 12.145 | 0.021183 | 0/24 | 0 |
| M1 | 0 | ORIGINAL_Q1 | DIRECT1024 | 24 | 5.987 | 0.011341 | 12.686 | 0.022779 | 0/24 | 0 |
| M1 | 1 | ORIGINAL_Q0 | AGG4096 | 24 | 1.658 | 0.003351 | 4.761 | 0.006830 | 0/24 | 0 |
| M1 | 1 | ORIGINAL_Q0 | S4FIT1024 | 24 | 3.978 | 0.008918 | 12.618 | 0.016171 | 0/24 | 0 |
| M1 | 1 | ORIGINAL_Q0 | DIRECT1024 | 24 | 4.102 | 0.006199 | 13.278 | 0.009742 | 0/24 | 0 |
| M1 | 1 | ORIGINAL_Q1 | AGG4096 | 24 | 4.080 | 0.009940 | 10.025 | 0.021072 | 0/24 | 0 |
| M1 | 1 | ORIGINAL_Q1 | S4FIT1024 | 24 | 4.187 | 0.020388 | 10.488 | 0.034273 | 0/24 | 0 |
| M1 | 1 | ORIGINAL_Q1 | DIRECT1024 | 24 | 7.452 | 0.029065 | 11.397 | 0.053099 | 0/24 | 0 |
| M3 | 0 | ORIGINAL_Q0 | AGG4096 | 24 | 2.468 | 0.008771 | 11.362 | 0.029312 | 0/24 | 0 |
| M3 | 0 | ORIGINAL_Q0 | S4FIT1024 | 24 | 4.395 | 0.004084 | 13.586 | 0.006718 | 0/24 | 0 |
| M3 | 0 | ORIGINAL_Q0 | DIRECT1024 | 24 | 3.559 | 0.004169 | 13.340 | 0.011118 | 0/24 | 0 |
| M3 | 0 | ORIGINAL_Q1 | AGG4096 | 24 | 2.736 | 0.006031 | 5.544 | 0.012202 | 1/24 | 0 |
| M3 | 0 | ORIGINAL_Q1 | S4FIT1024 | 24 | 2.766 | 0.005724 | 4.971 | 0.011303 | 1/24 | 0 |
| M3 | 0 | ORIGINAL_Q1 | DIRECT1024 | 24 | 2.887 | 0.005507 | 5.157 | 0.011620 | 1/24 | 0 |
| M3 | 1 | ORIGINAL_Q0 | AGG4096 | 24 | 1.582 | 0.006606 | 6.826 | 0.021123 | 0/24 | 0 |
| M3 | 1 | ORIGINAL_Q0 | S4FIT1024 | 24 | 4.882 | 0.005647 | 16.306 | 0.008934 | 0/24 | 0 |
| M3 | 1 | ORIGINAL_Q0 | DIRECT1024 | 24 | 4.769 | 0.003710 | 17.930 | 0.007046 | 0/24 | 0 |
| M3 | 1 | ORIGINAL_Q1 | AGG4096 | 24 | 1.680 | 0.005161 | 3.982 | 0.009671 | 0/24 | 0 |
| M3 | 1 | ORIGINAL_Q1 | S4FIT1024 | 24 | 1.528 | 0.006205 | 3.746 | 0.012487 | 0/24 | 0 |
| M3 | 1 | ORIGINAL_Q1 | DIRECT1024 | 24 | 1.661 | 0.006487 | 3.681 | 0.012811 | 0/24 | 0 |

## Complete K8 banks, no averaging policy

All192 banks (two policies×96 Q3 inputs) are complete; no smaller K or replacement draw. The reused finite AGG rejection remains in its bank. DIRECT is evaluated once and logically compared to each matched bank, not counted as eight independent draws.

| Policy | Task | Banks | Median prefix MSE m2 | Median empirical mean-error share % | Median within-bank dispersion m2 |
|---|---|---|---|---|---|
| AGG4096 | M1 | 48 | 4.84586156e-05 | 90.510 | 4.2608462e-06 |
| AGG4096 | M3 | 48 | 2.70526428e-06 | 91.996 | 1.81287221e-07 |
| S4FIT1024 | M1 | 48 | 3.54044008e-05 | 90.981 | 2.7061893e-06 |
| S4FIT1024 | M3 | 48 | 1.64840542e-06 | 85.748 | 2.28166626e-07 |

DIRECT-minus-S4FIT completeK8 per-condition median prefix MSE is +3.41596882e-7m2 for M1 Q3 and -1.72637739e-7m2 for M3 Q3. These are finite-bank descriptive mean-error/dispersion identities, not population bias estimates; no averaged trajectory was executed. No K8 claim transfers to terminal components or visited-state A3.

## Compute, correctness and preservation

- One Hopper H200 allocation646325, start2026-10-04 22:35:05SGT, end23:31:21, exit0. Official2048 updates plus32 planned canary updates =2080; zero technical recovery updates. Actual training heads24987/32768, including parity and recomputation. S4FIT main18440 heads /1890.6555s loop; DIRECT main6145 heads /937.9158s. Canaries304/98 heads. Equal updates are not equal FLOPs; loop times exclude model load/archive.
- Exactly2148/2560 quality attempts,6228/7168 heads, zero technical neural retries. S4FIT1168 support+192 guard; DIRECT0 296; DIRECT1024 296+192 guard+4 repeats. All four repeats have exact native output and saved-array bytes, generation leaves RNG unchanged. The1360 reused AGG outputs plus2144 non-repeat new outputs yield3504 metric rows.
- CPU replay PASS for all2148 new predictions: saved S4 Euler and FP32 decode; DIRECT saved raw decode without fictitious Euler or recomputing its neural network. Original gate/metric functions are shared; physical metrics have independent FP64 arithmetic cross-checks, per-condition arithmetic checks, full-bank identities, exact output/source hashes. CPU audit makes zero new neural/gradient/optimizer calls.
- All four complete Trainers have25 files each, fixed HF revisions and full-file readback PASS (model, optimizer, scheduler, RNG, config/processor/statistics, provenance and explicit mode). Local recoveries were not deleted; Corelab only received model/config files.

| Arm | Step | Fixed HF revision | Complete tree SHA256 |
|---|---|---|---|
| S4FIT | 512 | 0e12a801640fafdddbd80c8c7a49a95424b5cc8e | 8ef5a9190b4fcc4fa2570ae1d1c21d4e0669d2443f86ff82141badc2fb6a4e71 |
| S4FIT | 1024 | dadca28b18645f66d502c230d368378c6335da2c | b49b1234dd33ba98dc1ae789069256ed194bc949f6ccff65beeb5c26219880ba |
| DIRECT | 512 | b6c6ae4ebe1f4c6a9b9308f0bb20f0e6008a52c5 | d2df6662d226f1ea3c9ff298c1e80a0dd828ed25a329a2e08b4fc20f46aeb484 |
| DIRECT | 1024 | 888ec80628932e5dec294330e69cf171d13a4122 | e25154502f778e6a3c1c47e4b5b4260e9268f8a3e5308871bd9dc94f2c9ce67e |

HF repo `leoyuer/pi0246-gate-pass-campaign`, prefixes `arms/PI0272/{arm}/checkpoint-{step}`. DIRECT weights require the explicit DIRECT mode sidecar; stock S4 loading is a different effective policy. Prediction archive revision `e7cf79062be247d298ea50b9a96999c27e335927`, prefix `evidence/PI0272/matched-direct-fitting-20261004/predictions`, tree `cbb096a9fd550c6006273a30e557dc5d861edf649a69838f212121a8629ddd57` preserves all4296 output/receipt members and has anonymous full-file/all-member readback. The summary archive and final canonical receipts bind this report after generation.

All source cards, raw budget journals, every condition/family/command metric, full vectors, complete gate reasons, scope audits, four readbacks and166-file training-source export are retained. No new video is required because new episodes/renders/WBC/physics/teacher labels/hardware are all0. No monitor, mail, broad cleanup, PR77 pointer or merge.

## Decision boundary

**PI0272_MATCHED_DIRECT_READOUT_FITTING_COMPLETE — STOP_FOR_FULL_SUPPORT_FEEDBACK_OR_FITTING_DECISION.** The bounded comparison closes with mixed fitting findings and no demonstrated primary DIRECT superiority. Additional same-support fitting repairs the terminal defect in both readouts while losing early-support accuracy. This does not establish that data quantity alone is the cause, that the model cannot represent the task, or that all direct architectures/objectives fail.

A later PI decision may address full-support/feedback coverage while explicitly retaining these guard losses; no such run is authorized here. The separate two-family M1 development/visited-state gap was not reevaluated or solved. PI0270 original/reliable VALID_QUALITY_FAIL, its reliable6/64 and positive LEGACY result, PI0269/68 FAIL and PI0267 WBC PARTIAL remain unchanged. No task threshold or selector was created; selected/deployment remain NONE.

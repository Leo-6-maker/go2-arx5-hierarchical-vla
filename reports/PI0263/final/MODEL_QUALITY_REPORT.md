# PI0263 — completed F/X template counterbalancing pilot

**Execution:** both arms completed 2000 updates. **Quality:** F2000 and X2000 are valid FAIL on both qualifications. **Synthetic research candidate / formal selected / deployment:** NONE / NONE / NONE.

Authority: commit `1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28`, Issue #11 canonical 5913659202, PR #77 pointer 5913663574. PI0261 full qualification FAIL and PI0262 COMPLETE/MIXED remain unchanged.

## Scientific finding

Counterbalanced X learned the newly supervised TRAIN T0/T1 combinations: original bindings are 60/60 in both noises, other-T bindings 60/60 and 58/60; F retains 45/60 and 45/60 on other-T. This supports the benefit of the complete counterbalanced assignment policy in this setting. Its marginal template frequencies also differ; this does not isolate decorrelation at fixed marginals.

Transfer is incomplete. X fails original M1 SEMANTIC/B1 at 8/12 (c0=3/6, c1=5/6). The new KIN-VAL criterion fails T1/B0=16/20, V0/B1=14/20 and V1/B0=16/20. All eight crossed numeric groups pass, so low geometric errors alone do not establish correct target choice. X improves five of eight robustness groups to passing versus F two, but the preregistered criterion requires all eight.

F also improves multiple panels versus P. Both arms share another 2000 updates and the repeated-unit sampling design; their common gains cannot be credited uniquely to counterbalancing. X KIN-VAL V0/B1 drops below F (14 versus 15), and TRAIN V1/B0 drops below F (40 versus 42). The result remains MIXED rather than universal improvement.

X T0/T1 TRAIN recombinations are now TRAIN-exposed. V0/V1 remain absent from optimization and statistics fitting, but this is DEVELOPMENT VALIDATION informed by prior research, not blind TEST or necessarily unseen base-model language. There are 30 TRAIN and 10 KIN-VAL geometry families; template/noise rows are correlated. No binomial independence, unique neural mechanism, physical generalization, WBC or hardware success is claimed.

## Frozen execution and actual integrity

Both arms start at PI0261 P2000 `875abdde2cc7c9e6858fac008e53f2f4b3375255`, `arms/PI0261/P/checkpoint-2000`. Plan SHA256 `602dc213ca91f2dcd5099bd97a45587fe5fe1e0d205243596d48e07c5f662617`. 1000 original source units, 250 per M1/M3 × PAIR/REPLAY stratum, each visited twice; each arm 4000 examples, M1/M3 each 2000, 50% common-start and 50% ordinary replay. X M1 T0/T1 each 1000 examples. Source frequencies are inherited, not globally equalized.

Actual 8000 microbatches verify source, color, labels, masks and nonlanguage inputs. Corresponding native epsilon/tau and S4 starting packets match within units and across arms. Both real continuous8 versus 4/save/fresh-process-restore/4 canaries pass with zero model, optimizer and scheduler differences. All four Euler steps retain gradients and modes restore; no expert prefix or padding intervention. 559 frozen tensors remain byte-identical to P; all 471 trainable tensors changed. BF16 frozen backbone, FP32 trainable head and original BF16 inference are retained.

Fresh AdamW LR1e-5, betas(.9,.999), eps1e-8, weight decay1e-5, clip1, warmup10/cosine2000, microbatch1/accumulation2. Inherited full-S4 objective: native + .4 position + .1 increment + .1 rotation + .1 peak-speed margin. Read-only final audit replays actual FP32 loss arithmetic exactly. The first reader incorrectly compared a double-precision sum; the reader was corrected without changing training, coefficients or tolerances.

F main PBS640396 and X main PBS640622 finish exit0. PBS640356 failed at the CRLF shebang before Python (zero updates/heads); its logs remain. No quality retries, duplicate healthy jobs or scientific recovery updates.

## Accounting

| Item | Actual | Limit / disposition |
| --- | --- | --- |
| Main updates / microbatches | 4000 / 8000 | 4000 / 8000 |
| Discarded canary updates | 32 | 32 |
| Completed technical recovery updates | 0 | 256 |
| All completed updates | 4032 | 4288 |
| Raw training heads | 72624 | 100000 |
| Unique logical rows P/F/X | 1152 / 1152 / 1152 | 3456 |
| Legacy / crossed memberships per point | 680 / 640 | 168 overlap; 1152 unique |
| Donor rows reused | 1152 | 0 new donor queries |
| New valid queries / inference heads | 2232 / 8928 | 2432 / 9728 hard cap |
| Exact forward reuse F/X | 36 / 36 | labels individually rescored |
| Invalid attempts / quality retries | 0 / 0 | no spare-budget probes |
| New scenes / renders / IK / stats fits | 0 / 0 / 0 / 0 | 0 |
| Physics / WBC / TEST / DEV-F / real hardware | 0 / 0 / 0 / 0 / 0 | 0 |

First valid outputs are retained. A sole producer performed the independent computation-path CPU replay of all 3456 saved rows; this is not external signoff. Original decoder, actual noise, labels, gate and denominators are verified. No checkpoint1000 quality query or alternate candidate search was performed.

CPU receipt scope erratum: the completed replay checked3456 rows, but its final loop cursor X2000 was accidentally used to label the scope and donor-reuse count. The original receipt and computation code are retained under evidence/cpu_schema_erratum. A narrow check of the actual receipt expressions and all3456 record/tensor hashes corrected only the scope label and donor count. Metrics, provenance, quality FAIL and model-query totals are unchanged.

## Original qualification — unchanged gates

Four numeric groups each require median XYZ≤20mm, p95≤35mm, median SO3≤.050rad and zero REJECT. Eight semantic groups each require ≥10/12; ties and rejects are not correct.

| Point | Task | Noise | N | median XYZ mm | p95 XYZ mm | median SO3 rad | Reject | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2000 | M1 | B0 | 16 | 6.256 | 13.748 | 0.011085 | 0 | True |
| P2000 | M1 | B1 | 16 | 8.198 | 14.980 | 0.011623 | 0 | True |
| P2000 | M3 | B0 | 16 | 8.687 | 13.965 | 0.002041 | 0 | True |
| P2000 | M3 | B1 | 16 | 6.243 | 14.408 | 0.004857 | 0 | True |
| F2000 | M1 | B0 | 16 | 5.626 | 12.302 | 0.008985 | 0 | True |
| F2000 | M1 | B1 | 16 | 7.871 | 15.355 | 0.009103 | 0 | True |
| F2000 | M3 | B0 | 16 | 7.915 | 12.574 | 0.002441 | 0 | True |
| F2000 | M3 | B1 | 16 | 5.257 | 12.951 | 0.004051 | 0 | True |
| X2000 | M1 | B0 | 16 | 5.812 | 13.019 | 0.009003 | 0 | True |
| X2000 | M1 | B1 | 16 | 8.533 | 15.831 | 0.009440 | 0 | True |
| X2000 | M3 | B0 | 16 | 7.258 | 12.821 | 0.002535 | 0 | True |
| X2000 | M3 | B1 | 16 | 4.815 | 12.269 | 0.004118 | 0 | True |

| Point | Task | Wording | Noise | Correct/12 | c0/6 | c1/6 | Reject | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2000 | M1 | SEMANTIC | B0 | 8 | 5 | 3 | 0 | False |
| P2000 | M1 | PARAPHRASE | B0 | 8 | 5 | 3 | 0 | False |
| P2000 | M1 | SEMANTIC | B1 | 8 | 5 | 3 | 0 | False |
| P2000 | M1 | PARAPHRASE | B1 | 10 | 5 | 5 | 0 | True |
| P2000 | M3 | SEMANTIC | B0 | 11 | 6 | 5 | 0 | True |
| P2000 | M3 | PARAPHRASE | B0 | 11 | 6 | 5 | 0 | True |
| P2000 | M3 | SEMANTIC | B1 | 11 | 5 | 6 | 0 | True |
| P2000 | M3 | PARAPHRASE | B1 | 12 | 6 | 6 | 0 | True |
| F2000 | M1 | SEMANTIC | B0 | 9 | 4 | 5 | 0 | False |
| F2000 | M1 | PARAPHRASE | B0 | 9 | 5 | 4 | 0 | False |
| F2000 | M1 | SEMANTIC | B1 | 9 | 4 | 5 | 0 | False |
| F2000 | M1 | PARAPHRASE | B1 | 10 | 4 | 6 | 0 | True |
| F2000 | M3 | SEMANTIC | B0 | 11 | 6 | 5 | 0 | True |
| F2000 | M3 | PARAPHRASE | B0 | 11 | 6 | 5 | 0 | True |
| F2000 | M3 | SEMANTIC | B1 | 11 | 5 | 6 | 0 | True |
| F2000 | M3 | PARAPHRASE | B1 | 12 | 6 | 6 | 0 | True |
| X2000 | M1 | SEMANTIC | B0 | 11 | 6 | 5 | 0 | True |
| X2000 | M1 | PARAPHRASE | B0 | 11 | 6 | 5 | 0 | True |
| X2000 | M1 | SEMANTIC | B1 | 8 | 3 | 5 | 0 | False |
| X2000 | M1 | PARAPHRASE | B1 | 11 | 5 | 6 | 0 | True |
| X2000 | M3 | SEMANTIC | B0 | 11 | 6 | 5 | 0 | True |
| X2000 | M3 | PARAPHRASE | B0 | 11 | 6 | 5 | 0 | True |
| X2000 | M3 | SEMANTIC | B1 | 11 | 5 | 6 | 0 | True |
| X2000 | M3 | PARAPHRASE | B1 | 11 | 6 | 5 | 0 | True |

## Complete crossed M1 results

KIN-VAL: all eight template/noise groups must independently achieve ≥17/20 plus the numeric/zero-REJECT gates. TRAIN values are diagnostics. Correct c0/c1 and both-correct family counts remain separate.

| Point | Split | Template | Noise | Correct/N | c0 | c1 | Both families | median XYZ mm | p95 XYZ mm | median SO3 rad | Reject | Numeric pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2000 | TRAIN | T0 | B0 | 55/60 | 30 | 25 | 25 | 3.973 | 9.630 | 0.005612 | 0 | True |
| P2000 | TRAIN | T0 | B1 | 57/60 | 30 | 27 | 27 | 2.301 | 6.190 | 0.003467 | 0 | True |
| P2000 | TRAIN | T1 | B0 | 44/60 | 22 | 22 | 14 | 4.046 | 12.090 | 0.005992 | 0 | True |
| P2000 | TRAIN | T1 | B1 | 49/60 | 21 | 28 | 19 | 3.113 | 12.091 | 0.005213 | 0 | True |
| P2000 | TRAIN | V0 | B0 | 53/60 | 27 | 26 | 24 | 4.503 | 7.640 | 0.007606 | 0 | True |
| P2000 | TRAIN | V0 | B1 | 52/60 | 22 | 30 | 22 | 4.354 | 8.465 | 0.008250 | 0 | True |
| P2000 | TRAIN | V1 | B0 | 33/60 | 30 | 3 | 3 | 6.270 | 12.988 | 0.008992 | 0 | True |
| P2000 | TRAIN | V1 | B1 | 36/60 | 30 | 6 | 6 | 4.336 | 10.124 | 0.006295 | 0 | True |
| P2000 | KIN-VAL | T0 | B0 | 17/20 | 10 | 7 | 7 | 4.774 | 10.119 | 0.007406 | 0 | True |
| P2000 | KIN-VAL | T0 | B1 | 14/20 | 6 | 8 | 5 | 4.539 | 7.480 | 0.007182 | 0 | True |
| P2000 | KIN-VAL | T1 | B0 | 12/20 | 6 | 6 | 3 | 5.267 | 12.138 | 0.013262 | 0 | True |
| P2000 | KIN-VAL | T1 | B1 | 14/20 | 5 | 9 | 5 | 5.148 | 13.478 | 0.011953 | 0 | True |
| P2000 | KIN-VAL | V0 | B0 | 14/20 | 7 | 7 | 5 | 5.038 | 9.047 | 0.010251 | 0 | True |
| P2000 | KIN-VAL | V0 | B1 | 16/20 | 6 | 10 | 6 | 5.378 | 10.312 | 0.011317 | 0 | True |
| P2000 | KIN-VAL | V1 | B0 | 11/20 | 10 | 1 | 1 | 5.385 | 13.138 | 0.009951 | 0 | True |
| P2000 | KIN-VAL | V1 | B1 | 12/20 | 10 | 2 | 2 | 5.123 | 10.061 | 0.008283 | 0 | True |
| F2000 | TRAIN | T0 | B0 | 57/60 | 30 | 27 | 27 | 3.278 | 6.090 | 0.004566 | 0 | True |
| F2000 | TRAIN | T0 | B1 | 58/60 | 30 | 28 | 28 | 2.132 | 3.740 | 0.003232 | 0 | True |
| F2000 | TRAIN | T1 | B0 | 48/60 | 21 | 27 | 18 | 3.453 | 11.852 | 0.003695 | 0 | True |
| F2000 | TRAIN | T1 | B1 | 47/60 | 18 | 29 | 17 | 2.839 | 13.497 | 0.004706 | 0 | True |
| F2000 | TRAIN | V0 | B0 | 53/60 | 23 | 30 | 23 | 3.362 | 7.554 | 0.006311 | 0 | True |
| F2000 | TRAIN | V0 | B1 | 49/60 | 19 | 30 | 19 | 3.567 | 9.993 | 0.008005 | 0 | True |
| F2000 | TRAIN | V1 | B0 | 42/60 | 30 | 12 | 12 | 4.905 | 11.468 | 0.005626 | 0 | True |
| F2000 | TRAIN | V1 | B1 | 50/60 | 30 | 20 | 20 | 3.481 | 8.050 | 0.007455 | 0 | True |
| F2000 | KIN-VAL | T0 | B0 | 18/20 | 10 | 8 | 8 | 3.299 | 10.935 | 0.003121 | 0 | True |
| F2000 | KIN-VAL | T0 | B1 | 18/20 | 10 | 8 | 8 | 3.206 | 6.564 | 0.005195 | 0 | True |
| F2000 | KIN-VAL | T1 | B0 | 15/20 | 7 | 8 | 5 | 3.955 | 12.551 | 0.005959 | 0 | True |
| F2000 | KIN-VAL | T1 | B1 | 16/20 | 6 | 10 | 6 | 3.437 | 15.185 | 0.006953 | 0 | True |
| F2000 | KIN-VAL | V0 | B0 | 15/20 | 6 | 9 | 6 | 3.959 | 7.869 | 0.008879 | 0 | True |
| F2000 | KIN-VAL | V0 | B1 | 15/20 | 5 | 10 | 5 | 4.691 | 10.432 | 0.010540 | 0 | True |
| F2000 | KIN-VAL | V1 | B0 | 15/20 | 10 | 5 | 5 | 4.424 | 12.274 | 0.007133 | 0 | True |
| F2000 | KIN-VAL | V1 | B1 | 16/20 | 9 | 7 | 7 | 3.681 | 8.778 | 0.007998 | 0 | True |
| X2000 | TRAIN | T0 | B0 | 60/60 | 30 | 30 | 30 | 3.259 | 4.572 | 0.003955 | 0 | True |
| X2000 | TRAIN | T0 | B1 | 60/60 | 30 | 30 | 30 | 2.200 | 3.689 | 0.003264 | 0 | True |
| X2000 | TRAIN | T1 | B0 | 60/60 | 30 | 30 | 30 | 3.419 | 4.950 | 0.003775 | 0 | True |
| X2000 | TRAIN | T1 | B1 | 58/60 | 28 | 30 | 28 | 2.029 | 4.750 | 0.003407 | 0 | True |
| X2000 | TRAIN | V0 | B0 | 57/60 | 27 | 30 | 27 | 3.188 | 6.531 | 0.005647 | 0 | True |
| X2000 | TRAIN | V0 | B1 | 50/60 | 20 | 30 | 20 | 3.449 | 9.801 | 0.007292 | 0 | True |
| X2000 | TRAIN | V1 | B0 | 40/60 | 30 | 10 | 10 | 4.738 | 10.166 | 0.005864 | 0 | True |
| X2000 | TRAIN | V1 | B1 | 56/60 | 30 | 26 | 26 | 3.588 | 6.747 | 0.007481 | 0 | True |
| X2000 | KIN-VAL | T0 | B0 | 19/20 | 10 | 9 | 9 | 3.489 | 4.230 | 0.003379 | 0 | True |
| X2000 | KIN-VAL | T0 | B1 | 19/20 | 10 | 9 | 9 | 3.389 | 5.479 | 0.005184 | 0 | True |
| X2000 | KIN-VAL | T1 | B0 | 16/20 | 8 | 8 | 8 | 3.802 | 11.194 | 0.004694 | 0 | True |
| X2000 | KIN-VAL | T1 | B1 | 17/20 | 8 | 9 | 8 | 3.640 | 13.025 | 0.006090 | 0 | True |
| X2000 | KIN-VAL | V0 | B0 | 18/20 | 9 | 9 | 8 | 3.923 | 7.813 | 0.007976 | 0 | True |
| X2000 | KIN-VAL | V0 | B1 | 14/20 | 4 | 10 | 4 | 4.816 | 9.569 | 0.012176 | 0 | True |
| X2000 | KIN-VAL | V1 | B0 | 16/20 | 10 | 6 | 6 | 3.924 | 11.451 | 0.006103 | 0 | True |
| X2000 | KIN-VAL | V1 | B1 | 17/20 | 9 | 8 | 7 | 3.623 | 7.689 | 0.007817 | 0 | True |

X V0/B1 KIN-VAL is c0=4/10, c1=10/10, both=4/10; V1/B0 is c0=10/10, c1=6/10, both=6/10. TRAIN V1/B0 is c0=30/30, c1=10/30. Pooling commands or choosing a favorable noise would hide these failures.

## Original versus other TRAIN bindings

| Point | Binding | Noise | N | Correct | c0 | c1 | Reject |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2000 | V0 | B0 | 60 | 53 | 23 | 30 | 0 |
| F2000 | V0 | B1 | 60 | 49 | 19 | 30 | 0 |
| F2000 | V1 | B0 | 60 | 42 | 30 | 12 | 0 |
| F2000 | V1 | B1 | 60 | 50 | 30 | 20 | 0 |
| F2000 | original | B0 | 60 | 60 | 30 | 30 | 0 |
| F2000 | original | B1 | 60 | 60 | 30 | 30 | 0 |
| F2000 | other_T | B0 | 60 | 45 | 21 | 24 | 0 |
| F2000 | other_T | B1 | 60 | 45 | 18 | 27 | 0 |
| P2000 | V0 | B0 | 60 | 53 | 27 | 26 | 0 |
| P2000 | V0 | B1 | 60 | 52 | 22 | 30 | 0 |
| P2000 | V1 | B0 | 60 | 33 | 30 | 3 | 0 |
| P2000 | V1 | B1 | 60 | 36 | 30 | 6 | 0 |
| P2000 | original | B0 | 60 | 59 | 30 | 29 | 0 |
| P2000 | original | B1 | 60 | 60 | 30 | 30 | 0 |
| P2000 | other_T | B0 | 60 | 40 | 22 | 18 | 0 |
| P2000 | other_T | B1 | 60 | 46 | 21 | 25 | 0 |
| X2000 | V0 | B0 | 60 | 57 | 27 | 30 | 0 |
| X2000 | V0 | B1 | 60 | 50 | 20 | 30 | 0 |
| X2000 | V1 | B0 | 60 | 40 | 30 | 10 | 0 |
| X2000 | V1 | B1 | 60 | 56 | 30 | 26 | 0 |
| X2000 | original | B0 | 60 | 60 | 30 | 30 | 0 |
| X2000 | original | B1 | 60 | 60 | 30 | 30 | 0 |
| X2000 | other_T | B0 | 60 | 60 | 30 | 30 | 0 |
| X2000 | other_T | B1 | 60 | 58 | 28 | 30 | 0 |

## Direction, bias, wording flips and noise boundaries

Using the existing normalized signed target error s0/s1, g=(s1−s0)/2 and b=(s0+s1)/2. Both commands correct iff s0<0<s1, equivalently g>|b|. Positive ordering with one wrong target is not semantic success. These are output-space descriptions, not proven causal neural mechanisms.

| Point | Split | Template | Noise | Families | Both correct | Positive ordering/wrong | Nonpositive ordering | median g | median b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2000 | TRAIN | T0 | B0 | 30 | 25 | 5 | 0 | 0.85749 | -0.38155 |
| P2000 | TRAIN | T0 | B1 | 30 | 27 | 3 | 0 | 0.81318 | 0.06860 |
| P2000 | TRAIN | T1 | B0 | 30 | 14 | 13 | 3 | 0.57952 | -0.18505 |
| P2000 | TRAIN | T1 | B1 | 30 | 19 | 8 | 3 | 0.59451 | 0.30970 |
| P2000 | TRAIN | V0 | B0 | 30 | 24 | 5 | 1 | 0.57215 | -0.16604 |
| P2000 | TRAIN | V0 | B1 | 30 | 22 | 7 | 1 | 0.53525 | 0.23021 |
| P2000 | TRAIN | V1 | B0 | 30 | 3 | 26 | 1 | 0.36661 | -0.91952 |
| P2000 | TRAIN | V1 | B1 | 30 | 6 | 23 | 1 | 0.34190 | -0.49284 |
| P2000 | KIN-VAL | T0 | B0 | 10 | 7 | 1 | 2 | 0.63126 | -0.31239 |
| P2000 | KIN-VAL | T0 | B1 | 10 | 5 | 3 | 2 | 0.58273 | 0.11581 |
| P2000 | KIN-VAL | T1 | B0 | 10 | 3 | 5 | 2 | 0.50014 | -0.11249 |
| P2000 | KIN-VAL | T1 | B1 | 10 | 5 | 3 | 2 | 0.49527 | 0.48104 |
| P2000 | KIN-VAL | V0 | B0 | 10 | 5 | 3 | 2 | 0.42696 | -0.08172 |
| P2000 | KIN-VAL | V0 | B1 | 10 | 6 | 2 | 2 | 0.42881 | 0.29499 |
| P2000 | KIN-VAL | V1 | B0 | 10 | 1 | 7 | 2 | 0.31317 | -0.84192 |
| P2000 | KIN-VAL | V1 | B1 | 10 | 2 | 6 | 2 | 0.28539 | -0.40996 |
| F2000 | TRAIN | T0 | B0 | 30 | 27 | 3 | 0 | 0.97936 | -0.34494 |
| F2000 | TRAIN | T0 | B1 | 30 | 28 | 2 | 0 | 0.89601 | 0.11636 |
| F2000 | TRAIN | T1 | B0 | 30 | 18 | 11 | 1 | 0.76412 | -0.16288 |
| F2000 | TRAIN | T1 | B1 | 30 | 17 | 12 | 1 | 0.66787 | 0.34610 |
| F2000 | TRAIN | V0 | B0 | 30 | 23 | 7 | 0 | 0.65464 | -0.02777 |
| F2000 | TRAIN | V0 | B1 | 30 | 19 | 11 | 0 | 0.57823 | 0.40406 |
| F2000 | TRAIN | V1 | B0 | 30 | 12 | 18 | 0 | 0.58613 | -0.66905 |
| F2000 | TRAIN | V1 | B1 | 30 | 20 | 10 | 0 | 0.52941 | -0.22138 |
| F2000 | KIN-VAL | T0 | B0 | 10 | 8 | 0 | 2 | 1.02062 | -0.22883 |
| F2000 | KIN-VAL | T0 | B1 | 10 | 8 | 1 | 1 | 0.92975 | 0.24990 |
| F2000 | KIN-VAL | T1 | B0 | 10 | 5 | 3 | 2 | 0.75455 | -0.03276 |
| F2000 | KIN-VAL | T1 | B1 | 10 | 6 | 2 | 2 | 0.66562 | 0.51743 |
| F2000 | KIN-VAL | V0 | B0 | 10 | 6 | 3 | 1 | 0.50988 | 0.02558 |
| F2000 | KIN-VAL | V0 | B1 | 10 | 5 | 4 | 1 | 0.46464 | 0.49100 |
| F2000 | KIN-VAL | V1 | B0 | 10 | 5 | 3 | 2 | 0.58789 | -0.58957 |
| F2000 | KIN-VAL | V1 | B1 | 10 | 7 | 1 | 2 | 0.52929 | -0.09200 |
| X2000 | TRAIN | T0 | B0 | 30 | 30 | 0 | 0 | 1.03937 | -0.24480 |
| X2000 | TRAIN | T0 | B1 | 30 | 30 | 0 | 0 | 0.96439 | 0.23449 |
| X2000 | TRAIN | T1 | B0 | 30 | 30 | 0 | 0 | 1.00933 | -0.29413 |
| X2000 | TRAIN | T1 | B1 | 30 | 28 | 2 | 0 | 0.92870 | 0.21426 |
| X2000 | TRAIN | V0 | B0 | 30 | 27 | 3 | 0 | 0.71282 | -0.04024 |
| X2000 | TRAIN | V0 | B1 | 30 | 20 | 10 | 0 | 0.63978 | 0.38963 |
| X2000 | TRAIN | V1 | B0 | 30 | 10 | 20 | 0 | 0.53317 | -0.68560 |
| X2000 | TRAIN | V1 | B1 | 30 | 26 | 4 | 0 | 0.53333 | -0.16622 |
| X2000 | KIN-VAL | T0 | B0 | 10 | 9 | 1 | 0 | 0.97519 | -0.10147 |
| X2000 | KIN-VAL | T0 | B1 | 10 | 9 | 1 | 0 | 0.91254 | 0.42908 |
| X2000 | KIN-VAL | T1 | B0 | 10 | 8 | 0 | 2 | 0.98021 | -0.13771 |
| X2000 | KIN-VAL | T1 | B1 | 10 | 8 | 0 | 2 | 0.87974 | 0.41938 |
| X2000 | KIN-VAL | V0 | B0 | 10 | 8 | 1 | 1 | 0.59220 | 0.02143 |
| X2000 | KIN-VAL | V0 | B1 | 10 | 4 | 5 | 1 | 0.53561 | 0.51353 |
| X2000 | KIN-VAL | V1 | B0 | 10 | 6 | 3 | 1 | 0.55479 | -0.49135 |
| X2000 | KIN-VAL | V1 | B1 | 10 | 7 | 2 | 1 | 0.52505 | 0.00192 |

| Point | Split | Wording change | Noise | Rows | Correct→wrong | Wrong→correct |
| --- | --- | --- | --- | --- | --- | --- |
| P2000 | TRAIN | T0→T1 | B0 | 60 | 16 | 5 |
| P2000 | TRAIN | T0→T1 | B1 | 60 | 11 | 3 |
| P2000 | TRAIN | V0→V1 | B0 | 60 | 23 | 3 |
| P2000 | TRAIN | V0→V1 | B1 | 60 | 24 | 8 |
| P2000 | KIN-VAL | T0→T1 | B0 | 20 | 6 | 1 |
| P2000 | KIN-VAL | T0→T1 | B1 | 20 | 2 | 2 |
| P2000 | KIN-VAL | V0→V1 | B0 | 20 | 6 | 3 |
| P2000 | KIN-VAL | V0→V1 | B1 | 20 | 8 | 4 |
| F2000 | TRAIN | T0→T1 | B0 | 60 | 12 | 3 |
| F2000 | TRAIN | T0→T1 | B1 | 60 | 13 | 2 |
| F2000 | TRAIN | V0→V1 | B0 | 60 | 18 | 7 |
| F2000 | TRAIN | V0→V1 | B1 | 60 | 10 | 11 |
| F2000 | KIN-VAL | T0→T1 | B0 | 20 | 5 | 2 |
| F2000 | KIN-VAL | T0→T1 | B1 | 20 | 4 | 2 |
| F2000 | KIN-VAL | V0→V1 | B0 | 20 | 4 | 4 |
| F2000 | KIN-VAL | V0→V1 | B1 | 20 | 3 | 4 |
| X2000 | TRAIN | T0→T1 | B0 | 60 | 0 | 0 |
| X2000 | TRAIN | T0→T1 | B1 | 60 | 2 | 0 |
| X2000 | TRAIN | V0→V1 | B0 | 60 | 20 | 3 |
| X2000 | TRAIN | V0→V1 | B1 | 60 | 4 | 10 |
| X2000 | KIN-VAL | T0→T1 | B0 | 20 | 3 | 0 |
| X2000 | KIN-VAL | T0→T1 | B1 | 20 | 2 | 0 |
| X2000 | KIN-VAL | V0→V1 | B0 | 20 | 3 | 1 |
| X2000 | KIN-VAL | V0→V1 | B1 | 20 | 2 | 5 |

All six predefined template-pair comparisons, per-family signed margins and other wording flips are preserved in the raw CSVs.

| Point | Split | Template | Families | c0 B0/B1 crossing | c1 B0/B1 crossing |
| --- | --- | --- | --- | --- | --- |
| P2000 | TRAIN | T0 | 30 | 0 | 2 |
| P2000 | TRAIN | T1 | 30 | 1 | 6 |
| P2000 | TRAIN | V0 | 30 | 5 | 4 |
| P2000 | TRAIN | V1 | 30 | 0 | 3 |
| P2000 | KIN-VAL | T0 | 10 | 4 | 1 |
| P2000 | KIN-VAL | T1 | 10 | 1 | 3 |
| P2000 | KIN-VAL | V0 | 10 | 1 | 3 |
| P2000 | KIN-VAL | V1 | 10 | 0 | 1 |
| F2000 | TRAIN | T0 | 30 | 0 | 1 |
| F2000 | TRAIN | T1 | 30 | 3 | 2 |
| F2000 | TRAIN | V0 | 30 | 4 | 0 |
| F2000 | TRAIN | V1 | 30 | 0 | 8 |
| F2000 | KIN-VAL | T0 | 10 | 0 | 0 |
| F2000 | KIN-VAL | T1 | 10 | 1 | 2 |
| F2000 | KIN-VAL | V0 | 10 | 1 | 1 |
| F2000 | KIN-VAL | V1 | 10 | 1 | 2 |
| X2000 | TRAIN | T0 | 30 | 0 | 0 |
| X2000 | TRAIN | T1 | 30 | 2 | 0 |
| X2000 | TRAIN | V0 | 30 | 7 | 0 |
| X2000 | TRAIN | V1 | 30 | 0 | 16 |
| X2000 | KIN-VAL | T0 | 10 | 0 | 0 |
| X2000 | KIN-VAL | T1 | 10 | 0 | 1 |
| X2000 | KIN-VAL | V0 | 10 | 5 | 1 |
| X2000 | KIN-VAL | V1 | 10 | 1 | 2 |

### Every adverse X KIN-VAL family/template/noise outcome

| Family | Template | Noise | Category | s0 | s1 | g | b |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pi0260-family-120 | V0 | B1 | positive_ordering_bias_dominates | 0.022888 | 1.114181 | 0.545646 | 0.568535 |
| pi0260-family-122 | V0 | B1 | positive_ordering_bias_dominates | 0.379750 | 1.371184 | 0.495717 | 0.875467 |
| pi0260-family-141 | V0 | B1 | positive_ordering_bias_dominates | 0.453791 | 1.578462 | 0.562335 | 1.016126 |
| pi0260-family-142 | T0 | B0 | positive_ordering_bias_dominates | -0.785288 | -0.650394 | 0.067447 | -0.717841 |
| pi0260-family-142 | T0 | B1 | positive_ordering_bias_dominates | -0.202242 | -0.013100 | 0.094571 | -0.107671 |
| pi0260-family-142 | T1 | B0 | nonpositive_ordering | 0.654898 | -0.892615 | -0.773756 | -0.118859 |
| pi0260-family-142 | T1 | B1 | nonpositive_ordering | 1.132526 | -0.226362 | -0.679444 | 0.453082 |
| pi0260-family-142 | V0 | B0 | nonpositive_ordering | -0.146103 | -0.180128 | -0.017013 | -0.163116 |
| pi0260-family-142 | V0 | B1 | nonpositive_ordering | 0.400891 | 0.370018 | -0.015437 | 0.385455 |
| pi0260-family-142 | V1 | B0 | nonpositive_ordering | -0.583575 | -0.841902 | -0.129164 | -0.712738 |
| pi0260-family-142 | V1 | B1 | nonpositive_ordering | -0.002484 | -0.269499 | -0.133508 | -0.135991 |
| pi0260-family-147 | V1 | B0 | positive_ordering_bias_dominates | -0.989517 | -0.054681 | 0.467418 | -0.522099 |
| pi0260-family-148 | T1 | B0 | nonpositive_ordering | 0.518050 | -0.740067 | -0.629058 | -0.111008 |
| pi0260-family-148 | T1 | B1 | nonpositive_ordering | 1.073960 | 0.110491 | -0.481734 | 0.592225 |
| pi0260-family-148 | V0 | B1 | positive_ordering_bias_dominates | 0.356010 | 0.711843 | 0.177916 | 0.533926 |
| pi0260-family-148 | V1 | B0 | positive_ordering_bias_dominates | -0.903564 | -0.802423 | 0.050571 | -0.852993 |
| pi0260-family-148 | V1 | B1 | positive_ordering_bias_dominates | -0.335697 | -0.201432 | 0.067133 | -0.268565 |
| pi0260-family-156 | V1 | B0 | positive_ordering_bias_dominates | -1.346847 | -0.431298 | 0.457775 | -0.889072 |
| pi0260-family-157 | V0 | B0 | positive_ordering_bias_dominates | 0.800463 | 0.995619 | 0.097578 | 0.898041 |
| pi0260-family-157 | V0 | B1 | positive_ordering_bias_dominates | 1.155717 | 1.347712 | 0.095998 | 1.251715 |
| pi0260-family-157 | V1 | B1 | positive_ordering_bias_dominates | 0.074862 | 0.758042 | 0.341590 | 0.416452 |

## Color and command strata — complete denominators

Command0 is not a fixed color. The underlying registry controls requested colors; original unequal color/family frequencies are retained.

| Point | Split | Template | Color | Command | Noise | N | Correct | Reject |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F2000 | KIN-VAL | T0 | blue | 0 | B0 | 3 | 3 | 0 |
| F2000 | KIN-VAL | T0 | blue | 0 | B1 | 3 | 3 | 0 |
| F2000 | KIN-VAL | T0 | blue | 1 | B0 | 5 | 5 | 0 |
| F2000 | KIN-VAL | T0 | blue | 1 | B1 | 5 | 5 | 0 |
| F2000 | KIN-VAL | T0 | green | 0 | B0 | 2 | 2 | 0 |
| F2000 | KIN-VAL | T0 | green | 0 | B1 | 2 | 2 | 0 |
| F2000 | KIN-VAL | T0 | green | 1 | B0 | 3 | 1 | 0 |
| F2000 | KIN-VAL | T0 | green | 1 | B1 | 3 | 1 | 0 |
| F2000 | KIN-VAL | T0 | red | 0 | B0 | 5 | 5 | 0 |
| F2000 | KIN-VAL | T0 | red | 0 | B1 | 5 | 5 | 0 |
| F2000 | KIN-VAL | T0 | red | 1 | B0 | 2 | 2 | 0 |
| F2000 | KIN-VAL | T0 | red | 1 | B1 | 2 | 2 | 0 |
| F2000 | KIN-VAL | T1 | blue | 0 | B0 | 3 | 0 | 0 |
| F2000 | KIN-VAL | T1 | blue | 0 | B1 | 3 | 0 | 0 |
| F2000 | KIN-VAL | T1 | blue | 1 | B0 | 5 | 5 | 0 |
| F2000 | KIN-VAL | T1 | blue | 1 | B1 | 5 | 5 | 0 |
| F2000 | KIN-VAL | T1 | green | 0 | B0 | 2 | 2 | 0 |
| F2000 | KIN-VAL | T1 | green | 0 | B1 | 2 | 2 | 0 |
| F2000 | KIN-VAL | T1 | green | 1 | B0 | 3 | 3 | 0 |
| F2000 | KIN-VAL | T1 | green | 1 | B1 | 3 | 3 | 0 |
| F2000 | KIN-VAL | T1 | red | 0 | B0 | 5 | 5 | 0 |
| F2000 | KIN-VAL | T1 | red | 0 | B1 | 5 | 4 | 0 |
| F2000 | KIN-VAL | T1 | red | 1 | B0 | 2 | 0 | 0 |
| F2000 | KIN-VAL | T1 | red | 1 | B1 | 2 | 2 | 0 |
| F2000 | KIN-VAL | V0 | blue | 0 | B0 | 3 | 1 | 0 |
| F2000 | KIN-VAL | V0 | blue | 0 | B1 | 3 | 0 | 0 |
| F2000 | KIN-VAL | V0 | blue | 1 | B0 | 5 | 5 | 0 |
| F2000 | KIN-VAL | V0 | blue | 1 | B1 | 5 | 5 | 0 |
| F2000 | KIN-VAL | V0 | green | 0 | B0 | 2 | 1 | 0 |
| F2000 | KIN-VAL | V0 | green | 0 | B1 | 2 | 1 | 0 |
| F2000 | KIN-VAL | V0 | green | 1 | B0 | 3 | 2 | 0 |
| F2000 | KIN-VAL | V0 | green | 1 | B1 | 3 | 3 | 0 |
| F2000 | KIN-VAL | V0 | red | 0 | B0 | 5 | 4 | 0 |
| F2000 | KIN-VAL | V0 | red | 0 | B1 | 5 | 4 | 0 |
| F2000 | KIN-VAL | V0 | red | 1 | B0 | 2 | 2 | 0 |
| F2000 | KIN-VAL | V0 | red | 1 | B1 | 2 | 2 | 0 |
| F2000 | KIN-VAL | V1 | blue | 0 | B0 | 3 | 3 | 0 |
| F2000 | KIN-VAL | V1 | blue | 0 | B1 | 3 | 2 | 0 |
| F2000 | KIN-VAL | V1 | blue | 1 | B0 | 5 | 3 | 0 |
| F2000 | KIN-VAL | V1 | blue | 1 | B1 | 5 | 4 | 0 |
| F2000 | KIN-VAL | V1 | green | 0 | B0 | 2 | 2 | 0 |
| F2000 | KIN-VAL | V1 | green | 0 | B1 | 2 | 2 | 0 |
| F2000 | KIN-VAL | V1 | green | 1 | B0 | 3 | 1 | 0 |
| F2000 | KIN-VAL | V1 | green | 1 | B1 | 3 | 1 | 0 |
| F2000 | KIN-VAL | V1 | red | 0 | B0 | 5 | 5 | 0 |
| F2000 | KIN-VAL | V1 | red | 0 | B1 | 5 | 5 | 0 |
| F2000 | KIN-VAL | V1 | red | 1 | B0 | 2 | 1 | 0 |
| F2000 | KIN-VAL | V1 | red | 1 | B1 | 2 | 2 | 0 |
| F2000 | TRAIN | T0 | blue | 0 | B0 | 5 | 5 | 0 |
| F2000 | TRAIN | T0 | blue | 0 | B1 | 5 | 5 | 0 |
| F2000 | TRAIN | T0 | blue | 1 | B0 | 11 | 11 | 0 |
| F2000 | TRAIN | T0 | blue | 1 | B1 | 11 | 11 | 0 |
| F2000 | TRAIN | T0 | green | 0 | B0 | 14 | 14 | 0 |
| F2000 | TRAIN | T0 | green | 0 | B1 | 14 | 14 | 0 |
| F2000 | TRAIN | T0 | green | 1 | B0 | 5 | 2 | 0 |
| F2000 | TRAIN | T0 | green | 1 | B1 | 5 | 3 | 0 |
| F2000 | TRAIN | T0 | red | 0 | B0 | 11 | 11 | 0 |
| F2000 | TRAIN | T0 | red | 0 | B1 | 11 | 11 | 0 |
| F2000 | TRAIN | T0 | red | 1 | B0 | 14 | 14 | 0 |
| F2000 | TRAIN | T0 | red | 1 | B1 | 14 | 14 | 0 |
| F2000 | TRAIN | T1 | blue | 0 | B0 | 5 | 1 | 0 |
| F2000 | TRAIN | T1 | blue | 0 | B1 | 5 | 1 | 0 |
| F2000 | TRAIN | T1 | blue | 1 | B0 | 11 | 11 | 0 |
| F2000 | TRAIN | T1 | blue | 1 | B1 | 11 | 11 | 0 |
| F2000 | TRAIN | T1 | green | 0 | B0 | 14 | 9 | 0 |
| F2000 | TRAIN | T1 | green | 0 | B1 | 14 | 7 | 0 |
| F2000 | TRAIN | T1 | green | 1 | B0 | 5 | 4 | 0 |
| F2000 | TRAIN | T1 | green | 1 | B1 | 5 | 5 | 0 |
| F2000 | TRAIN | T1 | red | 0 | B0 | 11 | 11 | 0 |
| F2000 | TRAIN | T1 | red | 0 | B1 | 11 | 10 | 0 |
| F2000 | TRAIN | T1 | red | 1 | B0 | 14 | 12 | 0 |
| F2000 | TRAIN | T1 | red | 1 | B1 | 14 | 13 | 0 |
| F2000 | TRAIN | V0 | blue | 0 | B0 | 5 | 0 | 0 |
| F2000 | TRAIN | V0 | blue | 0 | B1 | 5 | 0 | 0 |
| F2000 | TRAIN | V0 | blue | 1 | B0 | 11 | 11 | 0 |
| F2000 | TRAIN | V0 | blue | 1 | B1 | 11 | 11 | 0 |
| F2000 | TRAIN | V0 | green | 0 | B0 | 14 | 12 | 0 |
| F2000 | TRAIN | V0 | green | 0 | B1 | 14 | 11 | 0 |
| F2000 | TRAIN | V0 | green | 1 | B0 | 5 | 5 | 0 |
| F2000 | TRAIN | V0 | green | 1 | B1 | 5 | 5 | 0 |
| F2000 | TRAIN | V0 | red | 0 | B0 | 11 | 11 | 0 |
| F2000 | TRAIN | V0 | red | 0 | B1 | 11 | 8 | 0 |
| F2000 | TRAIN | V0 | red | 1 | B0 | 14 | 14 | 0 |
| F2000 | TRAIN | V0 | red | 1 | B1 | 14 | 14 | 0 |
| F2000 | TRAIN | V1 | blue | 0 | B0 | 5 | 5 | 0 |
| F2000 | TRAIN | V1 | blue | 0 | B1 | 5 | 5 | 0 |
| F2000 | TRAIN | V1 | blue | 1 | B0 | 11 | 5 | 0 |
| F2000 | TRAIN | V1 | blue | 1 | B1 | 11 | 9 | 0 |
| F2000 | TRAIN | V1 | green | 0 | B0 | 14 | 14 | 0 |
| F2000 | TRAIN | V1 | green | 0 | B1 | 14 | 14 | 0 |
| F2000 | TRAIN | V1 | green | 1 | B0 | 5 | 0 | 0 |
| F2000 | TRAIN | V1 | green | 1 | B1 | 5 | 2 | 0 |
| F2000 | TRAIN | V1 | red | 0 | B0 | 11 | 11 | 0 |
| F2000 | TRAIN | V1 | red | 0 | B1 | 11 | 11 | 0 |
| F2000 | TRAIN | V1 | red | 1 | B0 | 14 | 7 | 0 |
| F2000 | TRAIN | V1 | red | 1 | B1 | 14 | 9 | 0 |
| P2000 | KIN-VAL | T0 | blue | 0 | B0 | 3 | 3 | 0 |
| P2000 | KIN-VAL | T0 | blue | 0 | B1 | 3 | 2 | 0 |
| P2000 | KIN-VAL | T0 | blue | 1 | B0 | 5 | 4 | 0 |
| P2000 | KIN-VAL | T0 | blue | 1 | B1 | 5 | 5 | 0 |
| P2000 | KIN-VAL | T0 | green | 0 | B0 | 2 | 2 | 0 |
| P2000 | KIN-VAL | T0 | green | 0 | B1 | 2 | 2 | 0 |
| P2000 | KIN-VAL | T0 | green | 1 | B0 | 3 | 1 | 0 |
| P2000 | KIN-VAL | T0 | green | 1 | B1 | 3 | 1 | 0 |
| P2000 | KIN-VAL | T0 | red | 0 | B0 | 5 | 5 | 0 |
| P2000 | KIN-VAL | T0 | red | 0 | B1 | 5 | 2 | 0 |
| P2000 | KIN-VAL | T0 | red | 1 | B0 | 2 | 2 | 0 |
| P2000 | KIN-VAL | T0 | red | 1 | B1 | 2 | 2 | 0 |
| P2000 | KIN-VAL | T1 | blue | 0 | B0 | 3 | 0 | 0 |
| P2000 | KIN-VAL | T1 | blue | 0 | B1 | 3 | 0 | 0 |
| P2000 | KIN-VAL | T1 | blue | 1 | B0 | 5 | 4 | 0 |
| P2000 | KIN-VAL | T1 | blue | 1 | B1 | 5 | 5 | 0 |
| P2000 | KIN-VAL | T1 | green | 0 | B0 | 2 | 2 | 0 |
| P2000 | KIN-VAL | T1 | green | 0 | B1 | 2 | 2 | 0 |
| P2000 | KIN-VAL | T1 | green | 1 | B0 | 3 | 2 | 0 |
| P2000 | KIN-VAL | T1 | green | 1 | B1 | 3 | 2 | 0 |
| P2000 | KIN-VAL | T1 | red | 0 | B0 | 5 | 4 | 0 |
| P2000 | KIN-VAL | T1 | red | 0 | B1 | 5 | 3 | 0 |
| P2000 | KIN-VAL | T1 | red | 1 | B0 | 2 | 0 | 0 |
| P2000 | KIN-VAL | T1 | red | 1 | B1 | 2 | 2 | 0 |
| P2000 | KIN-VAL | V0 | blue | 0 | B0 | 3 | 1 | 0 |
| P2000 | KIN-VAL | V0 | blue | 0 | B1 | 3 | 0 | 0 |
| P2000 | KIN-VAL | V0 | blue | 1 | B0 | 5 | 4 | 0 |
| P2000 | KIN-VAL | V0 | blue | 1 | B1 | 5 | 5 | 0 |
| P2000 | KIN-VAL | V0 | green | 0 | B0 | 2 | 2 | 0 |
| P2000 | KIN-VAL | V0 | green | 0 | B1 | 2 | 2 | 0 |
| P2000 | KIN-VAL | V0 | green | 1 | B0 | 3 | 1 | 0 |
| P2000 | KIN-VAL | V0 | green | 1 | B1 | 3 | 3 | 0 |
| P2000 | KIN-VAL | V0 | red | 0 | B0 | 5 | 4 | 0 |
| P2000 | KIN-VAL | V0 | red | 0 | B1 | 5 | 4 | 0 |
| P2000 | KIN-VAL | V0 | red | 1 | B0 | 2 | 2 | 0 |
| P2000 | KIN-VAL | V0 | red | 1 | B1 | 2 | 2 | 0 |
| P2000 | KIN-VAL | V1 | blue | 0 | B0 | 3 | 3 | 0 |
| P2000 | KIN-VAL | V1 | blue | 0 | B1 | 3 | 3 | 0 |
| P2000 | KIN-VAL | V1 | blue | 1 | B0 | 5 | 1 | 0 |
| P2000 | KIN-VAL | V1 | blue | 1 | B1 | 5 | 2 | 0 |
| P2000 | KIN-VAL | V1 | green | 0 | B0 | 2 | 2 | 0 |
| P2000 | KIN-VAL | V1 | green | 0 | B1 | 2 | 2 | 0 |
| P2000 | KIN-VAL | V1 | green | 1 | B0 | 3 | 0 | 0 |
| P2000 | KIN-VAL | V1 | green | 1 | B1 | 3 | 0 | 0 |
| P2000 | KIN-VAL | V1 | red | 0 | B0 | 5 | 5 | 0 |
| P2000 | KIN-VAL | V1 | red | 0 | B1 | 5 | 5 | 0 |
| P2000 | KIN-VAL | V1 | red | 1 | B0 | 2 | 0 | 0 |
| P2000 | KIN-VAL | V1 | red | 1 | B1 | 2 | 0 | 0 |
| P2000 | TRAIN | T0 | blue | 0 | B0 | 5 | 5 | 0 |
| P2000 | TRAIN | T0 | blue | 0 | B1 | 5 | 5 | 0 |
| P2000 | TRAIN | T0 | blue | 1 | B0 | 11 | 11 | 0 |
| P2000 | TRAIN | T0 | blue | 1 | B1 | 11 | 11 | 0 |
| P2000 | TRAIN | T0 | green | 0 | B0 | 14 | 14 | 0 |
| P2000 | TRAIN | T0 | green | 0 | B1 | 14 | 14 | 0 |
| P2000 | TRAIN | T0 | green | 1 | B0 | 5 | 1 | 0 |
| P2000 | TRAIN | T0 | green | 1 | B1 | 5 | 2 | 0 |
| P2000 | TRAIN | T0 | red | 0 | B0 | 11 | 11 | 0 |
| P2000 | TRAIN | T0 | red | 0 | B1 | 11 | 11 | 0 |
| P2000 | TRAIN | T0 | red | 1 | B0 | 14 | 13 | 0 |
| P2000 | TRAIN | T0 | red | 1 | B1 | 14 | 14 | 0 |
| P2000 | TRAIN | T1 | blue | 0 | B0 | 5 | 1 | 0 |
| P2000 | TRAIN | T1 | blue | 0 | B1 | 5 | 1 | 0 |
| P2000 | TRAIN | T1 | blue | 1 | B0 | 11 | 8 | 0 |
| P2000 | TRAIN | T1 | blue | 1 | B1 | 11 | 11 | 0 |
| P2000 | TRAIN | T1 | green | 0 | B0 | 14 | 10 | 0 |
| P2000 | TRAIN | T1 | green | 0 | B1 | 14 | 9 | 0 |
| P2000 | TRAIN | T1 | green | 1 | B0 | 5 | 4 | 0 |
| P2000 | TRAIN | T1 | green | 1 | B1 | 5 | 4 | 0 |
| P2000 | TRAIN | T1 | red | 0 | B0 | 11 | 11 | 0 |
| P2000 | TRAIN | T1 | red | 0 | B1 | 11 | 11 | 0 |
| P2000 | TRAIN | T1 | red | 1 | B0 | 14 | 10 | 0 |
| P2000 | TRAIN | T1 | red | 1 | B1 | 14 | 13 | 0 |
| P2000 | TRAIN | V0 | blue | 0 | B0 | 5 | 2 | 0 |
| P2000 | TRAIN | V0 | blue | 0 | B1 | 5 | 0 | 0 |
| P2000 | TRAIN | V0 | blue | 1 | B0 | 11 | 10 | 0 |
| P2000 | TRAIN | V0 | blue | 1 | B1 | 11 | 11 | 0 |
| P2000 | TRAIN | V0 | green | 0 | B0 | 14 | 14 | 0 |
| P2000 | TRAIN | V0 | green | 0 | B1 | 14 | 12 | 0 |
| P2000 | TRAIN | V0 | green | 1 | B0 | 5 | 3 | 0 |
| P2000 | TRAIN | V0 | green | 1 | B1 | 5 | 5 | 0 |
| P2000 | TRAIN | V0 | red | 0 | B0 | 11 | 11 | 0 |
| P2000 | TRAIN | V0 | red | 0 | B1 | 11 | 10 | 0 |
| P2000 | TRAIN | V0 | red | 1 | B0 | 14 | 13 | 0 |
| P2000 | TRAIN | V0 | red | 1 | B1 | 14 | 14 | 0 |
| P2000 | TRAIN | V1 | blue | 0 | B0 | 5 | 5 | 0 |
| P2000 | TRAIN | V1 | blue | 0 | B1 | 5 | 5 | 0 |
| P2000 | TRAIN | V1 | blue | 1 | B0 | 11 | 0 | 0 |
| P2000 | TRAIN | V1 | blue | 1 | B1 | 11 | 0 | 0 |
| P2000 | TRAIN | V1 | green | 0 | B0 | 14 | 14 | 0 |
| P2000 | TRAIN | V1 | green | 0 | B1 | 14 | 14 | 0 |
| P2000 | TRAIN | V1 | green | 1 | B0 | 5 | 0 | 0 |
| P2000 | TRAIN | V1 | green | 1 | B1 | 5 | 0 | 0 |
| P2000 | TRAIN | V1 | red | 0 | B0 | 11 | 11 | 0 |
| P2000 | TRAIN | V1 | red | 0 | B1 | 11 | 11 | 0 |
| P2000 | TRAIN | V1 | red | 1 | B0 | 14 | 3 | 0 |
| P2000 | TRAIN | V1 | red | 1 | B1 | 14 | 6 | 0 |
| X2000 | KIN-VAL | T0 | blue | 0 | B0 | 3 | 3 | 0 |
| X2000 | KIN-VAL | T0 | blue | 0 | B1 | 3 | 3 | 0 |
| X2000 | KIN-VAL | T0 | blue | 1 | B0 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T0 | blue | 1 | B1 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T0 | green | 0 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | T0 | green | 0 | B1 | 2 | 2 | 0 |
| X2000 | KIN-VAL | T0 | green | 1 | B0 | 3 | 2 | 0 |
| X2000 | KIN-VAL | T0 | green | 1 | B1 | 3 | 2 | 0 |
| X2000 | KIN-VAL | T0 | red | 0 | B0 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T0 | red | 0 | B1 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T0 | red | 1 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | T0 | red | 1 | B1 | 2 | 2 | 0 |
| X2000 | KIN-VAL | T1 | blue | 0 | B0 | 3 | 1 | 0 |
| X2000 | KIN-VAL | T1 | blue | 0 | B1 | 3 | 1 | 0 |
| X2000 | KIN-VAL | T1 | blue | 1 | B0 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T1 | blue | 1 | B1 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T1 | green | 0 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | T1 | green | 0 | B1 | 2 | 2 | 0 |
| X2000 | KIN-VAL | T1 | green | 1 | B0 | 3 | 1 | 0 |
| X2000 | KIN-VAL | T1 | green | 1 | B1 | 3 | 2 | 0 |
| X2000 | KIN-VAL | T1 | red | 0 | B0 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T1 | red | 0 | B1 | 5 | 5 | 0 |
| X2000 | KIN-VAL | T1 | red | 1 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | T1 | red | 1 | B1 | 2 | 2 | 0 |
| X2000 | KIN-VAL | V0 | blue | 0 | B0 | 3 | 2 | 0 |
| X2000 | KIN-VAL | V0 | blue | 0 | B1 | 3 | 0 | 0 |
| X2000 | KIN-VAL | V0 | blue | 1 | B0 | 5 | 5 | 0 |
| X2000 | KIN-VAL | V0 | blue | 1 | B1 | 5 | 5 | 0 |
| X2000 | KIN-VAL | V0 | green | 0 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | V0 | green | 0 | B1 | 2 | 1 | 0 |
| X2000 | KIN-VAL | V0 | green | 1 | B0 | 3 | 2 | 0 |
| X2000 | KIN-VAL | V0 | green | 1 | B1 | 3 | 3 | 0 |
| X2000 | KIN-VAL | V0 | red | 0 | B0 | 5 | 5 | 0 |
| X2000 | KIN-VAL | V0 | red | 0 | B1 | 5 | 3 | 0 |
| X2000 | KIN-VAL | V0 | red | 1 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | V0 | red | 1 | B1 | 2 | 2 | 0 |
| X2000 | KIN-VAL | V1 | blue | 0 | B0 | 3 | 3 | 0 |
| X2000 | KIN-VAL | V1 | blue | 0 | B1 | 3 | 2 | 0 |
| X2000 | KIN-VAL | V1 | blue | 1 | B0 | 5 | 3 | 0 |
| X2000 | KIN-VAL | V1 | blue | 1 | B1 | 5 | 5 | 0 |
| X2000 | KIN-VAL | V1 | green | 0 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | V1 | green | 0 | B1 | 2 | 2 | 0 |
| X2000 | KIN-VAL | V1 | green | 1 | B0 | 3 | 1 | 0 |
| X2000 | KIN-VAL | V1 | green | 1 | B1 | 3 | 1 | 0 |
| X2000 | KIN-VAL | V1 | red | 0 | B0 | 5 | 5 | 0 |
| X2000 | KIN-VAL | V1 | red | 0 | B1 | 5 | 5 | 0 |
| X2000 | KIN-VAL | V1 | red | 1 | B0 | 2 | 2 | 0 |
| X2000 | KIN-VAL | V1 | red | 1 | B1 | 2 | 2 | 0 |
| X2000 | TRAIN | T0 | blue | 0 | B0 | 5 | 5 | 0 |
| X2000 | TRAIN | T0 | blue | 0 | B1 | 5 | 5 | 0 |
| X2000 | TRAIN | T0 | blue | 1 | B0 | 11 | 11 | 0 |
| X2000 | TRAIN | T0 | blue | 1 | B1 | 11 | 11 | 0 |
| X2000 | TRAIN | T0 | green | 0 | B0 | 14 | 14 | 0 |
| X2000 | TRAIN | T0 | green | 0 | B1 | 14 | 14 | 0 |
| X2000 | TRAIN | T0 | green | 1 | B0 | 5 | 5 | 0 |
| X2000 | TRAIN | T0 | green | 1 | B1 | 5 | 5 | 0 |
| X2000 | TRAIN | T0 | red | 0 | B0 | 11 | 11 | 0 |
| X2000 | TRAIN | T0 | red | 0 | B1 | 11 | 11 | 0 |
| X2000 | TRAIN | T0 | red | 1 | B0 | 14 | 14 | 0 |
| X2000 | TRAIN | T0 | red | 1 | B1 | 14 | 14 | 0 |
| X2000 | TRAIN | T1 | blue | 0 | B0 | 5 | 5 | 0 |
| X2000 | TRAIN | T1 | blue | 0 | B1 | 5 | 3 | 0 |
| X2000 | TRAIN | T1 | blue | 1 | B0 | 11 | 11 | 0 |
| X2000 | TRAIN | T1 | blue | 1 | B1 | 11 | 11 | 0 |
| X2000 | TRAIN | T1 | green | 0 | B0 | 14 | 14 | 0 |
| X2000 | TRAIN | T1 | green | 0 | B1 | 14 | 14 | 0 |
| X2000 | TRAIN | T1 | green | 1 | B0 | 5 | 5 | 0 |
| X2000 | TRAIN | T1 | green | 1 | B1 | 5 | 5 | 0 |
| X2000 | TRAIN | T1 | red | 0 | B0 | 11 | 11 | 0 |
| X2000 | TRAIN | T1 | red | 0 | B1 | 11 | 11 | 0 |
| X2000 | TRAIN | T1 | red | 1 | B0 | 14 | 14 | 0 |
| X2000 | TRAIN | T1 | red | 1 | B1 | 14 | 14 | 0 |
| X2000 | TRAIN | V0 | blue | 0 | B0 | 5 | 3 | 0 |
| X2000 | TRAIN | V0 | blue | 0 | B1 | 5 | 0 | 0 |
| X2000 | TRAIN | V0 | blue | 1 | B0 | 11 | 11 | 0 |
| X2000 | TRAIN | V0 | blue | 1 | B1 | 11 | 11 | 0 |
| X2000 | TRAIN | V0 | green | 0 | B0 | 14 | 13 | 0 |
| X2000 | TRAIN | V0 | green | 0 | B1 | 14 | 11 | 0 |
| X2000 | TRAIN | V0 | green | 1 | B0 | 5 | 5 | 0 |
| X2000 | TRAIN | V0 | green | 1 | B1 | 5 | 5 | 0 |
| X2000 | TRAIN | V0 | red | 0 | B0 | 11 | 11 | 0 |
| X2000 | TRAIN | V0 | red | 0 | B1 | 11 | 9 | 0 |
| X2000 | TRAIN | V0 | red | 1 | B0 | 14 | 14 | 0 |
| X2000 | TRAIN | V0 | red | 1 | B1 | 14 | 14 | 0 |
| X2000 | TRAIN | V1 | blue | 0 | B0 | 5 | 5 | 0 |
| X2000 | TRAIN | V1 | blue | 0 | B1 | 5 | 5 | 0 |
| X2000 | TRAIN | V1 | blue | 1 | B0 | 11 | 1 | 0 |
| X2000 | TRAIN | V1 | blue | 1 | B1 | 11 | 10 | 0 |
| X2000 | TRAIN | V1 | green | 0 | B0 | 14 | 14 | 0 |
| X2000 | TRAIN | V1 | green | 0 | B1 | 14 | 14 | 0 |
| X2000 | TRAIN | V1 | green | 1 | B0 | 5 | 2 | 0 |
| X2000 | TRAIN | V1 | green | 1 | B1 | 5 | 5 | 0 |
| X2000 | TRAIN | V1 | red | 0 | B0 | 11 | 11 | 0 |
| X2000 | TRAIN | V1 | red | 0 | B1 | 11 | 11 | 0 |
| X2000 | TRAIN | V1 | red | 1 | B0 | 14 | 7 | 0 |
| X2000 | TRAIN | V1 | red | 1 | B1 | 14 | 11 | 0 |

## M3, geometry, empty-language and O64 regression

Original M3 semantic gates remain passing in both arms. X M3 PARAPHRASE/B1 falls from P/F 12/12 to 11/12 and remains above the original gate; this regression is retained. Empty-language rows are diagnostics, not a substitute semantic gate. O64 remains strongly failing: median XYZ about183–186mm and SO3 about1.09–1.10rad. No broad retention or generalist qualification is claimed.

| Point | Task | Kind | Noise | N | median XYZ mm | p95 XYZ mm | median SO3 rad | Reject | Numeric pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F2000 | M1 | EMPTY_LANGUAGE | B0 | 6 | 9.167 | 10.290 | 0.020249 | 0 | True |
| F2000 | M1 | EMPTY_LANGUAGE | B1 | 6 | 11.844 | 12.532 | 0.024996 | 0 | True |
| F2000 | M1 | KIN-VAL | B0 | 16 | 5.626 | 12.302 | 0.008985 | 0 | True |
| F2000 | M1 | KIN-VAL | B1 | 16 | 7.871 | 15.355 | 0.009103 | 0 | True |
| F2000 | M1 | PARAPHRASE | B0 | 12 | 3.630 | 7.554 | 0.005610 | 0 | True |
| F2000 | M1 | PARAPHRASE | B1 | 12 | 4.430 | 8.180 | 0.008305 | 0 | True |
| F2000 | M1 | SEMANTIC | B0 | 12 | 4.914 | 9.517 | 0.008559 | 0 | True |
| F2000 | M1 | SEMANTIC | B1 | 12 | 3.747 | 9.902 | 0.006582 | 0 | True |
| F2000 | M1 | TRAIN | B0 | 32 | 4.311 | 11.432 | 0.006571 | 0 | True |
| F2000 | M1 | TRAIN | B1 | 32 | 4.204 | 15.404 | 0.004099 | 0 | True |
| F2000 | M1 | TRAIN_ENTRY | B0 | 60 | 3.064 | 4.556 | 0.003427 | 0 | True |
| F2000 | M1 | TRAIN_ENTRY | B1 | 60 | 2.264 | 3.485 | 0.003483 | 0 | True |
| F2000 | M3 | EMPTY_LANGUAGE | B0 | 6 | 9.309 | 10.424 | 0.006371 | 0 | True |
| F2000 | M3 | EMPTY_LANGUAGE | B1 | 6 | 11.194 | 12.689 | 0.010072 | 0 | True |
| F2000 | M3 | KIN-VAL | B0 | 16 | 7.915 | 12.574 | 0.002441 | 0 | True |
| F2000 | M3 | KIN-VAL | B1 | 16 | 5.257 | 12.951 | 0.004051 | 0 | True |
| F2000 | M3 | PARAPHRASE | B0 | 12 | 4.525 | 6.869 | 0.001807 | 0 | True |
| F2000 | M3 | PARAPHRASE | B1 | 12 | 3.962 | 5.972 | 0.004843 | 0 | True |
| F2000 | M3 | SEMANTIC | B0 | 12 | 4.047 | 6.899 | 0.001777 | 0 | True |
| F2000 | M3 | SEMANTIC | B1 | 12 | 4.201 | 6.145 | 0.004845 | 0 | True |
| F2000 | M3 | TRAIN | B0 | 32 | 6.504 | 13.867 | 0.004358 | 0 | True |
| F2000 | M3 | TRAIN | B1 | 32 | 6.248 | 11.553 | 0.001892 | 0 | True |
| F2000 | M3 | TRAIN_ENTRY | B0 | 60 | 2.496 | 4.039 | 0.003386 | 0 | True |
| F2000 | M3 | TRAIN_ENTRY | B1 | 60 | 2.353 | 4.042 | 0.002246 | 0 | True |
| F2000 | O64 | O64 | B0 | 64 | 182.331 | 415.198 | 1.092583 | 0 | False |
| F2000 | O64 | O64 | B1 | 64 | 184.615 | 417.354 | 1.097824 | 0 | False |
| P2000 | M1 | EMPTY_LANGUAGE | B0 | 6 | 9.303 | 9.673 | 0.014306 | 0 | True |
| P2000 | M1 | EMPTY_LANGUAGE | B1 | 6 | 10.947 | 11.755 | 0.019003 | 0 | True |
| P2000 | M1 | KIN-VAL | B0 | 16 | 6.256 | 13.748 | 0.011085 | 0 | True |
| P2000 | M1 | KIN-VAL | B1 | 16 | 8.198 | 14.980 | 0.011623 | 0 | True |
| P2000 | M1 | PARAPHRASE | B0 | 12 | 4.798 | 9.394 | 0.007637 | 0 | True |
| P2000 | M1 | PARAPHRASE | B1 | 12 | 5.060 | 9.298 | 0.007927 | 0 | True |
| P2000 | M1 | SEMANTIC | B0 | 12 | 5.317 | 11.603 | 0.008162 | 0 | True |
| P2000 | M1 | SEMANTIC | B1 | 12 | 5.226 | 9.474 | 0.009806 | 0 | True |
| P2000 | M1 | TRAIN | B0 | 32 | 3.870 | 11.550 | 0.007159 | 0 | True |
| P2000 | M1 | TRAIN | B1 | 32 | 4.821 | 14.882 | 0.005522 | 0 | True |
| P2000 | M1 | TRAIN_ENTRY | B0 | 60 | 3.824 | 5.289 | 0.005125 | 0 | True |
| P2000 | M1 | TRAIN_ENTRY | B1 | 60 | 2.333 | 4.057 | 0.002845 | 0 | True |
| P2000 | M3 | EMPTY_LANGUAGE | B0 | 6 | 8.760 | 10.096 | 0.007846 | 0 | True |
| P2000 | M3 | EMPTY_LANGUAGE | B1 | 6 | 9.951 | 12.032 | 0.008162 | 0 | True |
| P2000 | M3 | KIN-VAL | B0 | 16 | 8.687 | 13.965 | 0.002041 | 0 | True |
| P2000 | M3 | KIN-VAL | B1 | 16 | 6.243 | 14.408 | 0.004857 | 0 | True |
| P2000 | M3 | PARAPHRASE | B0 | 12 | 4.940 | 6.413 | 0.001997 | 0 | True |
| P2000 | M3 | PARAPHRASE | B1 | 12 | 4.488 | 5.616 | 0.003858 | 0 | True |
| P2000 | M3 | SEMANTIC | B0 | 12 | 4.874 | 6.578 | 0.001877 | 0 | True |
| P2000 | M3 | SEMANTIC | B1 | 12 | 4.093 | 5.840 | 0.003826 | 0 | True |
| P2000 | M3 | TRAIN | B0 | 32 | 6.249 | 13.000 | 0.003054 | 0 | True |
| P2000 | M3 | TRAIN | B1 | 32 | 4.346 | 14.073 | 0.002360 | 0 | True |
| P2000 | M3 | TRAIN_ENTRY | B0 | 60 | 3.052 | 4.115 | 0.003278 | 0 | True |
| P2000 | M3 | TRAIN_ENTRY | B1 | 60 | 2.701 | 3.950 | 0.001778 | 0 | True |
| P2000 | O64 | O64 | B0 | 64 | 183.806 | 415.877 | 1.095150 | 0 | False |
| P2000 | O64 | O64 | B1 | 64 | 185.710 | 418.201 | 1.099635 | 0 | False |
| X2000 | M1 | EMPTY_LANGUAGE | B0 | 6 | 9.300 | 10.297 | 0.016883 | 0 | True |
| X2000 | M1 | EMPTY_LANGUAGE | B1 | 6 | 11.643 | 12.851 | 0.022660 | 0 | True |
| X2000 | M1 | KIN-VAL | B0 | 16 | 5.812 | 13.019 | 0.009003 | 0 | True |
| X2000 | M1 | KIN-VAL | B1 | 16 | 8.533 | 15.831 | 0.009440 | 0 | True |
| X2000 | M1 | PARAPHRASE | B0 | 12 | 3.494 | 6.932 | 0.005613 | 0 | True |
| X2000 | M1 | PARAPHRASE | B1 | 12 | 4.288 | 7.772 | 0.009131 | 0 | True |
| X2000 | M1 | SEMANTIC | B0 | 12 | 4.419 | 8.534 | 0.008537 | 0 | True |
| X2000 | M1 | SEMANTIC | B1 | 12 | 3.804 | 9.119 | 0.007214 | 0 | True |
| X2000 | M1 | TRAIN | B0 | 32 | 4.649 | 11.659 | 0.005976 | 0 | True |
| X2000 | M1 | TRAIN | B1 | 32 | 4.083 | 15.489 | 0.004010 | 0 | True |
| X2000 | M1 | TRAIN_ENTRY | B0 | 60 | 3.321 | 4.658 | 0.003924 | 0 | True |
| X2000 | M1 | TRAIN_ENTRY | B1 | 60 | 2.185 | 3.689 | 0.003283 | 0 | True |
| X2000 | M3 | EMPTY_LANGUAGE | B0 | 6 | 8.849 | 10.569 | 0.006085 | 0 | True |
| X2000 | M3 | EMPTY_LANGUAGE | B1 | 6 | 10.979 | 12.983 | 0.008269 | 0 | True |
| X2000 | M3 | KIN-VAL | B0 | 16 | 7.258 | 12.821 | 0.002535 | 0 | True |
| X2000 | M3 | KIN-VAL | B1 | 16 | 4.815 | 12.269 | 0.004118 | 0 | True |
| X2000 | M3 | PARAPHRASE | B0 | 12 | 4.471 | 7.236 | 0.001804 | 0 | True |
| X2000 | M3 | PARAPHRASE | B1 | 12 | 4.209 | 6.285 | 0.004674 | 0 | True |
| X2000 | M3 | SEMANTIC | B0 | 12 | 4.083 | 7.292 | 0.001775 | 0 | True |
| X2000 | M3 | SEMANTIC | B1 | 12 | 4.350 | 6.461 | 0.004417 | 0 | True |
| X2000 | M3 | TRAIN | B0 | 32 | 6.756 | 13.860 | 0.003952 | 0 | True |
| X2000 | M3 | TRAIN | B1 | 32 | 6.153 | 11.565 | 0.002080 | 0 | True |
| X2000 | M3 | TRAIN_ENTRY | B0 | 60 | 2.625 | 4.184 | 0.002894 | 0 | True |
| X2000 | M3 | TRAIN_ENTRY | B1 | 60 | 2.407 | 3.956 | 0.002537 | 0 | True |
| X2000 | O64 | O64 | B0 | 64 | 182.549 | 415.103 | 1.093458 | 0 | False |
| X2000 | O64 | O64 | B1 | 64 | 185.056 | 417.442 | 1.098778 | 0 | False |

## Recoverable complete checkpoints

All four archives are full16-file Trainer states, including optimizer/scheduler/RNG, each fixed-revision full-file readback PASS. Only model-only views were materialized on Corelab; these are not presented as resume states. No Trainer weights were relayed to PC.

| Arm | Step | HF revision | Full Trainer tree SHA256 | Files |
| --- | --- | --- | --- | --- |
| F | 1000 | 768cc6b74700d6e3a849064e3c91694f645bdf0a | cdf5155478c885ab77a93e8ff3c823c032c66afb45162ed5e766d69d2fdb1194 | 16 |
| F | 2000 | 4e5f40a4f6fe516003e5ee322a8c92eb7b03a6a4 | ac62469c76725ed28fd6f7eba99da159b940d89865cdc1ac2f9844d68ac6cd0b | 16 |
| X | 1000 | fb9b9f84b5fcb1b67cebb72aea831c68e9be2287 | 3f274001d714345fac898f9c5446add04716f708fd8b0984c185f84ee6031d6c | 16 |
| X | 2000 | a2f9f767a11e4af269377603f3da432f4931d623 | 4ec3b3a44ad40a057cfdcd7ad992a86b68889b4efba98be70ed109e571497694 | 16 |

Repo: `leoyuer/pi0246-gate-pass-campaign`; prefixes `arms/PI0263/{F|X}/checkpoint-{1000|2000}`. Checkpoint1000 is a preserved recovery/milestone state, not a screened candidate.

## Saved-array presentation and evidence

The prospectively sealed18-chapter video contains720 decoded frames,20fps,36seconds,1280×720, using existing RGB/reference/prediction arrays only. It visibly states KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC. Representative frames from all18 chapters were visually inspected; this is not a claim to have manually inspected every frame. Video SHA256 `3eae05f4fb9461f0481abacad74ae7775e692101c80d3920b5bebe1217667dc9`.

The public seal is recorded separately in archive/PUBLIC_RECEIPT.json after anonymous complete-file and every-tar-member readback. The computation closeout precedes that external receipt to avoid a circular archive hash. The final completion audit and GitHub canonical bind the receipt; this report does not infer upload success from a prepared archive.

Machine-readable tables: QUALIFICATION.json, LEGACY_PANEL_ROWS.csv, CROSSED_PANEL_ROWS.csv, COLOR_COMMAND_GROUPS.csv, ORIGINAL_VS_OTHER_BINDINGS.csv, LEGACY_REGRESSION_GROUPS.csv, WORDING_FLIPS.csv, TARGET_CHOICE_DECOMPOSITION.csv and NOISE_BOUNDARY_CROSSINGS.csv. All3456 raw records/tensors, plans, source checks, canaries, training logs, actual RNG journals, graph/loss/gradient audits and HF receipts are retained.

## Closed scientific scope

Both authorized endpoints completed, valid negative qualification retained. No next model is admitted. Stop at STOP_FOR_INDEPENDENT_GENERALIZATION_OR_PHYSICAL_ADMISSION, selected/deployment NONE. Independent generalization or physical admission requires a new decision; no replay of old validation, new loss, extra seed, adapter, scene, IK, physics, WBC, TEST, DEV-F, hardware, broad cleanup, monitor, mail or merge is performed.

## Source table identities

| Source | SHA256 |
| --- | --- |
| COLOR_COMMAND_GROUPS.csv | be0eb912f054524c21c2db61180c3dad58eaf989e16699bfbfc7f047cff45038 |
| CPU_REPLAY.json | 9fd0db9843d956c4ce61cd0da563ebd3847edac24b6cdc8584a905220b41df69 |
| CROSSED_PANEL_ROWS.csv | 9be00bd22d23e1a5c839bb53b4a6691348a4d3ce5d6f7838eb05c290325e8825 |
| FINAL_CLOSEOUT.json | 513d3bdc680030ab165144443248e55a0ce1f7b7161c28b72f0f7b85fcb11505 |
| LEGACY_PANEL_ROWS.csv | d7dd7920607d5c28627de46ec5f6fce178b89ae870dcb637cbcb800d1b8939b1 |
| LEGACY_REGRESSION_GROUPS.csv | c756dab7b0951f6885bc14247a1f506c7102c6a8c24efdf5769a0e201b1fec01 |
| NOISE_BOUNDARY_CROSSINGS.csv | 2f82461705e30044d9b4e5f976234b9963f8f04aa791dd2aed18241bbf1f015f |
| ORIGINAL_VS_OTHER_BINDINGS.csv | 1408e1d2dddcdfdcf151f6415618fc28113fbb9360e10c7a35980d5b861dc101 |
| QUALIFICATION.json | f79dfb35fcc75666364c6c7840e8a28aeb9aef93bd395b2697a834eea199d0be |
| TARGET_CHOICE_DECOMPOSITION.csv | 119914b1fcd328ec2dfa7842c84a0fb3b66a700d25254e98c393cfc5e3e770ad |
| VIDEO_RECEIPT.json | 523c67954b79e11897e2a552eb5852df1b0cd4eb6c71ea33f1eb739bcc888c19 |
| VISUAL_QA_RECEIPT.json | 4d3fccbb91858780142b4443a4d908dcaceb83a28a1d2633b0140d6531b85e85 |
| WORDING_FLIPS.csv | cac004e8c90e7e794b89c65af6c8072ec09c81d9050e0fa6c1753fa1e979cb9c |

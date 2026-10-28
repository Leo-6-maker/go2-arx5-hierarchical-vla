# PI0268 — actual exposure and saved optimization

## Actual consumption

All counts join actual sample logs to the unchanged source rows, not schedule length. 16,384 microbatches, 8,192 per arm, align with native/full loss records; 4,096 gradient records per arm align at the two-microbatch update boundary. Input identity includes the actual language and processed images/state. Numeric label duplicates and language repeats are retained.

| arm | microbatches | distinct input conditions | distinct raw sources | distinct RGB pairs | state / label / language identities | available but unconsumed | terminal / near-terminal microbatches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N | 8192 | 2816 | 1536 | 1464 | 1351 / 1440 / 26 | 9216 | 384 / 1536 |
| R | 8192 | 6912 | 5632 | 5344 | 5035 / 5396 / 26 | 5120 | 384 / 1536 |

Terminal means original anchor15; near-terminal means anchors12–15. Each arm has384 terminal microbatches. R divides these into192 nominal and192 recovery: each recovery variant has32 terminal microbatches across tasks/commands. Neither absent terminal exposure nor infinite terminal stabilization training is supported. The training target is still local H40; repeated hold/late drift remains a separate closed-loop question.

N repeats exact condition identities2/4/6 times (2048/256/512 distinct conditions). R repeats1/2/3 times (6144/256/512). All10752 proposals were eligible; N sees1536 sources, R5632. R recovery source coverage by task is342 of768 for variants1–4 and340 of768 for variants5–6. Full family/task/anchor/variant/template joint cells and unconsumed denominators are retained in [EXPOSURE.json](EXPOSURE.json), [EXPOSURE_SUPPLEMENT.json](EXPOSURE_SUPPLEMENT.json), and the exact [consumed-condition index](CONSUMED_CONDITION_INDEX.json.gz). These identities do not establish how well either model fits those examples.

## Saved loss and gradients

Early/middle/late bins are fixed updates1–1365,1366–2730,2731–4096. Full joined native, position, increment, rotation and speed-margin losses, target magnitudes and clipping indicators are in [OPTIMIZATION_JOIN.csv.gz](OPTIMIZATION_JOIN.csv.gz) and [OPTIMIZATION_PHASE_VARIANT.csv](OPTIMIZATION_PHASE_VARIANT.csv). Stored physical loss values are reported as loss values, not mislabelled metre RMSE. Gradient norms belong to accumulated updates shared by two microbatches, not independent per-example gradient attribution.

| late-phase arm/variant | microbatches | median native loss | median target squared magnitude | median accumulated gradient norm |
| --- | --- | --- | --- | --- |
| N/0 | 2732 | 0.033284 | 0.44272 | 1.9371 |
| R/0 | 1364 | 0.12577 | 0.44272 | 10.437 |
| R/1 | 232 | 8.3947e+08 | 8.9493e+08 | 1.2324e+08 |
| R/2 | 228 | 8.3701e+08 | 8.9497e+08 | 1.251e+08 |
| R/3 | 228 | 4.1895e+07 | 5.248e+07 | 2.6286e+07 |
| R/4 | 228 | 4.2253e+07 | 5.2481e+07 | 2.5148e+07 |
| R/5 | 224 | 0.29261 | 0.9097 | 22.382 |
| R/6 | 228 | 0.33098 | 0.62334 | 23.244 |

The native target is normalized action minus Gaussian epsilon; mask-weighted squared error is divided by active-mask sum. Thus target normalization changes effective native supervision weighting even if the declared loss coefficients are unchanged. Actual native epsilon values were not saved, so target squared magnitude is a scale proxy, not reconstruction of the exact sampled velocity target.

N clips4034/4096 updates; R clips4096/4096. Maximum accumulated norms are72.8648 and136810800. Median absolute step-to-step norm changes are0.6757 and842.9844; R's p95 change is1.077e8. Per-module gradients/activations and full training Euler tensor values are **UNOBSERVED** (training logs retain Euler hashes/dtypes and scalar losses). No fresh gradients were obtained. See [OPTIMIZER_STEP_SUMMARY.json](OPTIMIZER_STEP_SUMMARY.json).

## Saved inference localization — retrospective EVAL only

All2088 old offline rows are retained, including every REJECT and the440 reused donor outputs. Each row includes raw noise, each of four stored S4 x/v stages, final normalized XYZ/rotation, inverse-relative representation and composed physical motion. First-point and segment errors are retained. Matching legacy reference hashes permit all2088 prefix-vs-H40 comparisons; [LEGACY_PREFIX_COMPLETION.json](LEGACY_PREFIX_COMPLETION.json) records the targeted completion of880 legacy columns without model calls.

| M1 point / variant | rows / REJECT | median first S4 XYZ velocity absmax | median final normalized XYZ absmax | median inverse-relative displacement (m) | median prefix / full H40 XYZ RMSE (mm) |
| --- | --- | --- | --- | --- | --- |
| donor/1 | 11 / 0 | 3.672 | 0.8906 | 0.0217 | 5.75 / 17.10 |
| donor/4 | 10 / 0 | 3.656 | 0.9355 | 0.0148 | 3.18 / 8.59 |
| N4096/1 | 11 / 0 | 3.578 | 1.062 | 0.03196 | 3.76 / 8.63 |
| N4096/4 | 10 / 0 | 3.453 | 1.082 | 0.02971 | 1.97 / 9.15 |
| R4096/0 | 64 / 0 | 3.641 | 1.098 | 0.03755 | 2.36 / 9.42 |
| R4096/1 | 11 / 11 | 12.88 | 14.75 | 0.6733 | 89.21 / 382.56 |
| R4096/2 | 11 / 11 | 16.5 | 34.5 | 1.108 | 263.10 / 669.07 |
| R4096/3 | 10 / 10 | 15.41 | 46.38 | 1.842 | 286.33 / 1144.86 |
| R4096/4 | 10 / 10 | 46.12 | 79.25 | 4.743 | 684.35 / 2859.43 |
| R4096/5 | 11 / 0 | 3.516 | 1.07 | 0.03228 | 3.73 / 12.82 |
| R4096/6 | 11 / 0 | 3.453 | 1.164 | 0.03208 | 3.28 / 11.25 |

Full M1/M3/legacy and all variants are in [OFFLINE_TRACE_SUMMARY.json](OFFLINE_TRACE_SUMMARY.json); all rows and gates are in [OFFLINE_TRACE_ROWS.csv](OFFLINE_TRACE_ROWS.csv). Representatives are the first UID per point/task/variant, chosen by metadata rather than outcome. The initial noise stays around unit scale, while R X/Y XYZ velocity is already elevated at the first stored model velocity. Later S4 stages further grow normalized XYZ. Inverse-relative translation and post-anchor displacement norms agree closely: rigid anchor composition does not create the metre-scale magnitude. This localizes the observed growth upstream of final physical composition; it does not identify which learned module caused it.

**Counterevidence to a sufficient-cause claim:** donor and N receive the same very large X/Y state values without R's gate failures. R also improves virtual M1 first-hit attainment. No genuinely seen-TRAIN neural fit panel has run. A numerical intervention must still be evaluated against these alternatives.

The existing axis/gate and paired findings are reused from basis commit7023316d912146e8219b57aae7e938982f3ffee7, especially `OFFLINE_VARIANT_GATE_BREAKDOWN.json`, `CLOSED_RESULTS_PAIRED_ANALYSIS.json` and `OFFLINE_STAGE_RETURN.md`. N improves60/128 paired recovery rows; median paired delta is+0.404mm, despite its smaller aggregate recovery median. No aggregate median is substituted for the paired comparison.

# PI0254 U576 at1152 — fixed midpoint quality

**Diagnostic only. Continue frozen U576/R576 to4608; deployment selected remains NONE.**

384/384 first-valid cells; independent CPU replay PASS. Global704 model queries /2816 head forwards;0 invalid/retry queries.

|Group|Noise|Valid|Model REJECT|Expert REJECT|Model-only REJECT|Median XYZ (mm)|p95 XYZ (mm)|Median SO3 (rad)|Original retention gate|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|O64|B0|64|1|0|1|46.880|120.161|0.060823|FAIL|
|O64|B1|64|0|0|0|47.111|125.199|0.061720|FAIL|
|old16|B0|16|0|0|0|37.295|82.558|0.055097|FAIL|
|old16|B1|16|0|0|0|40.986|80.160|0.056124|FAIL|
|E64|B0|64|0|2|0|70.339|312.045|0.055364|diagnostic|
|E64|B1|64|0|2|0|70.541|313.798|0.055369|diagnostic|
|P64|B0|64|0|0|0|93.668|299.891|0.062841|diagnostic|
|P64|B1|64|0|0|0|94.623|301.315|0.061709|diagnostic|

## New-support comparison with frozen G

- E64_B0: median XYZ ratio to G = 0.763890.
- E64_B1: median XYZ ratio to G = 0.744971.
- P64_B0: median XYZ ratio to G = 0.903022.
- P64_B1: median XYZ ratio to G = 0.919046.

E64 is a trained-on fitting audit. P64 was excluded from this stage training, but FULL already trained on it and C_T is inherited; it is not a blind holdout. REJECT rows remain in all-row metrics. A–D, phase, source-rollout and expert-nonREJECT strata are in quality.json.

All four original O64/old16 × B0/B1 thresholds are reported observationally: zero REJECT, median XYZ≤20mm, p95 XYZ≤35mm, median SO3≤0.05rad. No1152 winner, retuning or quality stop. Only both4608 endpoints determine the research comparison.

Model revision: `d45cc9f2205463193c9b92607f415fa4ed5f664a`; complete Trainer tree: `54f5bbdcfedb202974c3158284531507e401d28cf29dd21167bafbb556afe06d`.
CPU report SHA256: `6dc4d04cebf441ba6aaa74088171e52842a5b46ceafc4e652857ab31e4a2caf5`.

GPU worker exited0. A separately started CPU replay was retired before generating a CPU replay when the original controller started its identical replay; its actual exit-15 and intent/completion/retirement records are retained, with zero added model queries.

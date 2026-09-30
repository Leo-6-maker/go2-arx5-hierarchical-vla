# PI0261 fixed donor/C/P offline quality

All values are from the frozen 2040 logical-row card and independent saved-array CPU replays. The original 440 donor rows were reused. This is kinematic rendered reference evaluation, with no physics, WBC, TEST, DEV-F, or hardware result.

**Research-qualified synthetic candidate:** `NONE`. **Formal selected/deployment checkpoint:** `NONE` / `NONE`.

## Original KIN-VAL numeric gates

| Point | Task | Noise | N | median XYZ mm | p95 XYZ mm | median SO3 rad | REJECT | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Donor | M1 | B0 | 16 | 5.3 | 11.9 | 0.0072 | 0 | PASS |
| Donor | M1 | B1 | 16 | 7.9 | 13.6 | 0.0095 | 0 | PASS |
| Donor | M3 | B0 | 16 | 8.6 | 15.1 | 0.0029 | 0 | PASS |
| Donor | M3 | B1 | 16 | 7.0 | 13.4 | 0.0047 | 0 | PASS |
| C2000 | M1 | B0 | 16 | 5.1 | 11.5 | 0.0081 | 0 | PASS |
| C2000 | M1 | B1 | 16 | 7.3 | 12.9 | 0.0098 | 0 | PASS |
| C2000 | M3 | B0 | 16 | 7.5 | 10.3 | 0.0019 | 0 | PASS |
| C2000 | M3 | B1 | 16 | 5.1 | 11.2 | 0.0051 | 0 | PASS |
| P2000 | M1 | B0 | 16 | 6.3 | 13.7 | 0.0111 | 0 | PASS |
| P2000 | M1 | B1 | 16 | 8.2 | 15.0 | 0.0116 | 0 | PASS |
| P2000 | M3 | B0 | 16 | 8.7 | 14.0 | 0.0020 | 0 | PASS |
| P2000 | M3 | B1 | 16 | 6.2 | 14.4 | 0.0049 | 0 | PASS |

Original gate: median XYZ <=20 mm, p95 XYZ <=35 mm, median SO3 <=.050 rad, zero REJECT.

## Original instruction and paraphrase gates

| Point | Task | Wording | Noise | Correct | c0/6 | c1/6 | both-correct families/6 | Ties | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Donor | M1 | semantic | B0 | 7/12 | 6 | 1 | 1 | 0 | FAIL |
| Donor | M1 | semantic | B1 | 8/12 | 6 | 2 | 2 | 0 | FAIL |
| Donor | M1 | paraphrase | B0 | 6/12 | 6 | 0 | 0 | 0 | FAIL |
| Donor | M1 | paraphrase | B1 | 7/12 | 5 | 2 | 2 | 0 | FAIL |
| Donor | M3 | semantic | B0 | 12/12 | 6 | 6 | 6 | 0 | PASS |
| Donor | M3 | semantic | B1 | 11/12 | 5 | 6 | 5 | 0 | PASS |
| Donor | M3 | paraphrase | B0 | 12/12 | 6 | 6 | 6 | 0 | PASS |
| Donor | M3 | paraphrase | B1 | 11/12 | 5 | 6 | 5 | 0 | PASS |
| C2000 | M1 | semantic | B0 | 6/12 | 5 | 1 | 1 | 0 | FAIL |
| C2000 | M1 | semantic | B1 | 7/12 | 5 | 2 | 2 | 0 | FAIL |
| C2000 | M1 | paraphrase | B0 | 6/12 | 6 | 0 | 0 | 0 | FAIL |
| C2000 | M1 | paraphrase | B1 | 6/12 | 4 | 2 | 1 | 0 | FAIL |
| C2000 | M3 | semantic | B0 | 12/12 | 6 | 6 | 6 | 0 | PASS |
| C2000 | M3 | semantic | B1 | 11/12 | 5 | 6 | 5 | 0 | PASS |
| C2000 | M3 | paraphrase | B0 | 12/12 | 6 | 6 | 6 | 0 | PASS |
| C2000 | M3 | paraphrase | B1 | 12/12 | 6 | 6 | 6 | 0 | PASS |
| P2000 | M1 | semantic | B0 | 8/12 | 5 | 3 | 3 | 0 | FAIL |
| P2000 | M1 | semantic | B1 | 8/12 | 5 | 3 | 3 | 0 | FAIL |
| P2000 | M1 | paraphrase | B0 | 8/12 | 5 | 3 | 2 | 0 | FAIL |
| P2000 | M1 | paraphrase | B1 | 10/12 | 5 | 5 | 4 | 0 | PASS |
| P2000 | M3 | semantic | B0 | 11/12 | 6 | 5 | 5 | 0 | PASS |
| P2000 | M3 | semantic | B1 | 11/12 | 5 | 6 | 5 | 0 | PASS |
| P2000 | M3 | paraphrase | B0 | 11/12 | 6 | 5 | 5 | 0 | PASS |
| P2000 | M3 | paraphrase | B1 | 12/12 | 6 | 6 | 6 | 0 | PASS |

Each task/wording/noise group requires >=10/12. Both commands use their original opposite references; ties count as incorrect.

## TRAIN common-start entry panel

| Point | Task | Noise | TRAIN correct | both-correct families | Ties |
| --- | --- | --- | --- | --- | --- |
| Donor | M1 | B0 | 41/60 | 15/30 | 0 |
| Donor | M1 | B1 | 41/60 | 16/30 | 0 |
| Donor | M3 | B0 | 60/60 | 30/30 | 0 |
| Donor | M3 | B1 | 60/60 | 30/30 | 0 |
| C2000 | M1 | B0 | 50/60 | 21/30 | 0 |
| C2000 | M1 | B1 | 51/60 | 21/30 | 0 |
| C2000 | M3 | B0 | 60/60 | 30/30 | 0 |
| C2000 | M3 | B1 | 60/60 | 30/30 | 0 |
| P2000 | M1 | B0 | 59/60 | 29/30 | 0 |
| P2000 | M1 | B1 | 60/60 | 30/30 | 0 |
| P2000 | M3 | B0 | 60/60 | 30/30 | 0 |
| P2000 | M3 | B1 | 60/60 | 30/30 | 0 |

TRAIN results measure the intervention support. They are not independent generalization evidence.

## O64 disclosed regression panel

| Point | Noise | N | median XYZ mm | p95 XYZ mm | median SO3 rad | REJECT |
| --- | --- | --- | --- | --- | --- | --- |
| Donor | B0 | 64 | 183.9 | 416.6 | 1.0941 | 0 |
| Donor | B1 | 64 | 186.9 | 419.9 | 1.0999 | 0 |
| C2000 | B0 | 64 | 183.2 | 417.5 | 1.0943 | 0 |
| C2000 | B1 | 64 | 186.0 | 420.7 | 1.0996 | 0 |
| P2000 | B0 | 64 | 183.8 | 415.9 | 1.0951 | 0 |
| P2000 | B1 | 64 | 185.7 | 418.2 | 1.0996 | 0 |

O64 is reported under the synthetic-domain processor; it is not a renewed historical qualification.

## Same-observation instruction-swap response

| Point | Task | VAL matched pairs | VAL output difference median mm | TRAIN output difference median mm |
| --- | --- | --- | --- | --- |
| Donor | M1 | 12 | 5.65 | 11.15 |
| Donor | M3 | 12 | 11.52 | 16.34 |
| C2000 | M1 | 12 | 8.09 | 13.37 |
| C2000 | M3 | 12 | 12.07 | 18.46 |
| P2000 | M1 | 12 | 7.97 | 12.33 |
| P2000 | M3 | 12 | 7.72 | 11.83 |

Output difference alone is not correctness; the primary own/opposite reference margin appears in the per-family saved-array rows.

## Reproducible row-level evidence

- `SEMANTIC_BY_FAMILY.csv`: every point, family, command, wording and noise, with own/opposite distance, signed margin and correctness.
- `MATCHED_INPUT_EFFECT.csv`: same-observation/noise output differences and reference separation.
- The three `*_CPU_REPLAY.json` files retain all 680 rows per point, TRAIN entry rows, group metrics, and the original gate computations.

CPU receipt SHA256: Donor `e7ebeb4b38bd8081e8e5a6e19c514a84ccd8df6e611d51ba230562903b055a6a`, C2000 `42495eaedf4eda2a8c3146823d9fc4953c0ab7d0830d19d617f88c8a5da4af97`, P2000 `8004c3d83016ef4bf44f9d3f5491b941a9b75576e885e7f5624415a86d8980aa`.

The semantic and paraphrase families informed PI0261 design. These are development validation results, not blind generalization or physical task success.

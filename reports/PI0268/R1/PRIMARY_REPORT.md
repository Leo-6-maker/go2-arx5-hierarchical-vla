# PI0268-R1 fixed normalization study

All four points and all fixed conditions are retained. Formal selected/deployment: NONE.
OLD0 and ROT0 use identical weights with different processors. Their difference is not fine-tuning gain.
Native losses use different numerical scales; physical errors and gate outcomes are compared below.

## Original task and independent final-five-second retention

| Point | Task/backend | Original /16 | Reliable /16 | Reliable command0/1 /8 | Collisions | Gate refusals |
|---|---|---:|---:|---|---:|---:|
| OLD0 | M1/E0 | 6 | 0 | 0, 0 | 0 | 0 |
| OLD0 | M1/E1 | 7 | 0 | 0, 0 | 0 | 0 |
| OLD0 | M3/E0 | 0 | 0 | 0, 0 | 0 | 0 |
| OLD0 | M3/E1 | 0 | 0 | 0, 0 | 0 | 0 |
| ROT0 | M1/E0 | 0 | 0 | 0, 0 | 0 | 16 |
| ROT0 | M1/E1 | 0 | 0 | 0, 0 | 0 | 16 |
| ROT0 | M3/E0 | 0 | 0 | 0, 0 | 0 | 16 |
| ROT0 | M3/E1 | 0 | 0 | 0, 0 | 0 | 16 |
| OLD2048 | M1/E0 | 15 | 0 | 0, 0 | 0 | 0 |
| OLD2048 | M1/E1 | 15 | 0 | 0, 0 | 0 | 0 |
| OLD2048 | M3/E0 | 0 | 0 | 0, 0 | 0 | 0 |
| OLD2048 | M3/E1 | 0 | 0 | 0, 0 | 0 | 0 |
| ROT2048 | M1/E0 | 11 | 0 | 0, 0 | 0 | 0 |
| ROT2048 | M1/E1 | 11 | 0 | 0, 0 | 0 | 0 |
| ROT2048 | M3/E0 | 0 | 0 | 0, 0 | 0 | 0 |
| ROT2048 | M3/E1 | 0 | 0 | 0, 0 | 0 | 0 |

## Physical fitting and original retention panel

| Point | Panel | Rows | REJECT | Nonfinite | Median XYZ RMSE (m) | Median SO3 RMSE (rad) |
|---|---|---:|---:|---:|---:|---:|
| OLD0 | fit | 224 | 0 | 0 | 0.016968466679890007 | 0.04757429172958828 |
| OLD0 | factor | 304 | 0 | 0 | 0.01355880355779708 | 0.047390281696373875 |
| OLD0 | legacy | 440 | 0 | 0 | 0.007349908555566685 | 0.005771907030069851 |
| ROT0 | fit | 224 | 181 | 0 | 0.016169505663903384 | 0.8403925118871145 |
| ROT0 | factor | 304 | 201 | 0 | 0.012872788595141987 | 0.9253497219579063 |
| ROT0 | legacy | 440 | 330 | 0 | 0.009489565884201683 | 0.8626350385865453 |
| OLD2048 | fit | 224 | 0 | 0 | 0.017569883795533998 | 0.04851998206543233 |
| OLD2048 | factor | 304 | 0 | 0 | 0.016505697888386867 | 0.03853146468581549 |
| OLD2048 | legacy | 440 | 0 | 0 | 0.013940902634710324 | 0.02342575850834791 |
| ROT2048 | fit | 224 | 0 | 0 | 0.00993981845187186 | 0.03687060262863101 |
| ROT2048 | factor | 304 | 0 | 0 | 0.01097700358869502 | 0.037924722798025445 |
| ROT2048 | legacy | 440 | 0 | 0 | 0.015121491176463085 | 0.046877545849429575 |

These aggregates do not replace subgroup qualification. OFFLINE_SUBGROUPS.csv retains each fixed task/command/variant/noise or factor condition. OFFLINE3872.csv keeps family, source identity, all metrics and all failures.
PAIRED_OFFLINE_CHANGES.csv separately reports each contract's own-step0 learning, same-weight contract change and final treatment comparison. Negative error deltas are improvements; undefined changes remain blank rather than converting failures into numerical scores.
VIRTUAL256.csv separates installed-reference endpoint, achieved endpoint, same-tick tracking, original V2/V3, gate/collision, first-success exit and every-tick final-five-second retention. The policy did not use those scores to stop or correct its output.
A nonfinite physical metric is ranked as positive infinity in fixed-denominator median summaries; an infinite median is reported null. Finite-only averages do not silently remove those rows.
CPU replay recomputes saved native BF16 Euler arithmetic and decoding, independent physical error metrics and independent entry geometry. Original task scoring shares the frozen V3 code; that limitation remains explicit.
These are related development conditions from eight existing scene families and fixed noise streams. E0/E1 are mathematical executors, not robots. No blind TEST, physical transfer, real-time or population reliability claim is made.

# Stage24 A2 — Sealed Temporal-Prefix Audit

Date: 2026-09-03 (Asia/Singapore)

## Result

```text
SEALED_PREFIX_AUDIT: PASS
CAUSAL_HORIZON_CLAIM: NOT_MADE
```

The audit re-used the sealed Stage23 BASE/O1 H=40 artifacts and truncated
them to `H ∈ {4, 8, 12, 16, 24, 32, 40}`. It applied no smoothing, clipping,
retiming, resampling, or trajectory correction. The 126 records are six
experiments (BASE and O1 at 10/25/50/100/200 steps), three inference seeds
(`2223/2224/2225`), and seven prefix lengths. Evidence is in
`reports/STAGE24_PREFIX_AUDIT.json`.

## O1@200 median profile by prefix

The table reports position mean error and endpoint error in metres, profile
ratios relative to the expert, and the test-only 2.0 m/s linear-speed gate.

| prefix | position mean | endpoint | path / mean-speed ratio | max-speed ratio | roughness ratio | gate |
|---:|---:|---:|---:|---:|---:|---|
| H4 | 0.00119 | 0.00369 | 2.343 | 3.086 | 5.990 | 3 ACCEPT / 0 REJECT |
| H8 | 0.00423 | 0.01173 | 2.837 | 3.962 | 15.561 | 3 / 0 |
| H12 | 0.00902 | 0.02481 | 5.653 | 9.103 | 58.033 | 3 / 0 |
| H16 | 0.01453 | 0.03637 | 7.740 | 16.543 | 106.043 | 3 / 0 |
| H24 | 0.03137 | 0.07516 | 13.445 | 28.293 | 359.073 | 0 / 3 |
| H32 | 0.04537 | 0.07553 | 18.332 | 44.080 | 563.206 | 0 / 3 |
| H40 | 0.05837 | 0.08007 | 25.566 | 59.205 | 597.662 | 0 / 3 |

The same qualitative pattern is present for BASE and O1@10/@25/@50/@100.
The defect is already visible in H4/H8 and worsens with longer prefixes; this
does not establish a causal horizon effect because these are truncated views
of one already-sampled H40 trajectory, not new H4/H8 model experiments.

## Interpretation

Shortening the evaluated prefix is not yet a justified primary fix: H4 still
has a median path ratio of about 2.34 and H8 about 2.84. The result supports
the narrower statement that the free-running profile is unhealthy from its
earliest audited points, rather than proving that H40 alone creates the
problem. A true H16 architecture/loss comparison remains a future experiment
and was not authorized here.

## Boundary

```text
source outputs: sealed Stage23 evidence only
new model forward/inference/training: false
validation e / holdout f: false
WBC/hardware/PBS: false
```

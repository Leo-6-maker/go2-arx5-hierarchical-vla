# Stage24 A3 — Normalization and Relative-EEF Scale Audit

Date: 2026-09-03 (Asia/Singapore)

## Result

```text
H2_STATISTICS_IDENTITY: PASS
PHYSICAL_TRANSLATION_DEGENERACY: STRUCTURAL_FIRST_POINT_ONLY
RELATIVE_EEF_SCALE: SHIFTED_BUT_REASONABLE_ON_TRANSLATION
ROTATION_SCALE: NOT_COMPARABLE_TASK_CONSTANT_FIXTURE
```

The previous candidate metadata wrapper (`dataset_statistics.json`) was not
used as the frozen Stage22 statistics because its bytes have a different
hash. The audit used the exact materialized H2 `meta/stats.json` and
`meta/relative_stats.json` files:

| input | SHA256 |
|---|---|
| H2 `stats.json` | `fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23` |
| H2 `relative_stats.json` | `8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1` |
| exact base `statistics.json` | `c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9` |
| exact base `processor_config.json` | `85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be` |

Machine-readable output is `reports/STAGE24_NORMALIZATION_AUDIT.json`.

## H2 partition audit

| partition | cells | zero-width cells | finding |
|---|---:|---:|---|
| relative EEF, all 9D × H40 | 360 | 243 | dominated by constant task dimensions |
| translation XYZ | 120 | 3 | one zero-width first relative point per component |
| rot6d, six dimensions | 240 | 240 | orientation held constant by fixture design |
| absolute gripper | 1 | 1 | gripper held constant by fixture design |
| absolute EEF XYZ+rot6d | 9 | 0 | global action statistics nondegenerate |

The H2 relative translation width median is `0.096105`. The three translation
components have 117/120 nonzero time-indexed q01–q99 cells. The first relative
point is structurally zero because the chunk is referenced to the last observed
EEF state. The orientation and gripper zero-width cells are expected for this
fixed-orientation, fixed-gripper visual-reach fixture.

## Comparison with pinned upstream statistics

The exact Stage10 base statistics were compared for the pinned relative-EEF
categories `OXE_DROID`, `XDOF`, and `REAL_G1`. Across their relative-EEF cells:

```text
translation observed cell-width range: 0.107890 .. 0.689208
translation per-dimension median range: 0.223358 .. 0.494698
H2 translation width median:            0.096105
H2 / upstream observed minimum:         0.8908x
H2 / upstream observed maximum:         0.1394x
```

H2 is just below the strict minimum of the observed upstream cell envelope,
but it is the same broad order of magnitude and is not an extreme scale
shift. The comparison is therefore classified as
`SHIFTED_BUT_REASONABLE_ON_TRANSLATION`, not as distributional equivalence.
Rotation is not comparable in this fixture because all six dimensions are
constant. This is descriptive support against a catastrophic translation
normalization mismatch; it does not prove normalization is harmless for
future robot data or explain the O1 oscillation.

The base processor JSON reports `state_dropout_prob=0.2`; the Stage23 O1
launcher explicitly overrode the effective run configuration to `0.0`. That is
a recipe difference recorded for A4, not a hidden normalization correction.

## Boundary

```text
JSON/statistics only; no model construction or tensor forward
no normalization rewrite or data mutation
no validation e / holdout f
no Hopper/PBS/WBC/hardware
```

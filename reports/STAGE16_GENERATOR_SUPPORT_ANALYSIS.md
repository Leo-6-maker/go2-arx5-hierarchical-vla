# Stage16 Generator Support Analysis

## Method

The analysis reuses the approximately 10,000 Stage13B reconstructed generator descriptor samples at:

`/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13b/generator/historical_generator_descriptors.csv`

For each of the 28 benchmark trajectories it computes a standardized nearest descriptor distance, using robust median/IQR scaling and the same curvature proxy definition as Stage13B. Support calibration uses leave-one-out historical distances.

## Result

| Quantity | Result |
|---|---:|
| historical descriptors | 10,000 |
| benchmark trajectories | 28 |
| leave-one-out support p95 | 0.516587 |
| leave-one-out support p99 | 0.697260 |
| benchmark distance min / median / max | 6.02646 / 15.00816 / 17.06324 |
| benchmark cases outside empirical support range | 28 / 28 |

All benchmark cases are marked `OUTSIDE_EMPIRICAL_SUPPORT_RANGE` under this descriptor calibration. The descriptor distance is a support diagnostic, not a claim about actual training OOD exposure.

## Error relationship

Across cases, position-mean error versus descriptor distance had correlation `0.1781`; rotation-mean error had correlation `0.4650`. The 27 completed cases and one tracking abort span the full benchmark distance range; distance alone is not a lifecycle classifier.

Stratified case-average errors:

| Stratum | Count | Position mean (m) | Rotation mean (rad) |
|---|---:|---:|---:|
| speed low | 10 | .030705 | .183242 |
| speed medium | 18 | .020071 | .137580 |
| curvature high | 8 | .018259 | .178780 |
| curvature low | 18 | .027209 | .146925 |
| curvature medium | 2 | .016254 | .116990 |
| progressive relocation | 8 | .017362 | .119482 |
| non-progressive | 20 | .026472 | .167651 |

The current benchmark is mostly low in historical angular-speed, orientation, and workspace quantiles. Those labels should not be inflated into a claim of broad distributional coverage.

## Interpretation boundary

All report/code names use `GENERATOR_DESCRIPTOR_DISTANCE` or `GENERATOR_SUPPORT_DISTANCE`. The result does not establish true training OOD, because the frequency with which these descriptors were actually visited during training is unknown.

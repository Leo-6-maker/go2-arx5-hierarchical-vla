# Stage15 native reachability analysis

## Method

The native reach table `go2_x5_2d.pt` was loaded with
`weights_only=True`. For every derived shape, target positions were evaluated
against the native reset shoulder and a bounded planar grid with extent
`±1.0 m` and step `0.05 m`.

The analysis reports:

- fixed-base maximum reach ratio;
- best single static-base maximum ratio;
- best pointwise ratio over the same grid;
- native placement and target-position bounds.

The table's observed radius range is approximately `0.2558..0.8484 m`.

This is a kinematic, position-only classification. It does not evaluate
orientation, balance, collision, contact, controller dynamics, or continuous
base tracking. It must not be called a dynamic feasibility proof.

## Result for all 114 shapes

| native class | count |
|---|---:|
| `NATIVE_ARM_EASY` | 70 |
| `NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT` | 17 |
| `NATIVE_CONTINUOUS_BASE_MOTION_REQUIRED` | 0 |
| `NATIVE_BOUNDARY` | 3 |
| `NATIVE_GEOMETRICALLY_INFEASIBLE` | 24 |

The absence of a native continuous-base class is an outcome of the reused
shape/placement set under the native geometry, not evidence that the native
robot can never require continuous base motion. A future native-specific
shape can be designed if that scientific case is needed.

## Representative class change

`E2_static_0_0` remains a native static-relocation candidate. The original 16
E2 continuous cases have target excursions outside the bounded native
reachability envelope; all 16 are reclassified as
`NATIVE_GEOMETRICALLY_INFEASIBLE`. The native reclassification is therefore
the correct interpretation of the benchmark, not a policy score.

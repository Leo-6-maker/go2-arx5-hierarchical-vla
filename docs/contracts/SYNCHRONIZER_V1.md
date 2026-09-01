# SynchronizerV1

`SynchronizerV1` aligns source streams for a derived example while preserving
the original timestamps and source indices.

## Rules

- The selected observation stream or explicit sample timestamps define the
  sample times; there is no production 20 Hz default.
- Native state, desired target, achieved EEF, and command streams are aligned
  by source timestamp. The result records indices and state↔command/camera
  skew, not just a resampled array.
- Effective rates are computed from timestamps and compared with declared
  metadata. A mismatch is a visible quality issue.
- Missing frames, gaps, non-monotonic timestamps, and clock discontinuities
  are counted and reported.
- If interpolation is requested, XYZ is interpolated linearly and orientation
  is decoded to SO(3), interpolated with quaternion SLERP, and re-encoded.
  Rot6d values are never linearly interpolated.

Production action timing is `TBD_BY_ADAPTATION_DATASET`. A fixture may use an
explicit test horizon or period only when that provenance is recorded.

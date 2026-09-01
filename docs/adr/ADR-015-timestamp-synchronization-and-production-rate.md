# ADR-015: Preserve source timestamps and defer production action rate

## Status

Accepted — Stage18

## Decision

`SynchronizerV1` preserves source timestamps and reports effective rate,
declared rate, skew, gaps, dropped frames, and clock discontinuities. It does
not assume 20 Hz. A production action sample period is
`TBD_BY_ADAPTATION_DATASET`; test periods must be explicitly labelled.

## Evidence

D3 native state timestamps are 50 Hz, while camera timestamps are spaced by
0.04 s (effective 25 Hz) even though the camera metadata says 20 Hz. The
pipeline retains the 0.04 s timestamps and emits
`D3_CAMERA_FPS_METADATA_MISMATCH`.

## Consequence

The D3 fixture conversion is reproducible with a visible warning, but the
metadata/source-rate disagreement must be reconciled before a real pilot is
declared ready.

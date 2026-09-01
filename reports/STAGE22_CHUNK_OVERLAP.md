# Stage22 Chunk-Overlap Inconsistency

The profile evaluator records both of the required views at every consecutive
same-scene replanning boundary:

1. previous prediction sampled at the new observation timestamp versus the
   new chunk's first point; and
2. the old and new predictions over their full common absolute-future
   interval, including position, rotation, and translational tangent error.

The overlap calculation uses timestamped linear position and quaternion SLERP
sampling and never changes either trajectory. The model-facing prediction
chunk retains its own absolute timestamps, so overlap is not confused with
index-wise comparison of two separately anchored horizons.

The resulting machine-readable field is
`chunk_overlap_metrics` in `stage22_action_profile_evaluation.json`. It is
pending the R0/R1 model checkpoints. The expert oracle itself has no model
profile to compare; its 6/6 pass only qualifies the same-scene expert
reference-update path.

```text
CHUNK_OVERLAP_INCONSISTENCY: IMPLEMENTED; MODEL MEASUREMENT PENDING
trajectory_correction: false
```

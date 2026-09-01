# Stage15 binding self-consistency

## Method

`scripts/native_reference_trace.py` records the native reset state, anchor,
Gamma trajectory, time law, preview, and final policy reference terms without
loading a checkpoint. `scripts/audit_native_reference_binding.py` independently
reconstructs the anchor, bound trajectory, preview, and policy reference from
those records. The comparison uses the native reach table through a safe
weights-only load and the repository's pure NumPy/SO(3) implementation.

## Result

`PASS` over 12 deterministic resets and 12 generated trajectories.

| quantity | maximum error |
|---|---:|
| anchor position | `7.2759e-08 m` |
| bound position | `0.0 m` |
| bound rotation | `4.2147e-08 rad` |
| nine-point preview position | `4.4178e-08 m` |
| nine-point preview rotation | `8.9605e-07 rad` |
| final policy-reference absolute value | `2.4875e-07` |

The complete machine-readable result is retained outside Git at
`/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage15/native_reference_audit.json`.

## Important implementation detail

The first comparison exposed a representation issue: the archived native
PyTorch3D compatibility code flattens the first two **rows** of the rotation
matrix, while the canonical GR00T chunk uses the first two **columns**. The
repository now has explicit `rotation_matrix_to_rot6d_rows` and
`rot6d_rows_to_matrix` functions. After this correction, preview rotation
errors fell to the values in the table above. No linear rot6d interpolation is
used.

## Gate interpretation

The reference-binding blocker is closed. This proves semantic reproduction of
the native reference path; it does not prove that every re-instantiated shape
is reachable or dynamically trackable by the policy.

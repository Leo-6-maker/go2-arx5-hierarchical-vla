# Stage20S Four-Error Decomposition

The accepted seeded easy candidate was run through the delivered native WBC
simulation after the safety gate. Metrics exclude post-reference idle time.
The native rollout has 25 valid logged control samples and ends in
`TRACKING_ABORT`; it is therefore a partial rollout, not a completed task.

| Quantity | Position mean / P95 / max (m) | Rotation mean / P95 / max (rad) | Samples |
|---|---:|---:|---:|
| `VLA_PREDICTION_ERROR` | 0.0184152 / 0.0384457 / 0.0417183 | 0 / 0 / 0 | 40 |
| `REFERENCE_RECONSTRUCTION_ERROR` | 0.0007106 / 0.0012732 / 0.0013653 | 1.69e-9 / 1.69e-8 / 2.11e-8 | 25 |
| `CONTROLLER_TRACKING_ERROR` | 0.0358798 / 0.0589921 / 0.0615563 | 0.254859 / 0.426132 / 0.435170 | 25 |
| `END_TO_END_TASK_SPACE_ERROR` | 0.0373483 / 0.0582339 / 0.0608458 | 0.254859 / 0.426132 / 0.435170 | 25 |

The final label-to-achieved error at the last logged sample was `0.03256 m`
and `0.16376 rad`. The small reference reconstruction error shows that the
provider/binding/native reference path was not the dominant error in this
rollout. The larger controller error and the timing abort are separate from
the model prediction error.

The evaluator estimates the native reset position anchor from logged native
desired references. It does not edit the candidate trajectory. Orientation is
constant in this fixture, so the rotation decomposition is not a broad
orientation claim.

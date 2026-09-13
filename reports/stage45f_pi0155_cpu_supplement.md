# PI0155 CPU supplement (accepted A correction layer)

This is a small, committed index of the read-only supplement produced on
`corelab-4080s`.  The accepted PI0153 seven-row packet was not rewritten and
no model/controller operation was performed.

## Provenance

| item | identity |
| --- | --- |
| PI gate / PR pointer | `PI_GATE_STATE_0155` / Issue comment `5650148861`, PR comment `5650151180` |
| accepted PI0153 packet | `a5f45b8242d4439d39915abd0d4ac9979060708343924f737d8f531b2326508d` |
| TRAIN source kind | `SEALED_FULL24_TRAIN_PARQUET` |
| TRAIN anchor derivation | `target_world_m_minus_canonical_parquet_action_pose39` |
| TRAIN frame classification | `REFERENCE_ORIGIN_DERIVED_NOT_PROVEN_MEASURED_EEF` |
| TRAIN absolute world target metrics | `UNVERIFIED` |
| TRAIN relative metrics | `VALID_FOR_SOURCE_ACTION_FRAME` |
| expected source noise | `5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd` |
| historical C2 actual `x0` | `UNKNOWN_NOT_RECORDED` (seed `4242`, no serialized tensor/effective-entry copy) |

## Existing TRAIN fixtures (CPU-only)

The expert reference has 40 samples at 50 ms, relative position path length
`0.12269180229491666 m`, mean speed `0.06291887297175211 m/s`, maximum speed
`0.09435760404465543 m/s`, and second-difference mean `0.0002357905986479165 m`.
The repeated-StateV1 hold fixture has zero path length, zero speed, and zero
second difference.  Expert-vs-hold XYZ RMSE is `0.04335744241498658 m` and
maximum absolute difference is `0.0942870881408453 m`.

The anchor-applied world rows are retained as a diagnostic view only; because
the anchor is not proven to be a measured EEF pose, their absolute target
progress is not promoted to a task claim.

## Execution boundary

`new_model_forwards=0`, `new_conditioning=0`, `new_action_head=0`,
`gpu=false`, `training=false`, `PBS=false`, `WBC=false`, `simulator=false`,
`hardware=false`, `Validation-E=false`, `Holdout-F=false`.

The complete JSON/CSV/one-page output remains at the sealed node-local output
root recorded in the PI return; this committed page is only the compact,
reviewable summary and does not contain weights, images, or credentials.

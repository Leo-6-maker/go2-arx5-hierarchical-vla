# PI0260-R3 H1: actual data admitted with measured time ambiguity

Authority `fd8e69b59a16aa90b1761d23b3a8f5a78cee8763`. Old R2 H1 BLOCKED and DEV4/20 FAIL remain unchanged. New status: **H1_R3_ADMITTED_WITH_MEASURED_TEMPORAL_AMBIGUITY**. This releases the two real recovery canaries and, if they pass, Base/G6000. It does not prove full observability or model quality.

| Split | Windows | Actual processed conditions | Aliased groups | Aliased windows | Uniform-window deterministic XYZ floor (sqrt MSE) |
|---|---:|---:|---:|---:|---:|
| TRAIN | 13,003 | 7,776 | 2,198 | 7,425 | 1.466 mm |
| KIN-VAL | 4,374 | 2,512 | 800 | 2,662 | 1.426 mm |

No extra many-to-one processor merges, mixed instruction/task identities, cross-split identical processed conditions, or duplicate numeric geometry families were found. The largest within-condition H40 XYZ RMSE separation is 15.767 mm; maximum pointwise separation 25.909 mm; maximum SO3 separation 0.05734 rad. The floor is only the empirical mean-square bound for a deterministic condition-only XYZ prediction under these windows. It is not a model error, median/p95 bound, or replacement training label. The separate frozen 6000-visit exposure floor is in `exposure_floor.json`.

The lead visually inspected 56 saved body/wrist RGB pairs selected by fixed split/task/command/phase and minimum required-cue rules. Target colours and pose axes (M1), and the open contour/endpoints (M3), are visible in those samples. This is a scoped manual review; the saved-array replay and pixel gates cover the remaining population numerically.

Actual processor checks used the original TRAIN-only statistics and frozen tokenizer: eight supervised/free pairs shared byte-identical visual, state and language condition tensors. Native supervised action and mask appear only on the loss path. Forbidden time/phase/source/target/future/depth fields were rejected by the R3 allowlist. An eight-case CPU binding compared the inherited differentiable XYZ/SO3 physical losses with the actual unclamped inference decoder, including out-of-range normalized values; it passed. The 30/10 geometry and full codec receipts were reused.

The 6000 visit rows are frozen at 3000 M1/3000 M3 with balanced commands. The first 156 Motion2 cases and original 64 O64 cases are frozen, producing 1760 logical scores across four model states and two shared noise packets. Twelve paired KIN-VAL semantic source observations were fixed at t0 with opposite-command reference separations retained in the case table. B0/B1 were drawn once on CPU using Torch seeds 26031/26032, shared across all labels and models. No model score has been read.

At this milestone model loads, optimizer updates and quality queries remain zero. Next work is actual initializer binding and continuous8 versus 4+restore+4 canaries, then both authorized 6000-update arms. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

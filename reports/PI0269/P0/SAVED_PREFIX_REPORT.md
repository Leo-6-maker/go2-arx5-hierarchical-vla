# PI0269 P0: saved-state supervision diagnosis

96 fixed proposals: 64 actual CPU label calls, 64 unambiguous labels; 32 missing scheduled queries. No new model, renderer, gradient, optimizer or physics call.

35 s is not on the frozen 0.4 s query grid. Those rows remain NO_QUERY_AT_SCHEDULED_TIME; neither 34.8 nor 35.2 s is substituted.

| Task | Point | Labelled rows | Median prefix XYZ RMSE (mm) | Negative teacher alignment |
|---|---|---:|---:|---:|
| M1 | OLD2048 | 16 | 2.589 | 7/16 |
| M1 | ROT2048 | 16 | 4.106 | 6/16 |
| M3 | OLD2048 | 16 | 4.032 | 2/16 |
| M3 | ROT2048 | 16 | 3.090 | 3/16 |

These encountered-state labels measure saved prefixes against the unchanged teacher at the same pose. They are diagnostic only and never enter TRAIN. The selected families and times are metadata fixed, and the overlapping directional/terminal observations do not identify a unique cause. No new source/time/frame correctness defect was found; the authorized temporal comparison proceeds.

Per-proposal source, prediction, observation and original trajectory hashes, current-state alignment, endpoint discrepancy, hold flags and terminal contraction are retained in SAVED_STATE_PREFIX_AUDIT.json.

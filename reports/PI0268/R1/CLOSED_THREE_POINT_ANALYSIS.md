# PI0268-R1: three closed points, final arm still pending

This is an interim analysis of complete OLD0, ROT0 and OLD2048 panels. ROT2048 virtual evaluation remains active; no partial ROT2048 result is used to change its execution or draw its final verdict. Selected/deployment remain NONE.

## Reaching a target and remaining there

| Point | M1 original success /32 | Successful episodes subsequently outside pose bounds | Median time from success to first exit |
|---|---:|---:|---:|
| OLD0 | 13 | 13 | 2.54 s |
| OLD2048 | 30 | 30 | 6.28 s |

All 32 M1 episodes at each point have complete final-five-second sampling, and every one of their 251 saved ticks is outside the joint position/orientation pose bounds. Training improves original attainment and delays the first exit among each point's successful cases; it does not establish persistent retention. The delay medians describe different successful subsets and are not a matched causal effect.

## M3 entry versus progression after entry

| Point | Broad readiness /32 | V3 entry /32 | Ordered-bin counts | Reverse after entry | Shortcut after entry |
|---|---:|---:|---|---:|---:|
| OLD0 | 1 | 0 | 32 episodes: 0 bins | 0 | 0 |
| OLD2048 | 8 | 6 | 26 episodes: 0 bins; 6 episodes: 1 bin | 6 | 3 |

The six entered OLD2048 episodes all register reverse failure; three also register shortcut failure. These overlapping labels are frozen scorer outputs, not mutually exclusive causes. Failure is therefore not exclusively failure to acquire the observable entry. These data do not identify a unique training or control cause and do not authorize a new intervention.

ROT0 has gate refusals in all 64 episodes and incomplete final-five-second windows. It is not counted as a complete 40-second drift trajectory. OLD0 and ROT0 share weights but use different processors: their difference is a numerical-contract effect, not fine-tuning gain.

## Evidence and boundaries

- `CLOSED_THREE_POINT_VIRTUAL_CPU_REPLAY.json`: 192 closed logical episodes, saved-array V3 replay and independent entry geometry.
- `CLOSED_THREE_POINT_MECHANISM_SUMMARY.json`: source SHA256, individual M3 result hashes and explicit aggregate denominators. SHA256 `9d769496284fdf3163becd19b6ce22ecf884cc0c28b8b72430e94ce983e1a649`.
- `RESULT_ARCHIVE_OLD2048.json`: public fixed revision `341f994c1a9cb4d65775153290e3863624566cc2`, tree `f5f28ec9439b0ee1ae2e2f3ff671d87afcf66e04b201db91af5c35ba5efe2f46`; 35,472 members individually verified from remote readback.

The mechanism aggregation reuses the frozen scorer; it is not a second independent scoring implementation. No model, gradient, renderer, physics or WBC query was added. All failures remain in their original denominators. These are virtual mathematical executors, not physical robot results. Full-study conclusion awaits ROT2048, final CPU replay, videos and final archives.

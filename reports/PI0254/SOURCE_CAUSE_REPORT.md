# PI0254 SOURCE_CAUSE_REPORT

Authority: `7e5a4aae69912ba8e593d50d313f288abc16a914`, Issue #11 canonical 5882151489.

## Evidence-backed conclusions

- The existing 704-row P0 and its 13-window union were reused; no new CALIB model queries were made.
- All 13 original materialized windows bind observation time, current state, desired H40 labels and NATIVE_WBC_RELATIVE_REFERENCE provenance. The original parquet SHA and exact arrays are in source_trace.json.
- Raw collector source_json is absent for all 13. Achieved audit arrays are represented by hashes, and reset/settling order and raw quaternion convention remain UNRESOLVED_SOURCE_EVIDENCE. This stops that traceback, not independent fitting.
- Stored state-to-first-label rotation jumps demonstrate local first-pose incompatibility where present. They do not establish a unique source cause or an H40 aggregate error lower bound.
- No verified per-window native-to-world transform was recovered. Historical native-relative-to-world-target minimum remains invalid for physical interpretation; WORLD_REFERENCE_DISTANCE_UNOBSERVED. No world-distance sidecar is reported.
- The frozen training support retains conflicting expert rows and reports their gate status. No labels, gate, masks, decoder or initialization were changed.

## Original 13 windows

| Source | Actual observation time (s) | State/first-label rotation jump (rad) | Raw evidence |
|---|---:|---:|---|
| pi0227_persistent_train_a_j00_nominal_s02_xm050_r01_r01_227009:obs-0000 | 0.000000000 | 1.6272606795 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_a_j04_ym015_s00_c0_r02_r02_227082:obs-0000 | 0.000000000 | 1.5407360883 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_b_j00_nominal_s04_ym050_r03_r03_227199:obs-0000 | 0.000000000 | 1.5380060141 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_b_j01_xp015_s02_xm050_r01_r01_227209:obs-0000 | 0.000000000 | 1.7384181394 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_b_j03_yp015_s00_c0_r02_r02_227242:obs-0000 | 0.000000000 | 1.8784199891 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_b_j05_zp015_s00_c0_r02_r02_227282:obs-0000 | 0.000000000 | 1.7100644490 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_b_j08_xm010_ym010_s04_ym050_r01_r01_227357:obs-0000 | 0.000000000 | 1.4752439217 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_d_j02_xm015_s02_xm050_r01_r01_227589:obs-0000 | 0.000000000 | 1.7881798866 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_d_j04_ym015_s00_c0_r02_r02_227622:obs-0000 | 0.000000000 | 2.0103091722 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_d_j04_ym015_s00_c0_r02_r02_227622:obs-0006 | 0.119999997 | 1.5413908853 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_d_j04_ym015_s00_c0_r02_r02_227622:obs-0013 | 0.259999994 | 1.7336783322 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_d_j05_zp015_s00_c0_r03_r03_227643:obs-0000 | 0.000000000 | 1.5956778719 | UNRESOLVED_SOURCE_EVIDENCE |
| pi0227_persistent_train_d_j06_zm015_s03_yp050_r02_r02_227674:obs-0000 | 0.000000000 | 1.7315844622 | UNRESOLVED_SOURCE_EVIDENCE |

Observation indices are not H40 action ticks. In particular obs-0013 is around 0.26 s in original observation timestamps; dt=0.05 s remains the unchanged action interval.

## Narrow source statements and proposed repair

`src/go2arx5_vla/data/materialization.py` binds raw state at each sampled observation and desired reference at the corresponding future indices. Stored timestamps/provenance support this materialization statement; absent collector arrays prevent proof of achieved-versus-desired timing and reset settling.

The existing expert reference remapping helper changes reference positions through its stored transform and preserves orientation/time fields. That statement does not prove the missing raw rotation convention for these 13 windows.

Proposed only: a future versioned label/initialization contract should persist achieved and desired arrays, collector timestamps, reset/settling events, quaternion ordering, frame transforms and first-label convention together, with source hashes. No repair is implemented in PI0254.

## Preparation and limits

CPU codec checks cover the fixed 640-window union; selection uses original actual observation times and no model-output criteria. P64 is excluded from both new arms by source rollout, but original FULL already trained on it and C_T is inherited. E64 is trained-on fitting audit. These are repeated development observations, not blind validation or independent training replications.

Source trace SHA256: `7abfd4738378e0e16e8d91e092924be581b63a66b54a7eb7b50c5e276b51b021`.

New model queries: 0 at source-report publication. New optimizer updates: 0. Physical episodes: 0. Deployment selected_checkpoint=NONE.

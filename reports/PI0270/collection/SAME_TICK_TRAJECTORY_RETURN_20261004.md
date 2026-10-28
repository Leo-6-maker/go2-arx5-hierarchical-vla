# PI0270 saved collection: installed reference, achieved trajectory and retention

Authority: release `7a6eea5ae38854bec7b80782213a92ca02fd0318`, Issue #11 canonical5969189182. This is an actual partial TRAIN-collector return, not AGG quality, main-EVAL qualification or the PI0270 terminal.

## Actual execution and replay scope

Finite CPU process3955615 started2026-10-03T19:12:18.215759Z with CUDA hidden, four CPU threads, nice19 and16GiB address-space limit. Its terminal log and completed receipt prove136 saved episodes from the first17 closed fixed families were replayed. The subsequent process query found no live replay PID. The original collector and data owner were not changed.

For each saved query, the replay bound the original observation/prediction/result/trajectory hashes and timestamp, decoded reference and original world anchor. It reconstructed all accepted0.4s prefixes at20ms ticks using the existing interpolation and E0/E1 arithmetic, and compared both installed and achieved XYZ/rotation arrays. Maximum element discrepancy across136 episodes was `8.881784197001252e-16` (threshold1e-12).

The complete original V3 result, including its V2 side report, was reproduced. M3's entry event was independently checked with the existing geometry implementation. The unchanged35..40s251-tick reliability predicate was applied to every completed saved trajectory. Full-episode same-tick XYZ RMS and SO3 RMS were computed from corresponding actual/installed samples; SO3 uses SciPy Rotation magnitude.

Limitations: task scoring shares frozen V3 and executor reconstruction shares the frozen mathematical executor functions. This is saved-array correctness evidence, not an independent physics engine or another neural forward. The prior10400 native-S4 replays remain valid and are not rerun by this trajectory program. Complete192-case S4/trajectory closure remains required after collection.

## Actual partial results

Each group below contains34 episodes from17 TRAIN families, both commands. Repeated commands/executors/ticks are not independent scene replications.

| Task / executor | Original task completion | Reliable | Full safe | Median installed-reference final XYZ error | Median achieved final XYZ error | Median same-tick XYZ RMS |
|---|---:|---:|---:|---:|---:|---:|
| M1 / E0 |33/34|0/34|34/34|140.541mm|140.541mm|0mm|
| M1 / E1 |33/34|0/34|34/34|168.322mm|167.774mm|1.255mm|
| M3 / E0 |0/34|0/34|34/34|62.073mm|62.073mm|0mm|
| M3 / E1 |0/34|0/34|34/34|62.937mm|62.982mm|0.827mm|

All66 M1 first completions occurred by35s and subsequently exited the required pose region. All four groups failed final-five-second retention: each group had8534/8534 saved late ticks outside the combined35mm/.10rad pose criterion. All136 episodes ran the full40s safely. These are original labels, not a shortened success-only rollout.

E0 achieved exactly the installed reference at matching ticks, yet both the installed reference and achieved pose drifted from the goal. E1's median tracking discrepancy was small compared with final target error. This separates generated/installed-reference behavior from mathematical-executor tracking in this observed subset. It supports proceeding with the authorized visited-state recovery intervention; it does not identify a unique neural cause or prove that AGG will learn it. M3 final target distance alone is not ordered-arc success, and its original V3 failures remain intact.

## Latest verified work and raw accounting

At2026-10-03T19:14:32.256089Z (03:14 SGT), actual process state confirmed collector3810700 and finite support owner3854541 live with the original commands. There were144/192 completed collection cases,17 completed family CPU receipts and4992 codec records. Actual journals contained14400 inference attempts,57600 heads,145 starts,14400 frame pairs and5195 teacher calls. These counters can intersect an in-flight attempt; they are not rounded to a fictitious complete episode. AGG optimizer updates and quality queries remain0.

This partial evidence does not satisfy the final96-cell/NEW_FIT admission. Finish the fixed192 original cases and6784 proposals without extra collection or performance-based exclusions, then inspect the actual coverage/mapping/NEW_FIT receipts. On PASS, continue the already-authorized new-data HF readback, Hopper data/runtime/storage admission, real8-vs4-save-freshresume4 canary and AGG4096; preserve both formal complete Trainers and fixed-revision readbacks. Complete all fixed quality panels, full new CPU replay, failure-preserving video, result archives/readbacks and terminal canonical. Explicit protocol stops retain their true evidence.

## Source receipts

- `TRAJECTORY_CPU_collection_136.json`: `2de562ccbf87766f315efe75a96e717230b5de491363f4f8445a54c280c31fd9`.
- `TRAJECTORY_MECHANISM_COLLECTION_136.json`: `9e852af3f2acd01f4660706130f002ccf83cf3075ca9763bfc5d93ceb0c11c8c`.
- `TRAJECTORY_REPLAY_PROCESS_START.json`: `1df85d2012615b0a03223c95943a50502e3803831d126a59462bd44f8f63668d`.
- `TRAJECTORY_RETURN_LIVE_SNAPSHOT_20261004T0315.json`: `c61ddada73ac2fd08da4dbb17e529fc4ffc04f8033fe4abe1a7608f3f75c727c`.
- `trajectory_replay.py`: `c3f89163a033ba64f0e5ed93d514a04bc7ac3e80c033c853600402930434ef05`.
- All136 per-case trajectory receipts were copied byte-exactly into `replay/`; the batch retains their original Corelab paths and SHA256 identities.

New neural/teacher/optimizer/render/physics consumption for this replay and analysis:0. Goal ACTIVE; selected/deployment NONE. No healthy restart, new control, scientific branch, cleanup, monitor/mail, WBC, PR77 pointer or merge.

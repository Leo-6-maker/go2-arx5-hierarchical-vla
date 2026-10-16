# TF-R3 actual reactive Teacher progress

Status: actual new revision running; full task success remains 0. selected/deployment=NONE.

## Preserved actual negative results

- Midpoint source replay `TF_R3_MIDPOINT_T2_DEV_00`: 6300 native packets, task 7.6s, 153 RGB pairs, 19 installed intents. Original joint ranges and native safety passed; no object contact. Next free-approach H40 was rejected for left-pad/cube collision. Valid planning/task FAIL retained.
- Geometry reactive V1 `TF_R3_REACTIVE_T2_DEV_00`: runtime source f36abf4c1df852eab5051081973da6cd74bf81c5. 10700 native packets, task 16.4s, 329 actual RGB pairs, 41 installed intents. No native safety stop or illegal contacts; no object contact or complete success. Query 41 was rejected for original joint acceleration. Independent CPU raw-state/FK/reference/gripper/pause replay passed; task quality remains FAIL.
- Original TF-R2 S7 joint-limit safety FAIL and all old evidence remain unchanged.

## Task-side geometry, explicitly changed

Fixed protocol 8917716022fb8c80ab526b8948248e1468558e8a uses actual object/pad geometry, one measured startup orientation and a pad-envelope clearance derived from the original 0.06m/s x 0.05s limit. Original source EEF is ancestry, not an unchanged replay. Gripper remains midpoint 22mm; BASE intent remains zero. The first overhead approach was certified across 29 H40 windows and both 42/22mm finger geometries on a saved-root clone; fresh native startup and live gates remain required.

No WBC, plant, gains, native safety, collision rule, camera, task scorer, seed or model changes. No GR00T query, optimizer, GPU/PBS, public raw/video or cleanup.

## Affected reference-timing repair

Runtime source f8dd557b951e9d2554e6a9c7225472c7fbc168f4 has complete GitHub blob readback. Old V1 used j/7 for the first eight quintic points: seven 50ms intervals. New explicitly versioned card uses (j+1)/8 end samples for the full 0.4s prefix, preserving the same 2mm endpoint and 32-point hold tail.

Saved query 41 old reference bytes reproduce original acceleration rejection at 1.025305301794832rad/s². Corrected samples on the same measured state, seed-only route witness and endpoint pass fresh 500Hz native IK/collision checks: max acceleration 0.8542819627109121rad/s², max velocity 0.11366932300626242rad/s. Original caps stay 1.0 and 0.7. All 41 installed old V1 windows replay exactly under the new version-compatible code.

A bounded preinstallation governor may try only factors 1/1.25/1.5/2, using step amplitude / factor², and only after rate-only rejection. All factors share the same physical query's original IK/RRT random packet. Collision/limit/unknown failures propagate. No rejected reference is installed; first valid wins, no task-score retries. Each candidate's actual work is separately metered.

## Actual new execution

`TF_R3_REACTIVE_T2_DEV_00_E1`, teacher TF_R3_GEOMETRY_REACTIVE_V2, ROS domain 215, registered immutable source f8dd557b951e9d2554e6a9c7225472c7fbc168f4. Fresh native worker started, PID/start identity verified; no duplicate launch or runtime hot edit. CPU admission is not a dynamic safety or task-success PASS. The next deliverable is the actual contact/task result, then repeatability and data if successful.

Same inherited T0 and deadlines, 4-CPU native worker, 8-CPU concurrency bound; original per-episode physics/duration and per-bundle internal limits retained. Removed cumulative caps stay fully accounted.

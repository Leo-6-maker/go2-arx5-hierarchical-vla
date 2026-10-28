# Layout, Noise And Terminal Analysis

Ten independent scene groups per task; two fixed banks per group. T1: four0/2,two1/2,four2/2; T2: seven0/2,two1/2,one2/2.
All20 pairs have identical initial two-camera hashes and EEF state. In all four discordant pairs, recorded H40 arrays already differ at t=0 with different original fixed noise receipts.
The first original20Hz camera/TCP divergence is reported below;0.4s is only the policy-query interval, not the raw camera sampling rate.

| Task / Scene | First H40 Difference s | First Actual Camera Difference s | First Actual TCP/Gripper Difference s | Outcomes |
|---|---:|---:|---:|---|
| T1 / PI0294/SOURCE_CONFIRM_V1/F1/002 | 0.0 | 0.05 | 0.05 | VALID_TASK_FAIL,SUCCESS |
| T1 / PI0294/SOURCE_CONFIRM_V1/F2/002 | 0.0 | 0.05 | 0.05 | SUCCESS,VALID_TASK_FAIL |
| T2 / PI0294/SOURCE_CONFIRM_V1/F2/000 | 0.0 | 0.05 | 0.05 | SUCCESS,VALID_TASK_FAIL |
| T2 / PI0294/SOURCE_CONFIRM_V1/F2/002 | 0.0 | 0.05 | 0.05 | VALID_TASK_FAIL,SUCCESS |

The original2319 action/noise joins are exact. Different initial noise and different first actions precede later state divergence; this is not a unique stochastic root-cause proof or evidence to select a favorable bank.
T2 successes remain4/20 and all lie inF2 (4/8 within F2). Geometry, push direction, object yaw, colors, lighting and camera parameters are confounded by family.
The original30 T2 TRAIN demonstrations cover all8 Teacher stages, including detach/withdraw/hold; stages are not simply absent. Their labels describe expert-controlled data, unlike the observer Teacher on student rollouts.
TRAIN30 direction range is approximately[-1.593,1.605]rad, distance[0.1563,0.1998]m; new confirmation direction[-1.597,1.588]rad,distance[0.1557,0.1830]m. Overlapping bounding ranges do not prove adequate joint geometric or student-state coverage.
T1 complete failure: transient zone entry followed by retained grasp, terminal release0/41 and hold0/41. Both T2 complete failures end with zone0/41 and violate the original5mm no-lift history predicate. F2/001 briefly enters the zone at17.65s; saying it never entered is incorrect.
The complete per-case signed geometry, timing, original terminal subpredicates and paired draws are in the JSON companions. No original score, split, seed or safety threshold changed.

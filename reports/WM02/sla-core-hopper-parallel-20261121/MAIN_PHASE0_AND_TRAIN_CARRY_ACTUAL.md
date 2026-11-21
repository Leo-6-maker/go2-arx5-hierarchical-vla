# First H200 MAIN updates, phase-zero formal evaluation and Core TRAIN contact audit

At 17:13 SGT on 21 November, H200 job `727341.hopper-m-02` on `hopper-19` had committed 49 formal updates, 3136 Go2, 1568 RoboDojo and 1568 LIBERO visits, and 196 full training H40 generations. The exact phase-zero seal `d7949c75ecc33253bea605a3aed738df4ecdc473d8737848a1bca787ab253c18` contains 140 arrays. Its 2,500 ancestral updates remain separate. The first new update recorded gradient norm 1.08832216 and learning rate 3.618751168e-6. The next scheduled cold boundary is phase 100.

## Independent phase-zero evaluation

VANDA received the complete private checkpoint from Hopper. PBS job `1601959.stdct-mgmt-02` finished on `GN-A40-020` in 243 seconds. All 288 ordinary and 88 phase-diagnostic outputs were archived. A second cold process generated another 48 and matched the ordinary outputs 48/48. A previous allocation, `1601940.stdct-mgmt-02`, stopped at a Python import before its first generation. Its 16 charged seconds were subtracted from the replacement allocation. `1601955.stdct-mgmt-02` runs the independent CPU scheduler for later sealed endpoints.

| Target TRAIN diagnostic | Ordinary 96 | Phase-selected 88 |
|---|---:|---:|
| Full H40 and seam pass | 92/96 | 48/88 |
| First failed finger velocity | 3 | 40 |
| First failed previous-command gripper speed | 1 | 0 |
| Per-window peak P10 speed, p95 | 29.16 mm/s | 45.38 mm/s |
| Per-window peak P10 acceleration, p95 | 524.84 mm/s² | 563.26 mm/s² |

The expert commanded a P10 close in 24 phase-selected windows. The model also commanded a P10 close in all 24. Nine passed the combined timing and original action gates. The 24 paired crossing times had a mean absolute error of 10.42 ms. Speed estimates include the measured preceding command. Acceleration uses adjacent speeds within that sequence. The phase-zero normalized active-action MAE was 0.07857 on the 96 RoboDojo rows and 0.20778 on the 96 LIBERO rows. These values establish the baseline for later cross-endpoint change. Student native requests and integrations are both zero.

## What stopped the expert

Only terminal TRAIN cases were used for this diagnosis. The analysis read the measured contact geometry IDs, the frozen observer's actual allowlist, the final RGB query record and the exact compiled scene.

| F2 TRAIN cells | Native evidence |
|---|---|
| Red HOLD, red-left PLACE, red-right PLACE | All six runs stopped on descent when `x5_link6` contacted the unselected blue cube. |
| Blue-right PLACE | Both runs had already grasped and lifted the blue cube. Both stopped when the carried cube contacted the red cube. Between the last query and STOP the blue cube center dropped 9.06 mm over 78 ms and 8.65 mm over 440 ms respectively. |
| Blue-left PLACE | The first failed on cube-to-cube contact in CARRY. The second failed on right-pad contact with the red cube in OPEN. |

F1 blue-left PLACE also had two different contacts on descent, once between the wrist link and the selected blue cube, once between the right pad and the table. A repeated top-level `ILLEGAL_CONTACT` flag alone cannot establish a shared mechanism for those two cells.

The separate F2 blue-right pilot changes the high-level carrying route to clear the obstacle in Y before crossing in X. The two planar legs have 50.0 mm and 25.5 mm projected clearance on the unopened variant 02 scene. The actual first carrying states of the two failed expert trials give 51.1/27.2 mm and 51.0/27.2 mm for the same route. These bounds use the projected cube footprints at the recorded orientation. They do not establish closed-loop tracking or clearance of the whole moving robot.

Core service `wm02-sla-core-hopper-v1-blue-right-pilot-20261121-01.service` was independently observed active with PID 1973911. It waits for the original TRAIN12→VAL12→TRAIN12→VAL12 prefix and the single-world lock. It is limited to one original unstarted condition, `CHV1-TRAIN-F2-PLACE-blue-right-02`, with the original WBC, contact observer and STOP. Pilot requests are zero at this readback.

Core's original collector had 24 terminal TRAIN requests, 15 grasps and lifts, and seven admitted successes. VAL had 14 terminal requests, eight grasps and lifts, and four admitted successes. Closed ticks totaled 1,081,996. Each of the 25 hard stops had zero later ticks. One further VAL episode was active, and the cumulative sealed TRAIN snapshot contained four episodes.

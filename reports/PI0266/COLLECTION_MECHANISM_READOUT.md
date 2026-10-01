# PI0266 fixed96 feedback collection and data-admission stop

Overall scientific result: PARTIAL / PI0266_FEEDBACK_WINDOW_SHORTFALL.
Generator fine-tuning and learned-policy closed-loop evaluation were not executed.
Formal selected_checkpoint=NONE; deployment_selected_checkpoint=NONE.

## Actual fixed workload

All24 FEEDBACK_TRAIN families x M1/M3 x two commands completed:96/96 first-valid results.79 TASK_CRITERIA_FAIL,17 IK_RESIDUAL,0 task successes,0 recorded safety failures. No failed scene was replaced.17 IK residual cases were M3 command0 and did not install the rejected candidate.

7917 proposed windows;7900 valid, distinct source windows and7900 unique observable contents from all24 families.17 IK_RESIDUAL windows retained as rejected. No exact-visible-content conflicting-future exclusions. A valid local reference is not a full-task demonstration.

## Final window coverage

| Task | Command | Q0 | Q1 | Q2 | Q3 | Endpoint hold | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0 | 393 | 72 | 97 | 1712 | 126 | 2400 |
| M1 | 1 | 2357 | 2 | 6 | 30 | 5 | 2400 |
| M3 | 0 | 697 | 3 | 0 | 0 | 0 | 700 |
| M3 | 1 | 1517 | 856 | 0 | 0 | 27 | 2400 |

The minimum8 windows in each active task/command quartile is violated by M1 command1 Q1/Q2; M3 command0 Q1/Q2/Q3; M3 command1 Q2/Q3. M3 command0 has no endpoint-hold pool. Therefore the fixed256 transfer-unit bank cannot be constructed;0 bank units and0 model optimization are admitted. Total windows and family count do not substitute for these fixed strata.

## Independent saved-array recomputation

Same-owner, separate full-URDF tree traversal of actual root/DOFs matches all96 saved trajectories; prospective V2 scorer is shared and independently re-evaluated on saved arrays. No external reviewer signature is claimed. All7917 saved H40 candidates replay the unmodified original gate; maximum metric difference0.0. All7900 admitted local witnesses /316000 saved points pass separate full-tree FK at the original2mm/.01rad limits, hard/velocity ledger consistency. Producer collision evidence is retained; no new IK/collision search or mobile dynamics guarantee is claimed.

Actual current StateV1 and codec are checked for every admitted window. Fixed-root local witness does not prove full-task execution. There is no result here about donor/R/F quality, data-effect significance or physical generalization.

## Raw accounting

116 physical starts =4 resets +16 diagnostics +96 TRAIN cases; the prephysics invalid attempt is separately retained with0 physical starts.380000 actor forwards,190116 control ticks,760464 simulate calls/integration substeps,3802.319915011525 simulated seconds. All implicit initialization ticks included. GR00T/model/gradient queries0; optimizer updates0. No EVAL images/trajectories/results opened.

## Limits and scientific stop

G0, fixed reset, finite teacher tests and local-window integrity completed. Whole-task results are preserved separately. The fixed bank fails its prospective support rule, so true model restore canaries, R/F4096, four Trainers,32 NEW_SIM_EVAL expert episodes,192 model episodes, legacy R/F panels and TRAIN-fit predictions are not run because their data-admission dependency failed. No additional scenes, seed, teacher formula or quality retry is authorized.

Historical PI0265 PARTIAL/FAIL and all model/language failures remain unchanged. WBC weights/gains/dynamics, gate, H40/dt, original processor and statistics remain frozen. There is no scoped sim candidate.

## Actual video

PI0266_ACTUAL_TEACHERS.mp4 contains all96 chapters from actual body/wrist camera records, including failures and explicit last-frame pauses.2280 frames,2.5fps, every fourth .4-second acquisition (4x playback), NON_REALTIME_LOCKSTEP. Full decode passed. These are ACTUAL_WBC_EXPERT records, with0 GR00T, not model closed-loop or kinematic replacement.


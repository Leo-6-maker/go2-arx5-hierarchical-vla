# E2 real MuJoCo prefix — not a completed panel

As of `2026-11-17T17:17:22.462272+00:00`: **16/24 new native requests started;15 terminal;1 still active.** There are10 composite successes among15 closed requests(10/16 launched, with1 pending). Two old interface results are reused evidence, not new requests. No candidate has been selected; confirmation,E3,DEV and TEST requests remain0.

| Condition | Status | Native ticks | Verified model generations / H40 pass | Grasp / lift / hold | Required base motion |
|---|---|---:|---:|---|---:|
| E2-F1-T0-B0 | ILLEGAL_CONTACT | 20041 | 58 / 58 | 0 grasped complete atoms; interrupted detail unavailable | 116.93mm |
| E2-F1-T0-B1 | COMPOSITE_SUCCESS | 23703 | 72 / 72 | true / 25.91mm / 1.35s | 101.46mm |
| E2-F1-T0-B2 | COMPOSITE_SUCCESS | 23453 | 71 / 71 | true / 25.97mm / 1.15s | 103.28mm |
| E2-F1-T1-B0 | ORIGINAL_FULL_H40_POLICY_REJECTED | 5953 | 2 / 1 | false / 0.00mm / 0.00s | 26.57mm |
| E2-F1-T1-B1 | COMPOSITE_SUCCESS | 22703 | 68 / 68 | true / 24.64mm / 1.10s | 103.97mm |
| E2-F1-T1-B2 | COMPOSITE_SUCCESS | 23203 | 70 / 70 | true / 25.77mm / 1.35s | 102.09mm |
| E2-F1-T2-B0 | COMPOSITE_SUCCESS | 23703 | 72 / 72 | true / 23.71mm / 1.05s | 99.99mm |
| E2-F1-T2-B1 | COMPOSITE_SUCCESS | 22453 | 67 / 67 | true / 24.69mm / 1.05s | 98.05mm |
| E2-F1-T2-B2 | ORIGINAL_FULL_H40_POLICY_REJECTED | 19203 | 55 / 54 | false / 0.00mm / 0.00s | 110.00mm |
| E2-F1-T3-B0 | COMPOSITE_SUCCESS | 22953 | 69 / 69 | true / 25.73mm / 1.25s | 107.04mm |
| E2-F1-T3-B1 | ORIGINAL_FULL_H40_POLICY_REJECTED | 6453 | 4 / 3 | false / 0.00mm / 0.00s | 38.02mm |
| E2-F1-T3-B2 | ILLEGAL_CONTACT | 20226 | 59 / 59 | 0 grasped complete atoms; interrupted detail unavailable | 118.99mm |
| E2-F2-T0-B0 | COMPOSITE_SUCCESS | 22703 | 68 / 68 | true / 25.12mm / 1.20s | 90.14mm |
| E2-F2-T0-B1 | COMPOSITE_SUCCESS | 22203 | 66 / 66 | true / 25.39mm / 1.25s | 90.02mm |
| E2-F2-T0-B2 | COMPOSITE_SUCCESS | 23703 | 72 / 72 | true / 24.53mm / 1.05s | 95.85mm |
| E2-F2-T1-B2 | RUNNING | ≥13953 | 34 / 33 | pending | pending |

In the first complete F1/T0 triplet,B0 stopped on illegal contact while B1 and B2 each genuinely grasped,lifted20mm,held continuously for at least1s,and moved beyond80mm in the required direction. B1 used72 full P10s and B2 used71. Both ended without STOP or technical failure. This is paused-simulation task evidence,not continuous wall-clock real-time qualification or a completed eight-condition model result.

B0 made58 verified returns,all58 passed the original fullH40 gate. It consumed57 full P10s plus3 complete atoms and13 partial ticks,then the original contact checker stopped at tick20,041. Its sealed trace has0 following integrations and no trace error. There were0 recorded grasped complete atoms and no task success. This retained contact failure is not a trajectory-envelope refusal.

Closed-case integrations reconcile exactly: `8683×25+36+15×5703=302656`. Live prefixes independently confirm at least13953 further integrations. At this snapshot the GPU worker had907 completed generations;closed-case verified returns total873,live-case verified returns34. The2,880 native-policy cap is separate from792 E1 offline generations.

The original native result field `model_generations=0` scans obsolete `queries/`. This report instead checks `shift7-queries/` request/return identities,immutable bindings and action hashes. Original result bytes are retained;no active source was hotpatched to repair the report.

Same scene/reset/noise does not imply identical complete control state. Actual tick5,703 packets were checked against each query0 state hash. F1/T0 B0–B1 maximum absolute differences areqpos0.02831055(mixed linear/angular units),qvel0.00092035 andctrl0.87135555;B1–B2 are0.00003337,0.00051865 and0.00806461. Full WBC/selected-reference state identity remains unestablished. The preregistered panel is kept unchanged;these differences limit causal interpretation.

Native source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`;E2 spec `8c9e3555a2094d5905db5eda8d2a821673ebe7284a46d1ffcf6bebacc0f99e82`. Per-condition sources/specs/results,STOP/tick counters and post-init differences are in [the machine-readable prefix](E2_PREFIX_ACTUAL.json). The server queue continues its remaining immutable conditions;no automatic scientific retry or new training.

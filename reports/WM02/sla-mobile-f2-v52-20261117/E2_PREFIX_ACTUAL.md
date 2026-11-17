# E2 real MuJoCo prefix — not a completed panel

As of `2026-11-17T16:14:05.877174+00:00`: **4/24 new native requests started;3 terminal;1 still active.** There are2 composite successes among3 closed requests(2/4 launched, with1 pending). Two old interface results are reused evidence, not new requests. No candidate has been selected; confirmation,E3,DEV and TEST requests remain0.

| Condition | Status | Native ticks | Verified model generations / H40 pass | Grasp / lift / hold | Required base motion |
|---|---|---:|---:|---|---:|
| E2-F1-T0-B0 | ILLEGAL_CONTACT | 20041 | 58 / 58 | 0 grasped complete atoms; interrupted detail unavailable | 116.93mm |
| E2-F1-T0-B1 | COMPOSITE_SUCCESS | 23703 | 72 / 72 | true / 25.91mm / 1.35s | 101.46mm |
| E2-F1-T0-B2 | COMPOSITE_SUCCESS | 23453 | 71 / 71 | true / 25.97mm / 1.15s | 103.28mm |
| E2-F1-T1-B1 | RUNNING | ≥19953 | 58 / 57 | pending | pending |

In the first complete F1/T0 triplet,B0 stopped on illegal contact while B1 and B2 each genuinely grasped,lifted20mm,held continuously for at least1s,and moved beyond80mm in the required direction. B1 used72 full P10s and B2 used71. Both ended without STOP or technical failure. This is paused-simulation task evidence,not continuous wall-clock real-time qualification or a completed eight-condition model result.

B0 made58 verified returns,all58 passed the original fullH40 gate. It consumed57 full P10s plus3 complete atoms and13 partial ticks,then the original contact checker stopped at tick20,041. Its sealed trace has0 following integrations and no trace error. There were0 recorded grasped complete atoms and no task success. This retained contact failure is not a trajectory-envelope refusal.

Closed-case integrations reconcile exactly: `2003×25+13+3×5703=67197`. Live prefixes independently confirm at least19953 further integrations. At this snapshot the GPU worker had259 completed generations;closed-case verified returns total201,live-case verified returns58. The2,880 native-policy cap is separate from792 E1 offline generations.

The original native result field `model_generations=0` scans obsolete `queries/`. This report instead checks `shift7-queries/` request/return identities,immutable bindings and action hashes. Original result bytes are retained;no active source was hotpatched to repair the report.

Same scene/reset/noise does not imply identical complete control state. Actual tick5,703 packets were checked against each query0 state hash. F1/T0 B0–B1 maximum absolute differences areqpos0.02831055(mixed linear/angular units),qvel0.00092035 andctrl0.87135555;B1–B2 are0.00003337,0.00051865 and0.00806461. Full WBC/selected-reference state identity remains unestablished. The preregistered panel is kept unchanged;these differences limit causal interpretation.

Native source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`;E2 spec `8c9e3555a2094d5905db5eda8d2a821673ebe7284a46d1ffcf6bebacc0f99e82`. Per-condition sources/specs/results,STOP/tick counters and post-init differences are in [the machine-readable prefix](E2_PREFIX_ACTUAL.json). The server queue continues its remaining immutable conditions;no automatic scientific retry or new training.

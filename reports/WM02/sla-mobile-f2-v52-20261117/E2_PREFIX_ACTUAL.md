# E2 real MuJoCo prefix — not a completed panel

As of `2026-11-17T16:04:48.419630+00:00`: **3/24 new native requests started;2 terminal;1 still active.** There is one composite success among the2 closed requests(1/3 launched, with one pending). The two old interface results are reused evidence, not new requests. No candidate has been selected; confirmation, E3, DEV and TEST requests remain0.

| Condition | Status | New native ticks | Verified model generations / H40 pass | Grasp / lift / hold | Directional base motion |
|---|---|---:|---:|---|---:|
| E2-F1-T0-B0 | ILLEGAL_CONTACT | 20041 | 58 / 58 | 0 grasped complete atoms; interrupted detail unavailable | 116.93mm |
| E2-F1-T0-B1 | COMPOSITE_SUCCESS | 23703 | 72 / 72 | true / 25.91mm / 1.35s | 101.46mm |
| E2-F1-T0-B2 | RUNNING | ≥11203 | 23 / 22 | pending | pending |

B0 made58 complete model returns, all58 passed the original full H40 gate. It consumed57 full P10s plus3 complete atoms and13 partial ticks, then the original contact checker stopped at tick20,041 for `ILLEGAL_CONTACT`. The sealed native trace has0 following integrations and no trace error. There were0 recorded grasped complete atoms and no task success; this is a retained contact failure, not a trajectory-envelope refusal.

B1 consumed72 full P10s/720 atoms from72 verified full-H40 model returns. It actually grasped, lifted25.91mm, held1.35s, and moved101.46mm along the required direction. It ended at tick23,703(5,703 initialization+18,000 task ticks), without a STOP or technical failure. This is one genuine paused-simulation mobile-grasp success, not continuous wall-clock real-time qualification or a completed8-condition model result.

Closed-case integrations reconcile exactly: `(573+720)×25+13+2×5703=43,744`. The live B2 prefix independently confirms at least11,203 further integrations. The GPU worker had153 attempted/completed generations at this snapshot:130 belong to the two closed cases and23 to the running one. Its2,880 native-policy cap is separate from792 E1 offline generations.

The old native result field `model_generations=0` scans the obsolete `queries/` directory. This report instead checks actual `shift7-queries/` request/return identities, immutable bindings and action hashes, reconciling with the worker total. Original result bytes and failed conditions are retained. No active source was changed to correct the report.

Same scene/reset/noise does not imply identical complete control state. Actual tick5,703 packets were checked against each query0 state hash. F1/T0 B0–B1 maximum absolute differences areqpos0.02831055(mixed linear/angular units),qvel0.00092035 andctrl0.87135555;B1–B2 are0.00003337,0.00051865 and0.00806461. Full WBC/selected-reference state identity remains unestablished. These differences limit a causal reading of this first pair; the preregistered panel is kept unchanged.

Native source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`;E2 spec `8c9e3555a2094d5905db5eda8d2a821673ebe7284a46d1ffcf6bebacc0f99e82`. Per-condition specs/results, STOP/tick counters and post-init differences are in [the machine-readable prefix](E2_PREFIX_ACTUAL.json). The single server-managed queue continues its remaining immutable conditions, with no automatic scientific retry or new training.

## PI0276 actual canary PASS; TASK8192 running

Authority: release `57fd8a9c53ce241c05a8d5b351034bed7ba56b14`, canonical #6000169298. Snapshot: 2026-10-06T02:49:08.045509+00:00.

- Actual job `649362.hopper-m-02`: R on hopper-26, legal auto -> small, one H200. Healthy main training retained; no duplicate submission or reset.
- Real continuous 8 vs 4 + save + fresh-process restore + 4: **PASS_PI0276_REAL_EXACT_RESUME_CANARY**. Model max difference 0 across 1030 tensors; optimizer max difference 0 across 1413 tensors; saved RNG and scheduler exact. Raw completed canary updates: 16.
- Stock S4 parity: all four step inputs/velocities and final raw/decoded outputs identical; shared actual BF16 native epsilon, FP32 generated query/Euler states, unchanged CPU/CUDA RNG. All four Euler steps have nonzero gradients.
- **Parity PASS is integrity, not task quality**: both untrained initial probe paths were REJECT for linear speed 3.1815 m/s versus original 2.0 m/s gate. No rejected reference was physically installed; no gate/threshold change.
- Main continuing HEALTH128: PASS, permitted tensor actually changed, native gripper gradient nonzero, reset=false.
- Authoritative raw journal snapshot: **167/8192 formal completed updates**, **183 total completed updates including canary**. In-flight heads remain in raw accounting. No final-model quality claim.
- Actual 16384-microbatch input admission and fixed task statistics remain sealed; data HF revision `0f64f053ad69b6b4803e712000a0b136cc4623d3`, tree `d30e98ac1979e1f6a8ad98c3131d50f98eb97492fb087a89895921a2f25c9fd1`, all files/members fully read back.
- Three discarded canary payloads retired by the existing helper after comparison; 62,646,721,824 logical bytes, small evidence retained. No general storage cleanup.
- Earlier job 649333 remains preserved as a zero-model-update engineering failure: composite ROT statistics path binding was corrected once and actual closed CPU admission reused. Scientific budget unchanged.

Next: same healthy TASK8192 -> complete Trainer4096/8192 with fixed HF full-file readback -> fixed 384 offline queries and 144 physical conditions (including 80 model episodes) -> independent CPU replay, actual watermarked videos and terminal closeout. TEST remains locked until final policy admission. Frozen WBC/hardware calls remain 0; selected/deployment=NONE.

Evidence: `reports/PI0276/execution/ACTUAL_CANARY_AND_MAIN_HEALTH_20261006.json`; source canary SHA256 `0de3b974d57ebd18aa807db1b7c7f394351e4084d41b020406d59dc4288b865a`; HEALTH128 SHA256 `f38f182ed9be944e98e2499c6c97cfe6ecd95c26e2a9f6139df97eb484e0d5b7`.

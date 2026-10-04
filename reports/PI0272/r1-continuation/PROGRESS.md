# PI0272 R1 execution progress

Authority: release `8a2ffbefe0c57536c0a7e340da0616c7987c74f1`, Issue #11 canonical `5980876835`.

- The scientific precision blocker is resolved. Explicit phases bind training generated queries to FP32 and quality queries to BF16, without casting the qualified S4 path.
- The original 296 conditions, both complete 2048-micro CPU admissions, labels, statistics, schedule and fixed 2148-query membership were reused. Original requested-mode cards and conflict receipts remain intact.
- Changed phase fixtures passed on Hopper CPU; wrong-phase inputs are rejected. Both actual inherited runtime-hook bindings passed. The runtime fileset contains 54 exact files.
- One serialized job was submitted at **2026-10-04 22:21:49 SGT**: **646325.hopper-m-02**, `auto` routed to `small`, project `CFP05-CF-052`, one H200, 225 GB RAM, 12 hours, mail disabled. It began running on **hopper-20 at 22:35 SGT**. No duplicate job or hold was created.
- At **22:44:09 SGT**, both real 8-versus-4+4 canaries have **PASS** receipts. Each compared 1030 model tensors and 1413 optimizer tensors with maximum difference **0.0**, plus exact saved RNG and scheduler. `R1_REAL_CANARIES_AND_MAIN_START.json` binds both receipts. The 32 planned canary updates are charged to the original budget.
- At that snapshot, S4FIT main had **26 completed updates** (53 recorded microbatches, including one in flight); DIRECT main had not started. The ledger held **58 total completed updates / 896 actual heads**. This is a timestamped running snapshot, not either final endpoint or a quality claim.
- The phase-bound quality entry is prepared behind both-final training-audit and exact-fileset guards. Corelab CPU checks passed source imports, rejection of premature evaluation, per-condition arithmetic, saved S4/decoder replay and retention of constructed invalid outputs. These checks made **zero learned queries**. The fixed quality card remains 2148 calls / 6228 planned heads.
- Read-only training audit, model-only materialization, serialized fixed evaluation, CPU report and qualified result archive wrappers are prepared separately from the sealed training runtime. Full actual training audits and final-quality execution remain pending.

The goal remains active. Next: continue the existing healthy allocation through both 1024 endpoints, then fixed quality/CPU replay and full new Trainer archive/readbacks. Scientific conditions and cumulative budgets are unchanged. No quality result, selected checkpoint or deployment candidate is claimed.

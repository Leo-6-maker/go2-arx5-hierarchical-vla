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

## Closed training and active quality (2026-10-04 23:52 SGT)

- PBS job646325 finished at23:31:21 SGT with Exit_status=0. Both arms completed1024 official updates. The actual paired audit passed all4096 main microbatches, native random packets, input/label identities, generated gradients/mode lifecycle and four full Trainer inventories.
- Total including canaries:2080 completed updates,24987 actual training heads; no technical recovery updates. Main-loop runtimes were1890.6555s for S4FIT and937.9158s for DIRECT (model load/archive excluded).
- Four new Trainers have fixed HF uploads. S4FIT512, S4FIT1024 and DIRECT512 have complete25-file readback PASS; DIRECT1024 readback remains active. Local full Trainers remain protected.
- Both new final model views passed exact pinned model-file hashes and559 frozen/471 trainable tensor-scope checks. DIRECT0 reuses the original AGG model view with the explicit DIRECT contract. No optimizer was copied to Corelab.
- Quality fileset75 was sealed and fixed serialized evaluation started on CorelabGPU0. At this timestamp590 S4FIT attempts were registered and589 outputs had durable receipts (one in flight). This is a running progress snapshot, not a final quality verdict.
- Next: finish all2148 fixed queries, CPU saved-array replay and3504-row analysis, complete full Trainer/result readbacks and publish the one terminal canonical. No new science branch, selected or deployment candidate is authorized.

## Fixed result and preservation complete (2026-10-05 SGT)

Both1024 endpoints, all four full HF Trainer readbacks, exactly2148 quality queries/6228 heads, four exact DIRECT repeats, CPU replay and3504-row analysis are complete. There were no technical neural retries. Prediction archive e7cf79062be247d298ea50b9a96999c27e335927 preserves4296 members; summary archive4ae9b50af558fb438b1794e58e8bab592a11cf18 preserves229 members. Both have full anonymous file/member readback.

The primary DIRECT advantage is not established: M3 terminal wrong direction improves AGG35/48 to S4FIT0/48 and DIRECT0/24, but DIRECT-minus-S4FIT median paired prefix MSE is +4.59272529e-7m2. Both fits lose early-support XYZ accuracy relative to AGG; all command/family tradeoffs remain in the full research return. Formal selected/deployment remain NONE.

The prepublication audit passed. Only the single terminal Issue11 canonical and exact-readback/closeout receipt remain before goal closure.

## Terminal closeout (2026-10-05 00:14 SGT)

Single terminal Issue11 canonical5981967270 was published and exactly read back. Final publication and closeout files have anonymous exact-byte HF readback at revision e2a9589382eca22c28299c00aa7b041792584f0c. The final closeout reports no missing scientific requirements. Execution COMPLETE; primary DIRECT advantage NOT ESTABLISHED. Stop at STOP_FOR_FULL_SUPPORT_FEEDBACK_OR_FITTING_DECISION; selected/deployment NONE. No successor run is authorized or started.

# PI0254 execution binding and failure accounting

Authority: 7e5a4aae69912ba8e593d50d313f288abc16a914 / Issue #11 5882151489.

- Scientific arms are serial U576 then R576, each 4608 successful updates from original pinned G2000, fresh identical AdamW and new cosine4608.
- Complete recovery saves: 1152, 2304, 3456, 4608. Only 1152 and 4608 are scientific evaluation milestones. Saves 2304/3456 are recovery states, never candidate screens.
- Keep current complete persistent recovery state and the 1152 snapshot. Retire temporary 1152 full state only after its own final complete state is fixed-revision, full-file SHA read back. Final complete state survives canonical retirement as hardlinks in final_recovery. The CPU lifecycle fixture exercises this adapter with synthetic tiny files; it does not claim HF or GPU training occurred.
- The actual optimizer ledger is durable before each optimizer call and conservative if interrupted. Global cap 9216 includes lost/repeated work. One initial submission plus at most one exact-state engineering resubmission per arm; no fresh restart of consumed work. A recovery that cannot meet the endpoint within the remaining cap retains partial evidence.
- Both schedules use isolated SHA ordering. Random model draws couple by source/visit, not optimizer-step index. 3584 visits overlap on complete arms; R has 1024 additional visits. If U becomes independently invalid, a hash-bound terminal partial receipt permits independent unchanged R with only available actual shared draws verified; no fictitious full paired PASS.
- Inference plan: G E64/P64 B0/B1 256, FULL10000 P64 B0 64, U/R at 1152/4608 O64/E64/P64 B0/B1 1536 = 1856. Cap1920 includes failures; remaining64 only invalid/no-result engineering attempts. Actual head cap7680 counts partial attempts. First valid outputs retained; valid results never retried.
- Training native forwards, S4 rollout forwards and backward recomputations are logged separately using actual training hooks. First real update closes all-four-state gradients/mode, using inherited S4 parity with zero extra parity queries.
- GPU0 Corelab is inference only and permits other projects to coexist. H200 performs capacity-required training. No unrelated process stopped, no old cleanup, mail/monitor off.
- Frozen C_T, original O64 noise, H40/dt .05, decoder, gate and loss remain unchanged. P/E actual dtype/shape/noise bytes and condition hashes are archived before queries. No model input contains expert prefix, future, source ID, or target.
- P1 is diagnostic; a valid mixed result continues unchanged. Qualification is research only; deployment selected_checkpoint=NONE and no sim admission.
- Source64/old16 and original TRAIN dataset are retained. No statistics recomputation. Quota headroom remains UNKNOWN; filesystem space and the bounded write test are not quota proof.

The code commit is bound in regenerated cards. A subsequent prequery freeze commit contains the final cards, CPU receipts, actual noise manifest and fixed HF evidence revision before any new query.
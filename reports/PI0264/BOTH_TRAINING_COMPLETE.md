# PI0264 — both authorized training endpoints complete

Authority: 8245e324b21b25f643eba26df6b2f189981bb40e; Issue #11 5922591496.

- B640758 and L640876 each completed2048 optimizer updates,4096 supervised microbatches; both PBS exit0, run_count1. L completed on Hopper25 H200, walltime1:01:59.
- Both real8 versus4/save/fresh-resume4 canaries PASS. Both actual main input/loss/gradient audits and all2048 cross-arm random packets PASS. No added model or gradient queries by these read-only audits.
- Actual raw accounting:4128 completed updates =4096 main+32 discarded canary;74352 training heads; completed-update technical reserve0. Hard caps4384/110000 unchanged.
- Three complete16-file fixed-HF readbacks PASS: B1024 e3ba080ac786bdc9f30f325107e12ff7e444ad90; B2048 18aa7e584452d9ed173648a0739a94559aed9751; L1024 fdbcd30fe0fc529db3e7f0f604e168ae4c50e08a. L2048 is complete locally and uploading; full readback pending.
- L/global evidence transfer contains93 small members,49,561,600 bytes, SHA256 8aa670acea263d03f950acad392678f353fe59faeb561398f091041cfdefee02. No Trainer weights/optimizer/scheduler/RNG payload is copied to PC or Corelab. All41 frozen execution files retain their original hashes.
- New quality queries remain0. Fixed X2000/B2048/L2048,6336-row evaluation follows final model binding/scope admission. Donor old1152 rows are reused; no1024 quality, new case, seed, loss or training extension.

Training completion is not quality qualification or goal completion. The goal remains ACTIVE until all fixed evaluation, CPU/HF receipts, quality tables, fixed-array video and public evidence readback are complete.

Historical PI0263 COMPLETE/F-X valid FAIL are unchanged. selected_checkpoint=NONE; deployment_selected_checkpoint=NONE. No physics/WBC/TEST/DEV-F/real hardware, monitor/mail, broad cleanup or merge.

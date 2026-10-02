# PI0267-R1 N2048 durable checkpoint return

Release a0f10306419d77242ba31c84d34ce7ce3c59f419; canonical 5944285937.

- Formal job 642638.hopper-m-02 remains RUNNING on hopper-25. Last inspected raw ledger: N2999/4096, R0; no re-submission or healthy training edits.
- Complete N2048 Trainer: 16 recursive files, 20882751041 bytes. Fixed public HF revision `3de45ea32dc4bd326426385a574285fbad30bbd3`, prefix `arms/PI0267/N/checkpoint-2048`, tree `17c19f64b163baa797d47df6740d74909e0a7bcee7c326ffd5de9fec486b1759`. All 16 full files streamed and SHA256-verified. Checkpoint retained locally.
- Complete actual TRAIN input replay: 16384 microbatches, 6912 distinct processor inputs, same frozen statistics. No EVAL payload opened and no new quality query.
- Two strict real recovery canaries PASS; 36 actual discarded canary updates (32 planned plus 4 engineering recovery); no canary weights promoted.
- Scratch quota report 2026-10-02 13:33: 803.3 GB used of 1.0 TB. Approximately196.7 GB quota headroom; global filesystem df is not used as quota evidence. No cleanup performed.
- Original V2 closeout source retained. R1 terminal reconciliation is separately named `pi0267_r1_closeout.py`; original FAIL and 64 source arrays remain unchanged.
- Eight fixed NEW teacher trajectories shown from existing CPU arrays, not new episodes or camera capture. 808 plot frames; separately labelled from actual TRAIN RGB anchor montage. No model query/teacher re-execution/physics/render request.
- Both4096/four Trainer inventories remain the condition for held-out quality evaluation. No intermediate model quality evaluation or candidate selection.

Execution ongoing; model quality not yet measured. selected/deployment NONE. No monitor/mail/hardware/TEST/DEV-F/merge.

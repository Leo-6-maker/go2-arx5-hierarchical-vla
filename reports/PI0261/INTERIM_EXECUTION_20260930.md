## PI0261 controlled pairing — interim execution record

Authority: `6f2353a718cc877dddb3124029de76007eaa7df3`, Issue #11 comment `5903309366`. Evidence branch: `pi0261/controlled-instruction-pairing`, current commit `7fd4de9`.

- S0 source diagnosis and the 2000-unit C/P plan were frozen before new quality queries. The 60 TRAIN common-start pairs have identical processed nonlanguage inputs and distinct instruction conditions. The 2040-row evaluation card was frozen.
- The fixed R3 Base6000 donor full Trainer passed per-file HF revision readback. All 2565 required original source frames were separately staged with per-file SHA checks; the R3 source was not modified.
- The donor's 240 new TRAIN-entry rows completed once. With the reused original 440 rows, independent CPU replay passed for 680 rows. Donor research qualification is false: M1 semantic B0/B1 7/12 and 8/12; M1 paraphrase 6/12 and 7/12. M3 semantic 12/12 and 11/12; paraphrase 12/12 and 11/12.
- C real continuous8 versus split4+restore+4 canary passed. Model max absolute delta 0, optimizer delta 0, scheduler LR delta 0, with matching paired random packets and mode restoration. Job `639304.hopper-m-02` exited 0; 16 discarded canary updates total. Three verified C canary Trainer copies were retired after comparison: 62,647,409,385 logical bytes, while the small audit and raw logs remain.
- P canary `639345.hopper-m-02` is queued for H200. No C/P main update has started, no new endpoint has been selected, and no physical/WBC/TEST work was performed.

Engineering raw accounting: C jobs `639227` and `639242` were cancelled in queue with zero compute; `639255` exited during input prefetch before any model forward or optimizer update because the isolated source lacked frozen frames. The exact original frames are now staged and checked. This is an engineering repair, not a re-run of a valid scientific result.

Current state: `PI0261_CANARY_IN_PROGRESS`; `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

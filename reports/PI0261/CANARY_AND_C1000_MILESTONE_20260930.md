## PI0261 canary closure and first main milestone

Authority remains `PI0261_CONTROLLED_INSTRUCTION_PAIRING_FINETUNE_GO_V1`. This is an execution milestone, not a quality result or model selection.

| Arm | Real canary paths | Saved-state comparison | Raw job outcome |
|---|---|---|---|
| C | continuous8; split4+restore+4 | model/optimizer/LR max delta 0/0/0; packet and mode PASS | `639304.hopper-m-02` exit 0 |
| P | continuous8; split4+restore+4 | model/optimizer/LR max delta 0/0/0; packet and mode PASS | `639345.hopper-m-02` exit 1 after the complete split checkpoint-8 was saved; the redundant final output-root view hit quota. Independent CPU comparison `639369.hopper-m-02` exit 0. |

Both arms completed the fixed 32 discarded canary updates. The P job's exit status remains in raw accounting; its complete split checkpoint and comparison support the real recovery verdict. No additional canary updates were run. After the comparison, only canary Trainer copies and redundant output-root views were retired under exact inventories, leaving scratch at about 96 GiB free. The main code now skips only the final bare model-view save after verifying that the full requested checkpoint exists; checkpoint saves still use the original Trainer path.

C main `639370.hopper-m-02` completed 1000 canonical optimizer updates and 2000 supervised microbatches, exit 0. Its full checkpoint-1000 remains the local exact-resume source. C1000 HF PUBLIC fixed-revision full-file readback is in progress; its upload receipt alone is not archive PASS. C1000→2000 exact resume job `639399.hopper-m-02` is queued. P main has not started. Formal selected and deployment checkpoint remain `NONE`.

The CPU-only archive PBS submission `639397.hopper-m-02` was found routed to a GPU queue and cancelled while queued with zero compute. The same checkpoint is being archived from a low-priority, non-GPU session. No repeated training occurred.

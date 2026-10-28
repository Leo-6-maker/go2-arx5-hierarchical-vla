# C1 full Trainer archival readiness

Fixed PUBLIC prefix: `arms/PI0286/C1_CORRECT1024/checkpoint-1024`, repository `leoyuer/pi0246-gate-pass-campaign`.

Prepared isolated wrapper SHA `b0deb1f9acdbf77a845de9bf540650550b23072ef7b7e3de1e1d557fa9e1f350` reuses the mature PI0284 full14 upload and fixed-revision full-byte readback helper. The actual C1 endpoint, native TRAINING_CPU PASS, step1024/max1024, cursor8192, 2048 microbatches, exact14 members, and parent6144 card are required before upload. No C0 canary or model state substitutes for C1 evidence.

C1 sole archive waiter started at13:27:08SGT after C0 archival completed. It is waiting for the actual C1 complete endpoint and native CPU audit, with one global CPU archival lock, affinity4, 16GiB address-space limit, and upload concurrency4. Network libraries may create additional sleeping threads; CPU execution stays restricted to four cores. Offline training environment, code, and seal stay untouched.

The unique full Trainer remains on Hopper; no full model/optimizer copied to PC/Core. Archival may continue after14:00 with zero scientific calls. Final verifier compares every size/SHA in actual native CPU audit, upload receipt, and remote full-byte readback, recomputes campaign tree, binds audit SHA, and confirms local14 members remain.

Prepared command: `py -3.12 scripts/pi0286_execution/run_trainer_archive.py --arm C1_CORRECT1024`. Keep its foreground SSH session alive until completion. The wrapper waits for actual gates; archive failure may only trigger CPU archival/readback repair, never retraining.

The stopped pre-endpoint waiter was replaced with the direct proven parallel path: the unchanged mature helper executes local full14 hashing and one upload/receipt, then a controlled boundary before its first readback GET invokes the proven four-range ordered SHA implementation. Healthy native training code/seal/job stays unchanged. DNS interruption recovered without this worker changing network configuration.

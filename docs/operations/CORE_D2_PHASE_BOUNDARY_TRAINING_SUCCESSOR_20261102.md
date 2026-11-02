# Fixed-D2 phase/boundary training successor

This adopts F2 of [the controlled fine tuning protocol](../protocols/CONTROLLED_FINETUNING_AND_POLICY_DELIVERY_20261102.md). It changes future sealed jobs only. Publication does not start a job or prove native training qualification.

The existing Core D1/D2 chain finishes first. A successor may wait on Core without holding the mutation owner or loading a model. It must observe the predecessor's actual process exit, completed state, closed native actors and resolved canonical effects before claiming the existing owner lock. Record the D1/D2 endpoint/evaluation outcomes at that boundary. Keep their old valid results and unrun conditions.

Use exactly the admitted D2: 28 entries, 15932 windows, common statistics `187d870e469b38722733ab0f8ac42056c7a13cb38a1349cbc4c73eb7e2eacf39`. The phase index is `4b2056dc56eb9627fffbb35b0e733991e60fb1c41af334027242b45455f4f8ae`. All streams are TRAIN-only and labels stay unchanged. Stage/contact metadata only selects indices.

Run CONTROL, then BOUNDARY. Each first has one real ten-update qualification using the exact first320 draws and original4096 scheduler. Require actual model loading, forward/backward/AdamW updates, complete parent-dependent checkpoint restore, native training headroom and CLOSED producer before its formal job. Each formal job starts afresh at b248, runs4096 updates/batch32, and saves/restores1024/2048/4096 with its exact phase index. No qualification weights or optimizer are carried into the formal job. One known closed arm failure does not stop the independent arm; unknown effects do.

The pair uses identical parent, data, common statistics, M32 interface/DiT30–31, native loss, continuous decoder, seed, warmup128, LR1e-4, WD1e-5 and cosine horizon4096. Only two slots per batch move from uniform to the seven first16 takeover windows. The CPU exposure report and both full draw traces are fixed before any model result. Independent actual TRAIN trace auditing must reproduce those indices, including resumed cursors. A saved checkpoint must include the exact phase-index bytes, not just a same-named file.

GPU0 sharing is authorized. Other users' presence alone is not a refusal; actual24GiB entry,36GiB available host memory, original measured2GiB training margin and storage forecast remain. Project model loads are serial; CPU preparation can run concurrently within actual capacity. GPU1 formal training remains unqualified. Do not alter external processes, active capsules, infrastructure or user data.

The new native reset protocol is a separate pending qualification. This training successor must not silently reuse or amend truncated old panels, claim a PB behavior improvement from TRAIN error, or launch new physical evaluation before `BOOK_RESET_DEV10_V1` is qualified and the paired model set is fixed. Closed training endpoints remain available for that evaluation. Independent shared four-task baseline preparation does not wait for every BOOK mechanism study.

Every job has a finite wall limit (qualification1200s; formal43200s) and safe closure. A future queue may use a new finite96h dispatch window; it must not extend any predecessor's sealed deadline. Return the real manager identity, heartbeat, queue state, native receipts and endpoint paths after actual deployment. Until then report NOT_DEPLOYED.

Validation for the binding change:31 targeted tests passed, including the actual FitSampler's full131072 draws, native10 prefix, resumed cursor, strict recipe invariants and checkpoint index/restore-name checks. This is CPU validation; no new model update, physical condition or formal native qualification is claimed by this document.

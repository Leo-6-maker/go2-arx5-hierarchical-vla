# PI0270 actual AGG4096 training completion — 2026-10-04

This records execution, not a quality PASS. Historical PREFIX failures and WBC PARTIAL remain unchanged. Formal selected/deployment remain NONE.

## Actual training

PBS job `645483.hopper-m-02` completed with Exit_status 0 at 13:51:47 SGT. It ran once on hopper-25, with walltime 02:14:31. No repeated submission or training recovery occurred.

The real continuous8 versus 4-save-freshresume4 canary passed. AGG then completed 4096 optimizer updates from the fixed ROT2048 weights with a fresh optimizer. The frozen loss, processor, precision, scope, paired samples and random packets were unchanged.

`MAIN_TRAINING_AUDIT.json` reports PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY:

- 8192 actual microbatches match the full CPU input plan; 4096 COMMON inputs/labels match the historical control bit for bit.
- All 8192 actual random packets match the control; full S4 loss/gradient/mode checks pass.
- All 192 frozen NEW_FIT inputs are confirmed actually consumed.
- Both 2048 and 4096 complete Trainers have exact per-file inventories.
- Actual cumulative updates: 4112 = 16 canary + 4096 main; technical recovery updates: 0. Actual training heads: 74040, below the 80000 hard cap.

## Model preservation and evaluation status

2048 has complete anonymous fixed-revision full-file readback PASS at HF revision `306c8fcfc162fde03b71fc2c9bd055e10dcfcbde`.

4096 is uploaded at revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`; full-Trainer readback is still in progress at this report's observation. Local complete Trainers remain preserved. Corelab receives only the fixed final model view, not optimizer/Trainer payloads.

The 192 new-support input/label/noise and original executor CPU checks passed. Fixed new quality queries have not started at this observation. Old 1936 offline rows, 128 main episodes and 64 component results remain exact reuse. Pending new queries are the original 968 offline rows, 384 NEW_FIT rows, 64 main episodes and 32 component proposals, followed by CPU replay, labelled video, complete result archive/readback and terminal return.

This is not the study terminal and does not establish reliable virtual, physical, WBC or deployment success. No new scientific recipe, teacher collection, physics/WBC episode, monitor, mail or protected evaluation is introduced.

## Later preservation closure

At 14:14 SGT, the 4096 full-file readback completed successfully: all 18 files, including optimizer, scheduler and RNG state, match the fixed revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`. Tree SHA256 is `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`. The receipt is retained alongside this report. Both complete AGG Trainers are now recoverable from fixed HF revisions; complete local sources remain preserved.

## Fixed evaluation started

At 14:16:49 SGT, a single finite Corelab pipeline (PID 270715) started the existing quality evaluator, followed by the prepared CPU report and saved-frame video. The actual final model precision/frozen-scope check passed: 559 frozen and 471 permitted trainable tensors, total 1030. The 435-file quality execution seal and 192 actual input/noise/executor CPU checks passed.

At 14:18:21 SGT, 36 new offline records were already VALID. Cumulative model attempt/head journals were 19236/76944, including the original 19200/76800 collection consumption. This dated observation proves actual quality execution, not aggregate quality qualification. GPU0 coexists with other projects; unrelated processes remain untouched. Final model files are model-only on Corelab. No monitor or mail was created.

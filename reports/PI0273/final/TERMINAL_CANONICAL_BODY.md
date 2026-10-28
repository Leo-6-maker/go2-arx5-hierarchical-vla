## PI0273 terminal canonical — COMPLETE / both reliable candidates FAIL

Authority `PI0273_FULL_SUPPORT_S4_REPLAY_AND_RELIABILITY_GO_V1`, release `b6e4e436dced30445a014721e2c0df268989244f`, admission canonical #5982104756. This closes the fixed experiment; historical FAILs remain valid.

**PI0273_FULL_SUPPORT_REPLAY_RELIABILITY_COMPLETE — STOP_FOR_INDEPENDENT_TRANSFER_OR_SINGLE_TARGETED_REPAIR_DECISION**

Both BROAD/TARGET completed5120; four complete26-file Trainers have fixed-revision whole-file readback PASS. Both actual continuous8 versus fresh-process4+save+restore+4 canaries have zero model/optimizer difference. All10240 consumed microbatches per arm, full8192-base reuse, paired native/generated packets, original loss/scopes/phase precision, four-node gradients and mode restoration are audited. No quality-driven retraining or new endpoint was used.

| Point | M1 E0/E1 | M3 E0/E1 | Original | Reliable40s | Entire point safe | Original ROT legacy |
|---|---|---|---|---|---|---|
| AGG4096, reused |16/16,16/16|1/16,1/16|34/64|6/64|Yes|PASS|
| BROAD5120 |16/16,16/16|11/16,5/16|48/64|9/64|Yes|PASS|
| TARGET5120 |16/16,16/16|4/16,3/16|39/64|3/64|No:7 virtual collisions|PASS|

**Both original and reliable qualification FAIL; no reliable virtual research candidate.** Group/command AND thresholds and full-point safety remain unchanged. M1 first completion is not reliable holding. TARGET's raw M1 success remains16/16 per backend despite failing point-level safety.

TARGET improves M3 terminal correction over BROAD on20/24 independent condition-mean prefix MSE comparisons (median delta−1.7676252806703948e−5m²); both have0/48 wrong terminal directions versus AGG35/48. But TARGET is worse on early prefix XYZ in59/96 M1 and73/96 M3 paired outputs versus BROAD, has fewer reliable episodes and loses safety. The registered joint targeted-replay advantage is **NOT SUPPORTED**. These two-family TRAIN-support corrections do not establish complete tasks or independent generalization.

Saved-array diagnosis: BROAD M1 first-success exits27/32, final-window SO3 failures26/32; TARGET32/32 for both. M3 no observable entry is BROAD5/32, TARGET13/32; post-entry path/hold failure11/32 and12/32. Flags overlap. Same-tick reference/installed/achieved columns are preserved; E0 has zero tracking residual yet task failures, so executor lag alone does not explain them. No minima-difference is called tracking.

All fixed tables are complete: offline2904; primary early576 with old-S4FIT descriptive context768; late four-point context2368 / condition means1184; main192; component96. CPU replay covers all17152 new S4/decode outputs and128 main+64 component trajectories, with independent arithmetic/entry geometry and shared V3/gate/executor disclosed. Exact old model outputs were reused, not requeried.

Actual raw budget:10272 completed updates incl.32 canary updates;184944 training heads;17152 quality attempts/68608 heads;192 new starts;13648 frame pairs. Technical completed training updates0; technical quality retries0. New teacher/WBC/dynamic physics/hardware all0. All hard caps respected.

### Immutable delivery and readback

- [Full research return, fixed result commit0337d18](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/0337d18f3ab4c3063785c00d14bf3532731b284f/reports/PI0273/final/FINAL_RESEARCH_RETURN.md), SHA256 `ce4fbd502c7029b70d6ed94e5ea36bbc95417df50d828eacf1a63ea1c94db38d`.
- [Complete fixed tables and CPU/transition/video proofs](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/tree/0337d18f3ab4c3063785c00d14bf3532731b284f/reports/PI0273/final).
- Predictions HF PUBLIC revision `eb52b2915bd865066964cad71700f5cfebac30ed`, prefix `evidence/PI0273/full-support-replay-20261005/predictions`, tree `62697e20d984e2851985da5c42f3884a42be09945838598fc48d0680c3a2a991`, **80016 members**; anonymous all-file and every-tar-member streamed SHA/size PASS.
- Summary HF PUBLIC revision `5cd59ada4e39eaeac57734817f74ee58207a1acd`, prefix `evidence/PI0273/full-support-replay-20261005/summary`, tree `7644c534cea4ef0c7842d72ddfd8af48b58132ea4a67f5797b8919ff24dc33a7`, **369 members**; anonymous all-file/every-member PASS. This includes B/T/C/schedules, actual source/gradient/RNG/loss journals, canaries, policies/scopes, all result tables, report and videos. Final closure checks every requirement's hash against archived members; missing scientific requirements=[]
- [BROAD saved-RGB ALL8 video](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/5cd59ada4e39eaeac57734817f74ee58207a1acd/evidence/PI0273/full-support-replay-20261005/summary/PI0273_BROAD5120_VIRTUAL.mp4).
- [TARGET saved-RGB ALL8 video, including collision](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/5cd59ada4e39eaeac57734817f74ee58207a1acd/evidence/PI0273/full-support-replay-20261005/summary/PI0273_TARGET5120_VIRTUAL.mp4).

Four immutable Trainer revisions, HF repository `leoyuer/pi0246-gate-pass-campaign`, prefixes `arms/PI0273/{arm}/checkpoint-{step}`:

| Checkpoint | Revision |
|---|---|
| BROAD2560 |b168fa13b9042daf8e7155f47bee4c5c1b71472d|
| BROAD5120 |5bed9a0d209ccaf23cfcd8bb28fb5a4c41a699c4|
| TARGET2560 |145a8b62810f107aede84983dee6914790cc30da|
| TARGET5120 |dd5902dd8fb999f8891a8239069f8f3e6c6bb718|

Videos show first fixed family028, all8 chapters each, actual96x96 RGB and retained failure/late behavior, explicit2x playback, VIRTUAL/NOT_WBC/NON_REALTIME labels. V3/reliable watermarks are final episode outcomes displayed across the chapter, not current-tick success. TARGET collision chapter ends at its last saved observation14.0s; actual safety stop14.4s, with no fabricated later frame.

Execution COMPLETE and quality FAIL are separate. Single-seed developed geometry, virtual EEF executors, no physical or blind TEST claim. `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`. No automatic replay, recipe adjustment, transfer, new monitor/mail/PR77 pointer or merge. Retain this outcome for PI's independent transfer or single targeted-repair decision.

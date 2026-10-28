# PI0263 both main endpoints complete; X final quality pending

Authority is unchanged: commit `1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28`, canonical5913659202. Historical PI0261 full FAIL/PI0262 COMPLETE-MIXED remain. This closes both training arms, not the full pilot.

## Actual endpoint and integrity evidence

F640396 exit0 completed2000 units/4000 microbatches, wall1:02:38. X640622 exit0 completed2000/4000, wall1:01:10, Hopper H200 hopper-19, active CFP05-CF-052. Native terminal qstat, final segment log and complete local1000/2000 Trainer states agree. X continued unchanged after F's valid quality FAIL; no duplicate submit, new seed, loss or quality-driven intervention.

The source-bound complete read-only audit PASS covers both8000 microbatches: original unit/source/visit/template/color, condition/nonlanguage and labels/masks match the frozen processed plan; native epsilon/tau/free-S4 actual packets agree within every two-command unit and across F/X; recorded modes restored. Both four-step saved-graph checks PASS: full S4/H40, no expert-conditioning or active padding modification, original physical coefficients and four positive first-update Euler gradient norms, all2000 two-microbatch accumulated norms finite per arm. Full loss matches the exact FP32 operations and grouping of the executed source. A preliminary double-precision arithmetic check rejected large native-loss rows; the reader was corrected to replay FP32, then every4000-row sum matched exactly. No train code, canary/quality tolerance or optimizer update changed. The finalizer is isolated under Hopper `evidence/finalize`, leaving the healthy staged execution files intact.

F's serialized scope check also PASS:559 frozen tensors byte-identical to P2000,471 trainable tensors all changed, original493 BF16 backbone and537 FP32 action-head tensors. X's corresponding final-model check will run after fixed-HF materialization. The original-versus-other TRAIN binding report checks640 source memberships and preserves development templates separately; validation source-original V0/V1 is not falsely classified as a supervised TRAIN binding. The video footer names supervised templates without implying validation observations entered training.

## Final training accounting

| Activity | Actual | Authorized hard cap / plan |
|---|---:|---:|
| F main updates / microbatches |2000 /4000|2000 /4000|
| X main updates / microbatches |2000 /4000|2000 /4000|
| Discarded real canary updates |32|32 planned|
| All completed optimizer updates |4032|4288|
| Completed-update technical reserve used |0|256|
| Raw training action-head forwards |72624|100000|

No training is outstanding or newly authorized by unused capacity. Raw completed journal SHA256 `b864285a7633dc0284f6ed158a30e1807db61881e68d8a467fbc7dd3d98b310a`; raw training-head journal `05e9d0818f32fe63d3fac5b67179d6f377c5baac924bb5e2cbf728cb6d6440a2`. Both full journals are now on Corelab with these same hashes. The closed X small-evidence relay bundle (26MiB, no weights) matches all three machines: `e6e4821fce17444d57b1f2d552869f80644356187d7b70dbe071fece1b1458c1`.

## Archives and quality frontier

F1000/F2000 complete16-file fixed HF PUBLIC readbacks PASS (prior quality report and receipts). X1000 now also PASS, repo `leoyuer/pi0246-gate-pass-campaign`, prefix `arms/PI0263/X/checkpoint-1000`, revision `fb9b9f84b5fcb1b67cebb72aea831c68e9be2287`, tree `3f274001d714345fac898f9c5446add04716f708fd8b0984c185f84ee6031d6c`,20,882,605,872 bytes. No1000 quality query. X2000 full archive/readback is running in the existing foreground session; no full PASS claim until its own receipt exists.

P2000 reused1152 rows/CPU replay PASS. F2000 completed1152 rows/CPU replay PASS; original full qualification and new crossed M1 robustness both valid FAIL. Actual quality attempts1116/2432 and heads4464/9728,36 exact forward reuses,0 invalid-only reserve. X2000's fixed1152 rows have not begun; no scores are available. Complete3456 replay, comparative quality report, saved-array720-frame/36s video and final public evidence seal remain mandatory. The extra source/scope/saved-loss readers consumed0 model/gradient queries.

Continue from fixed X2000 readback to its model-only Corelab view, immutable S4 evaluation, independent computation-path CPU comparison and final evidence. No full Trainer transfer to Corelab/PC and no new training. Pilot IN_PROGRESS; synthetic_research_candidate=NONE pending both final quality gates, selected_checkpoint=NONE, deployment_selected_checkpoint=NONE. Physics/WBC/TEST/DEV-F/hardware/mail/monitor/merge remain0. The authorized terminal remains STOP_FOR_INDEPENDENT_GENERALIZATION_OR_PHYSICAL_ADMISSION after full delivery.

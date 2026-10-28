# PI0285 R2 — scoped codec admission and first-policy delivery by 2026-10-09 10:00 SGT

Authority: **PI0285_R2_SCOPED_ADMISSION_AND_1000_POLICY_DELIVERY_GO_V1**.
Parent protocol: `PI0285_FIRST_MANIPULATION_THEN_SHARED_TWO_TASK_V1.md`, release `7e840324c71aef47b401bd861b22710774334c97`, canonical #6058329368.
Review SHA: `dd98db9d1125642904e283ab858d4151b7cd12df`; existing branch `pi0267/executor-decoupled-recovery`.
Latest reviewed evidence: #6064560511 (2026-10-09 00:37:56 SGT), then blocked audit #6064629367 (00:41:59 SGT).
This document and `reports/PI0285/design/R2_DEADLINE_SCOPE_AND_LIMITS.json` become active only after a fixed-release Issue #11 canonical and file/comment readback. Publication is not remote job execution.

## 1. Priority, time and inherited work

The user now requests progress and bounded exploration until **2026-10-09 10:00 Asia/Singapore = 02:00 UTC**, deadline epoch1791511200. This is a new explicit deadline, not a restoration of the expired Oct8 09:00 deadline. No automatic rollover or restart on a new day.

Primary outcome is an independently confirmed T1 source pick/place policy. Save and protect it before any shared T1/T2 training. Shared two-task learning is conditional, not a reason to delay the first deliverable. Both tasks, their physical thresholds and all layout splits remain exactly R1. WBC/Go2/target physics/target queries/hardware remain0; no target demonstrations, target-WBC filtering or low-level retuning.

Adopt existing owner/processes, exact source bank, teacher audits, common TRAIN statistics, plans and language/two-worker proof. Do not re-collect the176 completed expert episodes, restart PI0284, rebuild every cache, redownload pristine weights, create a competing goal, or wait for subagent signatures. A newly healthy job discovered during adoption is not cancelled or edited in flight; reconcile its version and counters first.

## 2. Audit finding: real DEV target clipping, not a teacher or learned-policy failure

PILOT T1/T2 each8/8, TRAIN each64/64, DEV each16/16; all source task successes are expert results. There are160 TRAIN/DEV episodes and128 TRAIN episodes. Each TRAIN episode has601 genuine H40 windows, so76928 TRAIN assertions completed in the original checker before its first DEV assertion failed. The original aggregate TRAIN maxima were never emitted and remain UNOBSERVED.

Of192 fixed DEV representative target windows,190 pass and2 fail. Both failures are layout285200, anchor20, phase0, one per instruction: one shared initial geometry, not two independent failing layouts. Relative rot6d channel8 has34/40 out-of-range coefficients, maximum normalized magnitude1.4482065. Native clipped roundtrip SO3 deviation is0.035746579rad (about2.04813degrees), versus the1e-4rad encoding-fidelity tolerance. XYZ error is negligible and gripper error0; decoded rotations remain finite/nondegenerate. Independent matrix/SciPy calculation and an explicitly diagnostic unclipped inverse isolate clipping for these target roundtrips. They do not establish a general minimum possible model error, physical task failure, or the cause of previous StackCube failures.

Training/model/optimizer calls are0 in the latest receipt. The network-access blocker was resolved; source home exhaustion is not supported by the readback. The current blocker is the interpretation of section5 and the checker, not an observed training failure.

## 3. Binding scientific decision: supervised fidelity is a launch gate; held-out coverage is a disclosed diagnostic

**The two known non-supervised DEV target roundtrip failures DO NOT block PLACE training.** Strict fidelity tolerances continue to apply to every TRAIN target and every later recovery label used for gradients: XYZ<=1e-5m, SO3<=1e-4rad, gripper<=1e-5; finite, nondegenerate rotations and correct time/frame semantics are mandatory. Real policy input integrity, source task/physics validity and all original runtime SafetyGate rules remain mandatory on every split.

This is an explicit R2 amendment to the earlier broader admission scope, NOT a retroactive claim that the original check passed and NOT a same-semantics relabeling. Preserve the original exit1, logs, sealed checker, failed rows, bank/normalizer identities and all DEV denominators. Do not raise tolerance to0.036, delete the two windows, change their labels, fit on DEV, switch decoder, disable inverse clipping or manufacture a missing receipt.

Create a separate `R2_SCOPED_DATA_ADMISSION.json` with status `PASS_SUPERVISED_TRAIN_AND_INPUTS_WITH_DISCLOSED_DEV_EXTRAPOLATION`, original full-check status `FAIL_PRESERVED`, and `DEV_coverage=190/192_representative_windows_NOT_full_DEV_coverage`. It must bind:
- original checker/log/control-flow proof, complete ordered bank and immutable source/statistics/mapping/plan hashes;
- explicit adopted count76928 passed TRAIN assertions; aggregate maxima `UNOBSERVED` unless independently recomputed;
- both exact failed DEV identities and numeric decomposition;
- the actual24-input/six-batch native language/mask/two-spawn-worker proof and exact hash compatibility;
- real canary input from an admitted T1 TRAIN UID; current file availability and complete destination readback.

The documented loop/order/error evidence may be adopted without redoing all76928 checks when its chain is complete. If that chain cannot be established, run a resumable TRAIN-only CPU check once; never infer PASS from data-build exit0. A missing summary maximum alone does not require a full repeat. Existing exact input checks may be reused under unchanged identity.

Authorize the minimum versioned edits to admission/authority/launch readers so they accept this exact R2 status plus evidence, rather than expecting the old all-split PASS string. Keep old failed status files immutable. Do not use `skip_checks=True`, catch-all exception suppression, or a fabricated original PASS. Once this scoped receipt, destination readiness and real carry canary pass, the owner automatically proceeds; no further permission is required for this resolved issue.

### Evaluation treatment

Always compare generated decoded references with the ORIGINAL raw DEV teacher commands and use the unchanged physical object scorer. Also report raw-target versus codec-roundtripped-target error separately, by channel/phase/first8/fullH40. Never replace evaluation truth by clipped targets to improve fit. A native validation loss using transformed DEV targets may be logged, but label its support limitation and do not treat it as raw task accuracy. Keep every DEV condition and all invalid/rejected policy outputs. CONFIRM input/target coverage, if observable only during confirmation, is recorded without changing model or thresholds.

The production processor/statistics remain frozen (statistics SHA256 `c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81`; processor SHA256 `fae35db9d84083cc1faac16442c062c74a22691fa3358cfc58efb759e01977d3`; mapping SHA256 `fb47934b4bd4b5e11ec1c61ade0c5e03ca9110ec5f16abce9633b10cfd9edfbb`). A representational limitation remains a scientific caveat, not a reason to leak held-out data into TRAIN statistics.

## 4. Parallel work: expand information, not concurrent uncontrolled recipes

One owner and at most3 scoped workers:
1. TRAIN/DELIVERY: complete admission/canary, train the one active model, export immutable snapshots and manage the single training lease.
2. EVAL: Corelab model closure, raw-target phase metrics and object funnel; schedule on a distinct legal GPU when available, otherwise serialize at checkpoint boundaries. No reading live mutable weights.
3. DATA/ATTRIBUTION: CPU-only coverage and observability diagnostics, prepare the one possible corrective-data route and read-only two-task readiness. Cannot change a running processor/teacher/scene or submit competing training.

CPU tasks run concurrently only if they do not starve training I/O or evaluation. A status log, hash or new worker is not a new model result. Existing source data/teacher gates are adopted; no repeated teacher collection just to reproduce already valid successes.

### Read-only representation/behavior probes

Use stored raw arrays to report TRAIN and DEV state-versus-action range coverage separately, stage transitions, first8/tail-only events, and command/installed/achieved/object differences. Bind whether phase marks the issued command or the post-transition state; do not silently relabel the sealed plan.

Allow one CPU-only counterfactual of using the mathematical[-1,1] domain for all six VALID relative rotation-matrix components, with original XYZ/gripper scaling unchanged. Compute label roundtrips and scaling/gradient-unit implications on TRAIN and the already exposed DEV diagnostics. This is a candidate representation design, NOT an inference decoder change, not a new fitted normalizer and not authorized for gradients in R2. Alternative representations must preserve SO3 checks; rot6d redundancy means the observed roundtrip error is not automatically a model-wide irreducible error floor.

Allow stored-array temporal/near-observation analysis of approach, release and post-placement states. Exact duplicate pixels are not the only possible ambiguity; lack of duplicates does not prove full observability. Do not inject object truth, teacher phase or fake history into the policy. New history inputs, visual unfreezing, state dropout, K16 or a normalization-training arm are preparation-only and require a later scientific decision. Earlier K4/K16 and LR comparisons are not repeated here.

Cap these optional CPU preparations to90 aggregate worker wall-minutes and within total CPU budget; their absence cannot block an otherwise admitted mainline. This scope explores representation, conditional behavior and recovery while keeping only one main optimization recipe.

## 5. First-policy mainline: pristine PLACE4096 with early real feedback

Use the R1 actual common-scene T1 bank (64 TRAIN layouts) and immutable plan. Start ONCE from original `nvidia/GR00T-N1.7-3B` revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`, pinned official code `51d4c89f72fda44cbf77285c6a8114b52676b8a1`; no old task-specialized weights. T2 contributed common TRAIN normalization ranges, not T1 gradient labels; disclose known-task-set preprocessing.

Keep native masked flow-matching, row10 state/action interfaces plus shared DiT training, frozen VLM/other31 rows/position/vlln/VL attention, FP32 trainable masters with BF16 autocast, native BF16 K4 inference and FP32 physical decode, TF32/FP16 off. Effective batch8; preferred micro4 xaccum2; peak interface/DiT LR1e-4/1e-5, AdamW(.9,.999),eps1e-8, decay0/1e-5, gradclip1, cosine4096 with205warmup. State dropout0, current real128px two-view input/native256 preprocessing, H40/first8/.4s,20Hz control/100Hz physics remain fixed.

Use one real continuous8 versus4+save+fresh4 proof for the genuinely new language/data training path; count all proof updates. Do not repeat it per checkpoint or phase when the carry semantics are unchanged. Check new data consumption, native noise/time, Adam/scheduler/RNG/cursor, frozen rows and generated output. Unknown details stay unknown, not filled by labels.

Save full recovery at least each1024 updates and every phase boundary; immutable inference snapshots at1024/2048/4096. Fixed DIAG remains96 rows/task:8TRAIN+8DEV layouts with6 fixed behavior anchors; actual noise packets are fixed before results. Report dynamic stages separately from late holds, raw preclip excursions, and object funnel; a small overall gripper median is not release accuracy.

R2 permits TRAIN8 screening at1024,2048 and4096 (one additional early screen relative to R1). When TRAIN>=7/8, run the entire unchanged DEV16, not only in-range examples. The FIRST checkpoint with TRAIN>=7/8 and DEV>=13/16 is frozen immediately for CONFIRM_A32; do not keep optimizing for a prettier loss or wait to finish the nominal4096. If early qualification occurs while a later chunk is healthy, stop at a safe saved boundary and preserve unused/extra updates honestly; do not mix weights.

The additional1024 screen is optional when it would delay useful training or final confirmation; skipped checks remain NOT_EXECUTED, not quality FAIL. No other intermediate quality-search checkpoint is authorized. All valid results remain first-valid.

## 6. After primary4096: at most ONE T1 adaptation branch, not a search tree

T1 total formal updates remain<=6144. The reserved2048 may be used for either **RECOVERY2048** or **DOSE2048**, never both. Decide from the complete4096 fixed TRAIN evidence, before new branch training; write a frozen branch card. Do not start an optional branch after the cutoff in section9.

Use the existing first four nonterminal DIAG anchors for each of8TRAIN layouts (32 dynamic rows) to separate fit from closure. A diagnostic dynamic fit means prefix XYZ<=.01m,SO3<=.10rad,gripper MAE<=.10. These are diagnostic gates, not substitutes for object success. Define J as the mean of squared ratios to these three tolerances.

**RECOVERY priority:** route when >=24/32 dynamic TRAIN rows pass with at least4/8 in each nonterminal group, OR at least6/8 actual rollouts complete grasp/lift/transfer and enter the correct placement region but fail release/withdraw/hold. Coverage and fitting deficits can coexist; these gates identify a practical recovery hypothesis, not a unique cause. Perform the original32 requested actual learner-state branches (8TRAIN layouts x2,6,10,16s) using saved prefix replay or validated exact state restoration. Use the already qualified same teacher, with new real suffix observations/commands; teacher/object truth stays offline. Missing/unvisited/failed states count in32; no replacement seed. Require>=24/32 distinct, faithfully restored, physically successful, codec-compatible correction suffixes by the original end. All supervised labels keep strict roundtrip tolerances. If any required supervision cannot be represented, preserve it and stop this branch rather than silently clipping or refitting. If the recovery bank fails admission, no switch to DOSE simply to consume the remaining budget.

RECOVERY2048 uses50% original expert and50% admitted suffixes, same behavior sampler, native loss/scope/precision; carry Adam, new cosine2048/warmup103, peak interface/DiT5e-5/5e-6, as R1. No failed model prefix is positive supervision; no teacher runtime finish in evaluation.

**DOSE alternative:** only when RECOVERY priority is not met, dynamic fit remains<24/32, AND same-input fixed-noise dynamic median J at4096 is<=0.85 of2048, no per-channel dynamic p95 rises>25%, and no increase in invalid decode/gate rejects in fixed DIAG. Continue the same T1 bank, label/processor/behavior weights and scope for at most2048; freeze a new deterministic32768-or-smaller existing-plan continuation selecting exactly16384 actual draws. No new event weighting or harder/easier layout search. Carry Adam with a declared cosine2048/warmup103, peak interface/DiT5e-5/5e-6. This is a finite dose check on NEW task data, not reopening historical MORE/EVENT or an unlimited convergence claim.

If neither gate holds, report the relevant optimization/conditioning/contact gap; do not invent a third learning branch, increase LR, replace the actor or train on DEV. At the ONE chosen branch endpoint, perform one final TRAIN8/DEV16 screening round. No optional branch can run after a candidate has consumed confirmation. For T1 there are at most4 full screening rounds total including1024/2048/4096/one branch.

## 7. First deliverable and two-task extension

Qualification remains exactly R1: same candidate TRAIN>=7/8,DEV>=13/16; freeze weights+processor+statistics+mapping+K4+decode, then CONFIRM_A IDs285300..285331 once, >=26/32 full T1 successes. Full32s, all41samples28..30, true grasp/lift/transfer, raw object placement/static/not-grasped/clearance and collateral-object checks are unchanged. Confirm failures are not corrected by replacing seeds or tuning on results. Confirmation failure ends learned-policy experimentation in this goal; no T2 training.

On PASS, immediately publish/save `PLACE_V1_SOURCE_QUALIFIED` and its separate protected full recovery. Do not let optional archive/ablation/post-training jeopardize it. selected/deployment remainNONE for target hardware. If confirmation is not completed before deadline, status is DEVELOPMENT_ONLY_NOT_CONFIRMED, not usable/qualified.

Only then, and only with adequate remaining measured time before section9 cutoffs, execute the existing SHARED4096 from qualified PLACE_V1: same common processor and interface, task50/50, same native objective and R1 phase LR. T2 teacher banks and paired-layout gates already exist; reuse them. Preserve the pre-shared single T2 DEV16 acquisition baseline. Both tasks always use one model/row/decoder; task ID, phase, object truth and checkpoint dispatch remain forbidden online.

Use at most the original three shared screening rounds (2048/4096/one optional recovery endpoint), and report final available results for BOTH tasks. StageB keeps its existing single validated32-state recovery option; it does not acquire a second DOSE or normalization branch. Require each task TRAIN>=7/8,DEV>=13/16 and paired-both>=12/16, then unique CONFIRM_B IDs285400..285431, each task>=26/32 and paired-both>=24/32. A-confirmation data never become B training/selection data. On B failure or deadline, retain qualified A permanently and report B's actual partial/failed result.

## 8. Efficient execution without changing the science

Before queuing a GPU, ensure admitted data/model are actually readable at the destination and a scoped immutable model-only return path is ready. Use the restored authenticated Corelab path; do not modify remote network services or blame home capacity without evidence. No API keys, addresses or private network credentials are written into public files.

One legal training lease, single owner. H200 is preferred. If a queued H200 has no allocation after20min and an existing legal Corelab GPU can fit the same declared training scope, use it instead AFTER withdrawing the nonrunning duplicate request and logging the choice. Do not interrupt another project or reduce trainable scope/precision silently. If micro4 cannot fit, micro2xaccum4 or micro1xaccum8 may be bound once BEFORE the first update, with the real canary; effective batch8 and per-sample objective stay declared. Do not sweep batch sizes during a healthy run.

Use ordered two-worker native CPU processing with pinned memory and instruction-bound caches. Do not re-encode all data unnecessarily or introduce an unverified frozen-neural-feature cache. Avoid worker/compiler/precision/K/hyperparameter searches. Keep global finite-loss/gradient clipping each update, detailed checks at first8, each128 and checkpoints; no repeated full weight copies every microbatch. Save actual input/noise/time and state/cursor evidence.

H200 training and Corelab evaluation may overlap only on genuinely independent resources. A model view is immutable and must reconstruct exactly; optimizer stays on training host. HF bulk and full public readback are deferred; they are not the feedback transport. Local durable recovery, raw evidence and final model identity are mandatory. Source frames/videos are from actual arrays; no synthetic success footage. Old/unique data, pristine and qualified first-policy files are protected.

## 9. Absolute timetable and measured reserve

All times below are **2026-10-09 Asia/Singapore**:
- On adoption: resolve the known scope through section3; proceed through actual local transfer/canary into useful PLACE updates. Do not spend repeated turns waiting on the same resolved permission.
- 05:30: optional CPU design work ends; review the first-policy funnel and candidate state. No new standalone representation/history engineering project.
- 06:00: latest start of a new32-slot corrective teacher collection. Its preparation/run must fit a45min worker window plus real training/evaluation reserve; otherwise mark NOT_LAUNCHED_TIME.
- 06:30: latest start of an optional T1 extra optimization phase or SHARED primary training. Already healthy admitted phases may continue to the following bound; no policy is reset to meet a clock.
- **08:30: hard stop on starting new optimizer updates**, earlier whenever confirmation reserve requires it. Finish the in-flight atomic update, save full recovery and freeze available candidate.
- **09:45: model/teacher/physics work stops**. Incomplete conditions retain their actual technical/deadline status; no success extrapolation.
- **10:00: deliver local terminal report, loadable artifacts and actual evidence**, and post GitHub receipt when available. External archival failure does not authorize more science past the deadline.

Use measured completed model-episode latency, not CPU teacher speed or an early-rejected rollout, for remaining-time planning. Reserve only the NEXT admitted gate, not all unadmitted branches. After candidate qualification, reserve at least1.25x measured p90 episode duration times remaining conditions plus measured load/save/CPU audit and10min handoff. Use a conservative45min floor for A confirmation and75min for paired B confirmation unless all those steps already have measured evidence supporting a longer reserve. These are scheduling reserves, not promised run times. A candidate qualified earlier is immediately confirmed; no need to wait until08:30. A late qualified T1 is delivered/confirmed before attempting B. Budget and deadline jointly bind, whichever is stricter.

## 10. Cumulative budget and honest terminal statuses

R1+R2 share ONE PI0285 ledger; no reset and no inheritance of unused PI0284 allowances. Completed expert176 starts and all other real work are already charged. Formal caps remain A6144,B6144,total12288 at effective batch8; all completed<=12512 including64canary+32memory+128completed rollback; training technical recoveries<=4. DOSE and RECOVERY consume the SAME A2048 optional allocation. No added simultaneous control arm.

All existing resource ceilings remain:840physical starts,32technical physical replacements,quality32768,K4heads131072,control580000,integrations2900000,RGB580000,dispatch1160000,teacher controls400000,planner4096,GPU32hours,CPU256core-hours,new owned output/cache240GiB. Check actual remaining space/quota or owner-approved capacity, not shared df alone.

With teacher bank already frozen, the conservative remaining recipe is at most A128 model episodes (4xTRAIN8+4xDEV16+CONFIRM32),B224 (acquisition16+3xTRAIN16+3xDEV32+CONFIRM64),128correction physical passes, plus completed176teacher starts =656; adding32technical replacements gives688 below840. No need to spend the difference. Up to16DIAG96 panels and256 other allowed load/retention/diagnostic requests plus352x80 physical model queries is29952, below32768. Every attempted model/physics start is still counted; no uncharged repeated tests or excluded quality failures. Further confirmation, independent scene search or new task is not permitted.

Terminal states distinguish ORIGINAL_CODEC_FAIL_PRESERVED,SCOPED_ADMISSION_WITH_KNOWN_DEV_GAP,NOT_EXECUTED,VALID_QUALITY_FAIL,DEVELOPMENT_ONLY_NOT_CONFIRMED,PLACE_V1_SOURCE_QUALIFIED,SHARED_TWO_TASK_SOURCE_QUALIFIED,DEADLINE_PARTIAL. Teacher100% is not model100%; trainability is not qualification; source qualification is not cross-embodiment/real-time/WBC deployment. Source reports use raw intended/installed/achieved trajectories and object predicates separately. Runtime learned policy never reads teacher truth, task stage, a success-triggered stop or an oracle correction.

Required next owner receipt: fixed R2 authority readback; preserved old FAIL; new scoped admission; actual canary/data destination/lease status; real updates and first model feedback. No recurring approval/monitoring task is created by this protocol. The reviewer did not SSH, execute a neural model, render, simulate or operate a job.

## 11. Evidence actually reviewed

At review SHA: reports/PI0285/preparation/ACTUAL_CODEC_AND_PROGRESS.md; actual_bank/audit/codec-gap/REPORT.json; scripts/pi0285_execution/data.py,codec_check.py,admit.py; current source-bank/input/audit assertions reported in #6064560511 and #6064629367. Official pinned gr00t/data/utils.py confirms TRAIN minmax and inverse clipping. The original strict gate was broader in implementation; this R2 decision explicitly changes admission scope, not the observed mathematics. General preprocessing reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler (clipping held-out values can prevent exact inversion; this is general background, not the implementation used here).

Startup: adopt current owner and the R2 release; record scoped supervised admission while preserving DEV clipping FAIL; continue genuine pretrained PLACE/canary and real feedback; allow only the declared gated extra phase; independently confirm and protect first policy before shared learning; stop by Oct9 10:00SGT with truthful artifacts.
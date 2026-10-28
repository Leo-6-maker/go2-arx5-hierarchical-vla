# PI0268-P0 — Parallel gap audit and preparation of one next fine-tuning intervention

Authority: **PI0268_PARALLEL_GAP_AUDIT_AND_FINETUNE_PREPARATION_GO_V1**
Date: 2026-10-02. Scientific basis snapshot: `7023316d912146e8219b57aae7e938982f3ffee7`.
Status granted by this document: **GO for bounded CPU analysis, manifests, isolated implementation and tests; NO-GO for additional model queries, rendering, physical runs or optimization.**

## 0. Purpose, ownership and concurrency

While the main agent finishes the already frozen PI0267-R1 evaluation and preservation, a helper can reduce the uncertainty about the next intervention. It must not create a second evaluation coordinator or alter any active result, runtime, data, statistics, gate, checkpoint or scientific budget. If no helper is available, the main owner may perform this work between nonblocking orchestration tasks without interrupting evaluation. Neither lane waits for the other's signature.

Use Issue #11 for the new authorization and compact return. Continue the current repository workflow: no new general issue, PR77 pointer, automatic merge, force push, archival deletion or branch proliferation. Add preparation files only in `docs/protocols/PI0268_*`, `reports/PI0268/preparation/` and `scripts/pi0268_preparation/`. Use a separate local checkout/worktree and a separately named remote preparation directory; never sync a moving branch into a healthy PI0267 runtime. Serialize Git writes and preserve concurrent commits. Reading a new head is not permission to overwrite it.

This is not a third PI0267 training arm or a retrospective fix. Preserve its N/R comparison, all original FAILs, V2/V3 outcomes, and formal selected/deployment NONE. The main evaluation retains its original qualification and first-valid records. A result under preparation does not establish a new qualified model.

## 1. Evidence already available; do not rediscover completed work

Read the complete immutable `reports/PI0267/R1/OFFLINE_STAGE_RETURN.md` at the basis snapshot, plus:

- `TRAINING_LABEL_RUNTIME_RECONCILIATION.md`;
- `evidence/MAIN_TRAINING_AUDIT.json` and actual consumed-input/gradient/loss receipts;
- `evidence/OFFLINE_VARIANT_GATE_BREAKDOWN.json`, `CLOSED_RESULTS_PAIRED_ANALYSIS.json`;
- completed-point `VIRTUAL_DECOMPOSITION_donor.json`, `VIRTUAL_DECOMPOSITION_N4096.json` and query-chain receipts;
- `scripts/pi0267_train.py`, `pi0267_virtual_data.py`, `pi0260_r2_codec.py`, `pi0260_r2_inputs.py`, `pi0267_infer_worker.py`, and the actual sealed runtime dependency versions.

Verified published observations at that snapshot:

1. N and R each completed4096 optimizer updates. All2088 fixed offline rows are complete and finite. Data/optimization integrity and reproducible decoding are not evidence of good model quality.
2. R's128 recovery rows have86 original gate rejections. These are exactly the X/Y combined perturbations: variant1=22/22,2=22/22,3=21/21,4=21/21. Z combined variants5/6 have0/21 each. R medians by variants1-6 are390.455,717.908,1136.967,2847.566,19.278,13.484mm. Donor and N have no rejection in any variant. Variant is associated with different fixed windows, so this is not a rotation-only causal experiment.
3. N's completed virtual M1 groups have13/16 recorded successes each, but four tool collisions violate the safety requirement; M3 is0/32. Donor M1 is13/32 overall and M3 is0/32. Every recorded successful M1 episode for donor and N ends outside the original pose bounds when the fixed40s runner continues. These are distinct early-attainment and retention observations, not permission to change the first-hit rule.
4. E0 follows installed positions exactly, but installed and achieved M3 motion never acquires entry. WBC tracking is not necessary to explain these primary failures. E1 and real WBC remain distinct execution conditions.
5. R's1359 cross-host normalized label byte mismatches are disclosed and physically reconcile within the old bounds. That is not proof that normalization is numerically well-scaled for learning, nor proof that those mismatches caused the bad outputs.
6. The actual codec uses unclamped affine normalization with `width=max(q99-q01,1e-6)` and retained OLD statistics. Physical roundtrip and learner-input/target conditioning are different properties. The publication reviewer read this code but did not obtain the actual numeric statistics/arrays in this environment; actual affected feature widths and activation magnitudes are NOT claimed measured here.

Fresh execution state may have advanced. At entry, capture a timestamped snapshot of completed source partitions, audit hashes and remaining main-stage work. Do not poll indefinitely, restart a process, or report a partial R point as a final result. Use only immutable completed files or an atomic source snapshot with a verified before/after identity. Missing data is UNKNOWN, not a reason to fabricate it or block all other tasks.

## 2. The gaps to distinguish

G1 — numerical conditioning: a mathematically reversible normalizer may amplify almost-constant orientation features or relative-action components far beyond the original learning scale. State encoding, target encoding and physical decoding must be examined separately.

G2 — optimization versus generalization: the fixed held-out failures do not show whether R fits its genuinely consumed TRAIN recovery examples. Do not infer overfitting from the absence of a TRAIN model probe.

G3 — recovery support: the existing six variants are `(+/-10mm e_j, +/-0.060rad e_j)` with matched axis and sign. At each nominal anchor, their normalized local six-dimensional pose-displacement matrix has rank3, not6. They do not independently vary translation and rotation. This is not a claim that the entire dataset has rank3, and does not by itself explain failure on the same combined-variant evaluation family.

G4 — closed-loop objective: local H40 fitting, the first0.4s that is executed, entrance acquisition, repeated terminal stability and collision freedom are different requirements. N's aggregate offline recovery median is not a paired gain: the published median paired change is+0.404mm and60/128 rows improve. Preserve paired distributions and task differences.

G5 — transfer scope: the virtual camera/tool domain and narrow perturbation set do not prove cross-embodiment transfer. WBC refusals are not allowed to gate this numerical audit, and mathematical E0/E1 are not two robot embodiments.

## 3. Work package A — actual codec support and numeric-gain report, CPU only

Use the exact existing raw TRAIN corpus and the exact numerical statistics actually passed through `build(...)/set_statistics`, not merely a same-named JSON copied into a checkpoint. Bind byte hashes and parsed structure; explicitly distinguish archival config statistics from actual processor statistics if they differ. Do not alter either.

Stream the existing24-family corpus and actual consumed N/R logs once; cache small derived arrays in the preparation directory. Reuse existing full-input and label-replay identities. Use the pinned CPU processor in an isolated process with CUDA hidden and network/model loading disabled. No new learned forward or backward is permitted.

Produce, by state/action/relative-action, feature and horizon index:

- actual q01/q99/min/max/width/floor entries and affine forward gain2/width, inverse gainwidth/2;
- raw physical range and transformed min/max/p50/p95/p99, fractions outside[-1,1],10 and1000 in magnitude, nonfinite/clipping counts; these bands are diagnostics, not new admission thresholds;
- separate nominal, each of six combined variants, task, command and arclength stage;
- actual tensors at raw state, Gram-Schmidt/relative-pose conversion, affine normalization, padding/masking, collator and final documented dtype boundary;
- state versus action scale and native-loss target scale; padded coordinates must not be counted as physical DoFs;
- CPU FP32 and BF16 cast roundtrip error on already saved numeric tensors, clearly labelled arithmetic analysis, not model-precision ablation;
- sensitivity to small physically valid perturbations through the frozen codec, using fixed magnitudes1e-7 and1e-6 in metres/radians, never searching magnitudes for a preferred conclusion. These are numeric function probes, not newly admitted training data;
- if a nonlinear conversion is ill-conditioned, show the actual norm/denominator and affected rows instead of blaming the affine map alone.

For held-out records already opened by the main evaluation, descriptive range comparison is allowed, but no bounds, epsilon, loss coefficient or training sample selection may be fitted to them. Keep TRAIN-derived analysis and retrospective EVAL diagnostics in separately marked tables. Protected TEST/DEV-F remain inaccessible.

A tiny supplied constructed example shows that a yaw-only toy rotation distribution can encode0.06rad roll/pitch at about1.2e5 magnitude while decoding accurately. Its bounds are NOT the project's measured statistics. Run it as a sanity fixture; actual project measurements are the deliverable. Do not claim that a toy proof establishes the cause of R's failures.

Deliver `CODEC_CONDITIONING_REPORT.md`, feature-level CSV and a source-bound JSON. Do not change the live codec, introduce clipping, or silently increase epsilon. Such changes define a new policy contract and require a subsequent scientific launch decision.

## 4. Work package B — exposure and saved-gradient analysis, CPU only

From ACTUAL sample logs, not schedule length, count distinct image/state/label/language identities, repetitions and coverage for both arms. Stratify by task, command, family, anchor, recovery variant and template. Distinguish available but never consumed proposals from seen TRAIN samples. Report joint/conditional coverage, not only marginal balance. Quantify terminal-anchor and near-terminal recovery exposure separately from intermediate anchors.

Join existing native-loss, full-S4 physical-loss, optimizer/gradient records to the actual update/microbatch and variant. Examine early/middle/late records, step-to-step changes, and clipping indicators only where the underlying values were saved. Report unsaved per-module activations/gradient attribution as UNOBSERVED; do not reload models to manufacture them during preparation. Zero fresh gradient queries.

From saved inference traces identify where large output magnitude first appears: raw initial noise, each stored S4 state/velocity, relative normalized output, inverse-normalized representation, state-anchor composition, final physical trajectory. Compare donor/N/R matched rows without filtering out REJECT. Save first-point position/orientation mismatch and per-segment jumps, not only whole-H40 RMSE. The operator stages may localize amplification; they do not establish a neural causal mechanism without an intervention.

Do not choose the worst few cases as the only evidence. Report full strata first, then fixed representative IDs selected by metadata ordering. Training loss is neither task success nor a substitute for a TRAIN inference test.

Deliver `TRAIN_EXPOSURE_AND_OPTIMIZATION.md`, consumed-condition index and the compact existing-trace decomposition. Reuse the newly completed variant/paired analyses instead of recreating them verbatim.

## 5. Work package C — executed-prefix, terminal and safety analysis of closed points

Use completed donor/N virtual point archives now; append R only after its full fixed point is closed. Never modify existing results, V3/V2 or first-hit success definitions. Use the same existing source state/query/trajectory identities.

For each replan, compute from SAVED predictions and achieved state:

    first_point_error = predicted_p(0) - achieved_p(now)
    prefix_motion = predicted_p(0.4) - predicted_p(0)
    commanded_total = predicted_p(0.4) - achieved_p(now)

Verify `commanded_total=first_point_error+prefix_motion` at the original interpolation convention. Include SO3 first-point and relative-prefix quantities separately without treating rotation vectors as globally additive.

Report early approach progress, requested/installed/achieved entry separately; prefix error versus whole-H40 error; target-distance change; time of first qualifying hold and first subsequent exit; accumulated near-goal drift; dangerous collision time and precursor references. Match same-time reference and achieved data; never subtract minima at different times.

Compare recorded states to nominal/recovery TRAIN support as descriptive nearest-neighbour distances in a declared task-space metric, not as proof of visual equivalence. A high distance does not alone prove OOD caused the failure. Do not use hidden teacher projection or target geometry to repair any model execution. Task geometry is for offline diagnosis only.

No new success threshold, stop-on-success patch, return-to-goal controller, smoothing, or removal of late failures. If future terminal retention or prefix supervision is proposed, retain it as an explicit new experiment with its own negative controls. Do not quietly retrofit it onto PI0267.

Deliver `CLOSED_LOOP_GAP_REPORT.md` with complete-point denominators and a small number of fixed-array plots; media polish must not delay preparation. No new renders or frame requests.

## 6. Work package D — prepare the next bounded diagnostics; DO NOT RUN them

Prepare manifests and executable DRY-RUN entry points that exit before model loading unless a future authorization ID and actual approved run card are supplied. A command-line flag alone is not scientific permission. The following are proposals for a subsequent release, not a new query budget granted now.

D1: genuine consumed-TRAIN fit panel. Select four distinct actually consumed R inputs per task/command/variant cell (2 tasks x2 commands x7 nominal/combined variants x4=112), preferably one from each arclength quartile using metadata-only deterministic ordering. Nominal examples should be common replay consumed by both arms when available. Every selected item retains its exact training language, raw labels, processor input identity, actual consumption unit, family/anchor and seen-by-N/R flags. If a stratum is absent, report it; never label unconsumed examples as seen or borrow held-out examples. Score donor/N/R at two prospectively fixed actual-noise packets: maximum672 queries. This would distinguish fit failure from transfer failure; it is not a new generalization score.

D2: physically consistent factor-separation panel on TRAIN, not EVAL. Fix the first two TRAIN families in source order, both tasks/commands and existing anchor indices4 and11:16 nominal base conditions. For each, prepare nominal plus six position-only, six world-left orientation-only and six existing combined perturbations:19 variants/condition,304 maximum conditions. Perturbations remain10mm and0.060rad, with unchanged tool/camera geometry and teacher. Fresh matching RGB and each state's own correct label would be required when this is later authorized. Do not reuse nominal RGB with altered wrist pose/state. Invalid visibility/geometry/label cases remain visible and are not replaced. Three models and one shared actual-noise packet per base condition yield at most912 queries; combined D1/D2 proposed ceiling1584 planned,1664 attempts/6656 S4 heads. Rendering ceiling304 fresh frame pairs. All these counts are NOT_AUTHORIZED in this P0.

Physically consistent position-only/rotation-only interventions still change their dependent images appropriately; they distinguish pose-factor changes, not isolate vision from proprioception. Swapping inconsistent state/image pairs is not a valid task test. No model fitting, new noise generation for query execution, rendering or changed poses is performed now.

## 7. Work package E — prepare ONE selected next-training design, no launch

Write `HYPOTHESIS_MATRIX.md` with each proposed cause's supporting evidence, counterevidence, missing observation and falsifiable next test. Rank hypotheses only after packagesA-C; do not force a unique cause. The TRAIN panel has not executed during this P0, so missing fit evidence remains missing.

The preferred minimal design, conditional on measured rotational-scaling pathology, is a NORMALIZATION-ONLY comparison:

- common fixed original PI0264 L2048 donor (`e4290e6ca387abd2de4337b5f12b9c72a5f85c99`, `arms/PI0264/L/checkpoint-2048`), not an unqualified R chosen after its poor result;
- identical raw nominal/recovery examples and visitation, labels, noise packets, full-S4 loss, trainable scope, BF16/FP32 boundaries, optimizer and update budget;
- control retains the exact old numerical contract; treatment changes only the smallest evidenced problematic normalization block(s), with TRAIN-only or representation-theoretical bounds fixed before a run;
- fixed[-1,1] bounds are a defensible candidate ONLY for components verified to be an actual orthonormal rotation6D representation. Do not apply them to XYZ, arbitrary deltas, or a different relative-pose representation by assumption;
- state normalization and action normalization must be itemized. Changing both is one combined normalization policy treatment, NOT proof of a state-only or action-only mechanism;
- no new clipping/hidden target fields, no label retiming, no topology/adapter/loss/teacher changes. Native loss changes induced by target scaling are part of the declared treatment, not falsely called identical effective loss weighting;
- same starting weights under different processors do NOT define identical initial policies. Prepare physical-metric step0 measurements for each effective contract and clearly separate them from weight-learning gains. An inference-only normalizer swap on already trained R is not an automatically valid repair or releasable checkpoint.

If scaling is well-behaved and genuine seen-TRAIN inference later fits while factor separation fails, prepare instead a DATA-COVERAGE-ONLY comparison: same stable contract and nominal replay, existing coupled recovery versus independently varied pose recovery at matched update count. Do not train both proposed directions. If seen TRAIN is poor even with a sound contract, propose one bounded fit/optimization experiment rather than blindly expanding scenes or launching DAgger. If the dominant issue is terminal/prefix behavior, formulate it separately, retaining current score semantics and not mixing it with a normalization change.

For the selected draft, prepare complete dependency manifests, one-factor diff, sample counts, exact save/resume logic, prior-domain retention panel, and a prospective evaluation/stop plan. A tentative two-arm2048-update comparison (four complete1024/2048 Trainers) may be costed, but all optimizer/canary/query budgets remain unallocated. Keep `launch_allowed=false`, `approval_status=NOT_AUTHORIZED` and `missing_authorization` explicit. Do not create a PBS submission or make a callable training script auto-launch on successful CPU tests.

Do not turn the eight already examined NEW_SIM_EVAL families back into an untouched TEST. They are now development evidence for subsequent choices. Do not fit new normalization, choose windows or tune thresholds from those values. Any later independent-generalization claim needs separately preregistered fresh conditions. No new scene family or second robot is instantiated in this P0.

## 8. CPU tests, resources and deliverables

Preparation permissions: small source/config reads; one bounded stream over existing complete raw TRAIN data, consumed logs and completed evaluation partitions; source-bound arithmetic/codec replays; deterministic metadata selection; code authoring and CPU fixtures in isolated directories. Allowed checkpoints: read existing small manifests/configs only; no multi-GB weight loads or copying complete Trainer trees. No network-wide HF reread or private asset publication.

Default resource ceiling:4 CPU threads,16GiB RSS,10GiB working data and2GiB derived output, one analysis process per lane. Prefer PC; use an idle permitted CPU allocation on Corelab only with nice/ionice where supported. Hide CUDA, set thread caps, and do not allocate the active evaluator's GPU. Main evaluator RAM/IO/space needs take priority; a specific busy shard can be skipped and reported while other tasks continue. Do not migrate large archives solely to comply with this document. If source data already exceeds the working cap, stream it in place rather than copy.

Test with synthetic fixtures: nearly constant normalizer dimensions; a large but exactly invertible affine value; raw versus processed equality; coupled rank3 versus independent rank6 local pose support; original gate rejection retained; wrong unit/frame/state-label negative cases; missing consumed-stratum and duplicate input rejection; held-out split rejection; launch fails without new authority; no accidental model/renderer/PBS calls; partial-point rejection; active input files unchanged. Static/dummy-model tests must be called static/dummy tests, never a real canary.

No new numerical action output, model forward/backward, optimizer update, physics step, WBC call, renderer call, training data admission, API request to external models, monitor/email, broad storage cleanup or merge is authorized. Counterfactual manifests and code are preparation, not scientific results. Do not hide any computation under a CPU label: loading and querying a neural policy on CPU is also prohibited here.

Deliver once completed:

1. `SOURCE_SNAPSHOT.json` with finalized input identities, concurrency boundaries, actual counters and known omissions.
2. codec-conditioning tables; actual exposure/optimization and existing-trace analysis; closed-point prefix/retention/safety report.
3. proposed112-row fit and304-condition factor panels, metadata-level validation and exact unallocated costs.
4. one recommended treatment with a justified alternative/rejection rationale; executable guarded draft and regression tests; `NEXT_RUN_DRAFT.json` marked NOT_AUTHORIZED.
5. one concise Issue #11 preparation return linking immutable files. No routine historical PR pointer. No need to wait for the full main study to deliver these completed preparation artifacts.

Terminal: **PI0268_PREPARATION_COMPLETE_AWAITING_SEPARATE_SCIENTIFIC_LAUNCH**. This is completion of the authorized preparation task, not model improvement, the end of PI0267, or permission to train. The main owner independently finishes current evaluations. The next launch decision uses the complete main outcome plus these bounded diagnostics; no automatic training from partially observed results.

## 9. Publication review scope

The publication review fetched current GitHub receipts/source, including the new axis-specific R failure and completed N results. It ran only the supplied mathematical toy conditioning and local-support-rank checks. It did not read actual numeric statistics through the remote production processor, replay original raw model tensors, SSH to HPC, render, load weights or train. The constructed fixture must remain labelled as such. Plan publication and actual helper execution are separate events.

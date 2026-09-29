# PI0260-R2 — Dataset-driven admission to the first matched Motion2 model experiment

Status at publication: AUTHORIZED_SCIENTIFIC_AMENDMENT_NOT_EXECUTED.

## 0. Authority, objective and precedence

Execution authority: `PI0260_R2_DATASET_DRIVEN_MOTION2_TRAINING_GO_V1`.

One accountable lead owns the entire experiment. An optional helper is not an approval dependency. Retain the same-owner saved-array verification and autonomous in-scope engineering repair rules of PI0260-R1.

Read with these immutable sources:

- Scientific parent: `90cbb926a842f08c3f9a78756d7a24e0d7a63f78`, `docs/protocols/PI0260_FEASIBLE_OBSERVABLE_MOTION2_TO_TRAINING_V1.md`, Issue #11 comment5893504272.
- Single-owner amendment: `1f27d9c14ce0a91518930a8d8a2f6f0ffc2dd468`, `docs/protocols/PI0260_R1_SINGLE_OWNER_END_TO_END_EXECUTION_V1.md`, comment5893778973.
- Actual stopped source/result: `10bbef10a00c43a3197a0f6d9fb07a3469596c53`, `reports/PI0260/DEV_RESULT.md`, comment5894251286.
- Source reviewed: `scripts/pi0260_design.py`, actual eligibility-runner diff at `cc191f3675ac680e30fdcca1d2139c05eb7fcd8b`, and the old `scripts/pi0255_motion3.py` split names.

This is a prospective change to DATASET-SIZE ADMISSION, RESERVED-PROPOSAL EXECUTION and explicit metadata exclusion handling. It is NOT a change to trajectory quality, camera/marker design, IK/reference/collision rules, model recipe or model evaluation thresholds. It does not relabel the failed PI0260 DEV program.

The old `PI0260_ELIGIBLE_SCENE_SHORTFALL` is valid and remains immutable. The new question is: can the same generator supply a smaller but real, family-separated two-behavior learning dataset from the untouched proposal strata, allowing a matched Base/G pilot without first repairing all old DEV cases?

Actual dataset quality remains required. Actual OLD DEV count6 is no longer a prerequisite for checking prospective TRAIN/VAL. Four old eligible DEV families are diagnostic fixtures, not a replacement training or validation set. The goal is actual Base6000/G6000 plus the fixed1760-tuple evaluation and archival, not another camera report.

This amendment supersedes parent requirements for6 eligible DEV families, exactly40/10 selected TRAIN/VAL families, old64/16 usable-sequence and24/6 pair minima, and the old family/sequence/window budget limits ONLY where replaced below. R1's single-owner governance remains. All other scientific constraints apply unchanged.

## 1. What is established, and what is not

The executed DEV source reports:

- H0 first-rig observation calibration PASS;84 prior valid calibration pairs.
-20 proposed DEV families,80 actual sequence attempts;45 complete geometry sequences: M1 29/40, M3 16/40.
-35 sequence failures at bounded `IK_RESIDUAL`;12 families lack all-four complete geometry.
-8 families have all4 complete references and raw-gate eligibility;4 fail the unchanged M3 endpoint visibility requirement during motion. Their visibility fractions are approximately89.375–91.875%, below95%.
-4/20 jointly eligible families:000/004/009/013. No model query, update, physics or WBC call.
- Archived saved-array replay:11,237 FK/collision states,2,134 image pairs,4,192 raw-gate rows including3,520 active windows. This publication reads those source reports; it does not claim to have rerun their arrays or inspected the Windows video.

Interpretation:

1. Bounded IK failure is not proof that no physical or alternative kinematic solution exists. Do not diagnose a universal WBC or GR00T failure.
2. H0 sampled-pose visibility did not establish dense full-trajectory visibility. Keep the95%/mandatory-state checks, do not lower them to90% because four observed failures would then become easier to accept.
3. Joint family eligibility compounds four-sequence requirements. Descriptive yield4/20 is a warning about original40-of100 training and10-of40 validation targets, not an independent probability estimate for the untouched strata.
4. The old six-family DEV threshold is an experiment-design quota, not a proof that a dataset with four correct diagnostic fixtures must be unusable. The actual data used by learning must now decide admission.
5. There is a narrow exclusion-code issue to check before using new proposals: PI0260's `reserved = split.startswith('TEST')` does not itself recognize the older `MT-TEST_COMPOSITION`/`MT-TEST_GEOMETRY` labels. This is a demonstrated string-predicate gap, NOT proof that any selected scene actually overlaps a protected test geometry; another registry may already exclude the same numeric geometry. Verify the union from metadata, do not allege unobserved training leakage.

No new loss, new body camera, larger cue pack, alternate solver, easier reference or WBC experiment is needed just to try this dataset-driven continuation.

## 2. S0 — Preserve old work and freeze the amended registry

Refresh the current Issue #11 ledger, exact source tree, Corelab owners/processes and lawful Hopper scheduler state. Adopt valid work; do not restart a healthy job or reset counters. No new ordinary CPU test belongs on an H200 allocation.

Use a versioned R2 execution overlay/child card and separate new output namespace. Do NOT edit old `registry.json`, H0, DEV outcomes or their manifests in place. Existing physical geometry/source IDs remain traceable. Bind old source, old H0, R1 workflow authority and this R2 authority separately.

Preserve the selected first rig `[.65,0,.75]` BASE, its actual frozen extrinsic/look-at/projection,96x96 streams, wrist, cue pack, reference/solver and collision configuration. Do not recompute a look-at centroid after exclusions or after selecting successful scenes. Reuse H0 as evidence for the exact unchanged design, not as a substitute for full-trajectory checks.

### 2.1 Metadata-only exposure/test reconciliation

Before any new IK/render/model call:

- Read all applicable prior task-registry split metadata and actual executed IDs. Normalize documented test aliases, including `TEST`, `TEST_*`, `MT-TEST_*` and `MT_TEST_*`. Add unit cases for the concrete historic labels. Unknown aliases must be resolved from their registry definition, not silently treated as TRAIN.
- Compare numeric full geometry/initial-state identity as required by the parent; exclude protected TEST and previously inspected development/calibration geometry even under a color or ID alias. No TEST label, trajectory, image, expert or model computation is authorized.
- Report the delta between the old exclusion union and the correctly normalized union. If the delta is empty, say so and continue. If a candidate overlaps, mark it `EXCLUDED_PROTECTED_GEOMETRY`, preserve its original slot, and do not backfill from another stratum.
- All20 PI0260 DEV families stay development-exposed and are ineligible for either new TRAIN or KIN-VAL, including the four successes. Do not call old DEV failure remediation unseen validation.
- If any past rendered DEV geometry overlaps protected test metadata, preserve a contamination notice for future test claims. Do not erase the old outcome or execute that test again.

This check changes exclusion correctness only, not geometric support. It must not become an open-ended all-history audit.

### 2.2 Release only the already listed UNUSED proposal slots

The stopped registry reports204 remaining parent geometries after its original exclusions:20DEV,100TRAIN,40KIN-VAL,44 `UNUSED_RESERVED`. Verify this exact source registry; the new normalized exclusions may reduce the executable counts but may not expand the grid.

Freeze these lists BEFORE evaluating their outcomes:

- TRAIN: original proposal indices20..119, followed by indices160..191. Maximum132 families.
- KIN-VAL: original indices120..159, followed by indices192..203. Maximum52 families.
- DEV indices0..19: no new attempt, diagnostic evidence only.

Indices160..203 are released UNUSED candidates, not TEST and not previously evaluated successful scenes. Only metadata ownership changes for these rows. Original TRAIN/VAL membership is never transferred. Keep original geometry and colors. For the newly released VAL rows, explicitly set the reserved VAL language templates2/3 using the same deterministic index/command rule BEFORE any execution; original untouched rows retain their instruction identities. Save old/new row bindings.

If metadata reconciliation removes a slot, skip it with the exclusion reason; do not take a different index. If a purported unused slot was actually executed, classify its exposure and preserve the record before admission; never silently call it unseen. No fourth stratum, new grid, new random seed search or borrowing from TEST.

## 3. S1 — Check actual future datasets, not the old DEV quota

The concrete routing change is: H0 valid + R2 registry/exclusion checks PASS admits prospective TRAIN/VAL collection. The old `DEV_ACCEPTED.status == PASS` check and old six-family quota must not block it. Do not forge `DEV_ACCEPTED=PASS`; emit `R2_PRODUCTION_ADMITTED_FROM_EXISTING_DIAGNOSTIC_EVIDENCE` with the old DEV FAIL still attached.

Keep the original order within each list. Select the first at most40 jointly eligible TRAIN families and the first at most10 jointly eligible KIN-VAL families. Stop evaluating further entries when that stratum reaches its target. A list may also end with fewer eligible families; that is not automatically the whole-goal STOP. Save the actual yield for S2.

Every proposed family still requires both M1 commands and both M3-v2 commands. All four complete trajectories must meet the parent's geometry, hard joint limits, velocity, full-link/obstacle collision, reference semantics, raw SafetyGate, observed cue and phase-coverage rules. No M1-only family is used to conceal an M3 failure. No incomplete accepted prefix, missing suffix interpolation or per-sample reference repair becomes a training example.

Keep strict2mm/.01rad IK residual bounds, the same DLS/last-accepted continuation, fixed time law, scene/reference geometry, endpoint/axis pixel thresholds,95% active-window cue and raw-gate coverage, mandatory discriminative states and all motion quartiles. All failures stay in their original denominators.

Perform complete cheap geometry and raw-gate checks before rendering. Reuse identical-state images within a family with provenance; preserve RGB/masks/depth, actual FK/state/time and full camera transforms. No hidden dynamics in constructors or rendering.

In parallel with ordinary CPU processing, summarize the already saved35 first IK failure records and four visibility failures: first failing time/phase, residual, joint margins, last accepted state and recorded pixel/depth evidence. Missing a diagnostic field is reported, not invented. This summary cannot demand a new solver or camera sweep and does not gate otherwise correct production.

## 4. S2 — Variable-size but fixed-rule dataset admission

This pilot has a TARGET size, a predeclared MINIMUM usable size, and unchanged per-example quality:

| Complete joint families | Target | Minimum AFTER actual codec checks |
|---|---:|---:|
| TRAIN |40|16|
| KIN-VAL |10|6|

A complete post-codec family means all four sequences remain usable under the original95% active-window and mandatory-state/quartile rules after actual raw->processor->inverse decode->gate checks. Do not count color aliases as different geometric families. Preserve all rejected family/sequence/window records and the reasons.

Thus the minimum is32 usable TRAIN sequences and12 usable VAL sequences PER behavior, with16/6 complete opposing-command pairs respectively. These replace the old quantity minima explicitly; no pixel, pose, collision, gate or model-quality threshold is relaxed. Minimum16TRAIN families supports the fixed32 distinct TRAIN-sequence evaluation windows; minimum6VAL families supports the fixed six paired semantic conditions. These minima are a starting pilot design, NOT evidence of sufficient statistical power or generalization.

Fit the shared TRAIN-only statistics ONCE on the selected pre-codec raw-qualified TRAIN set, then apply the actual codec. Post-codec invalid families are excluded from BOTH task training sets; no refill, no re-fitting statistics or alternate selection because of that exclusion. Report the statistics-source population separately from the final admitted population.

Admission result:

-40 complete TRAIN and10 complete VAL -> `STANDARD_40_10`.
-At least16 complete TRAIN and6 complete VAL but below either target -> `COMPACT_<ntrain>_<nval>`; training is AUTOMATICALLY authorized after integrity/resources, without another PI reply.
-Below16TRAIN or6VAL -> `R2_ACTUAL_DATASET_SHORTFALL`; no training, report the actual list and dataset evidence. No adaptive fallback below these minima or another round of unseen sample additions.

Include ALL admitted complete families selected by the fixed order, up to40/10. Do not discard extra valid families to manufacture a smaller/easier cohort. Dataset size/tier is frozen before any model quality result, shared by Base and G. Report total proposed/evaluated/geometry-pass/raw-gate-pass/visible/codec-pass/selected counts. A conditional curated domain is not100% success on the unfiltered proposal space.

The lead owns H1 read-only replay, actual source-specific QA and natural-image discernibility review; helper availability is not a condition. Reuse matching prior checks, not producer-PASS fields. Use `SAME_OWNER_READONLY_REPLAY` and disclose shared low-level math/collision dependencies. No cosmetic video signoff is required.

## 5. S3 — Processor, paired inputs and real recovery tests

New dataset-bound identity: `PI0260_R2_C_KIN_MOTION2_V1`. The new identity reflects the actually admitted population, not a new transform algorithm.

Preserve the parent affine q01/q99 convention,1e-6 range floor, no percentile clamp, row-packed rotations, dense-relative/world/TCP conversions,10 active dimensions in132,H40,.05s and neutral unexecuted gripper.5. Deterministic full-view image processing remains; no random crop/flip/color jitter. Actual physical auxiliary losses and inference decode must use the SAME statistics and transform identity.

Actual label round-trip: position<=1e-5m, rotation<=1e-4rad, exact mask/time/frame meaning. Only body/wrist RGB, current correctly converted FK StateV1 and natural language enter the policy. No target XYZ, source/task/phase ID, segmentation/depth, expert future, success or time field may leak into conditioning.

Freeze evaluation IDs and actual two-noise packets before first quality queries. Freeze one6000-row TRAIN source/visit plan under the original local seed26003 with exactly3000M1 and3000M3 updates, balanced commands, uniform admitted family/sequence then phase/window. Both models consume the same actual processing and native/random inputs; detect mismatch BEFORE the affected update. The actual counts differ with the dataset tier and must be published.

Keep the parent real-model recovery test for EACH initializer: continuous8 versus4+full Trainer save+fresh process restore+4 using the6000-step scheduler.32 discarded updates total; main arms start afresh from original initializers. Use predeclared exact/numerical comparisons from the runtime card, no after-the-fact tolerance relaxation. A CPU-only stand-in does not establish actual Trainer recovery.

## 6. S4 — The two matched6000 training runs are already authorized

Initializers:

- Base: `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`.
- G: `leoyuer/pi0246-gate-pass-campaign@efecea485ae77000599d19171b02491f8d514f42`, `arms/PI0252/r1-speed-margin-20260928/G/checkpoint-2000`.
- Existing Cosmos/tokenizer lineage: `9ce19a195e423419c349abfc86fd07178b230561`.

Reuse actual pinned header/embodiment reconciliation. No latest model substitution. Step0 under the new common processor is measured; historical G qualification is not transferred by name.

Both arms:6000 canonical updates; fresh AdamW LR1e-5,betas(.9,.999),eps1e-8,WD1e-5,gradnorm1,batch1,accumulation1,warmup10,cosine6000. Retain BF16 frozen backbone and FP32 trainable projector/diffusion scope. Objective: native flow matching +.4 physical position +.1 physical increment +.1 physical rotation +.1 peak-speed margin through the original fullS4 generation. Speed margin1.8m/s; no new losses, optimizer search, backbone unfreezing or post-processing to force acceptance.

Save full1000/3000/6000 Trainers, rolling fixed-HF archives and a real recoverable source. Onlystep0/final6000 are quality evaluated. Sequential arms are permitted and preferred for simple ownership/storage; the first arm's scores cannot change the second. One arm's valid quality FAIL does not veto the other preapproved arm. Shared integrity failure does block affected descendants. G's historical adaptation remains extra cost; this is equal NEW updates, not equal total historical compute.

Do not stop at DATA_READY or a queued job. Once S2-S3 and resources genuinely pass, continue both model arms and the fixed screen without asking for another routine GO.

## 7. S5 — Fixed1760-tuple evaluation and interpretation

Preserve the parent four model states (Base/G step0/final) x2 noises. Per behavior:

-32 TRAIN windows from distinct sequences: possible even at16 admitted families because each behavior has32 command sequences.
-16 KIN-VAL windows. When fewer than16 unique sequences exist, take one metadata-ordered, phase-balanced window from each sequence first, then take the remaining nonduplicate windows round-robin. With6 families there are12 sequences, so four additional windows are correlated repeats, NOT four additional independent episodes. Freeze the rule before model scores.
-12 semantic inputs from six distinct paired VAL families with the same actual observation/state and two opposite instructions.
-12 fixed meaning-equivalent paraphrase cases.
-6 empty-language cases.

Add64 historical O64 inputs under the new common processor. Planned total remains `4*2*(2*78+64)=1760` logical tuples. Exact duplicate tuples can be reused and reported; no additional score-driven sampling. Expert-defined discriminative H40 must exist before evaluating language effects. A non-discriminative common prefix is not a model failure and cannot be silently removed to shrink denominators. If the fixed semantic set cannot be constructed, record actual evaluation-readiness BLOCKED before main training; no hidden future phase variable is added.

Report position/SO3 fit, first jumps, speed/gate, correct versus opposite-reference margin, task/direction predicates, all invalid outputs, each task/noise, macro/worst task and O64 change from newstep0. Aggregate uncertainty by geometric family, not by treating repeated windows as independent scenes. KIN-VAL is development evaluation in this explicitly conditioned synthetic domain, not broad unseen physical-task generalization.

Research-quality thresholds are unchanged: EACH behavior/noise VAL group medianXYZ<=.020m,p95XYZ<=.035m,medianSO3<=.050rad,zeroREJECT; at least10/12 correct semantic and10/12 equivalent-paraphrase cases using fixed denominators. No target hit alone replaces fit/gate, and no comparison video replaces model metrics. Save and CPU-recompute raw predictions, actual noise, decode/gate and semantic outcomes. Scientific FAIL may be the result of a complete experiment.

Finish a compact results index/video from already computed arrays using PI0257-v2. Clearly mark synthetic reference/offline prediction. No new query or physical step for media. `run-episode` remains GUARD_STUB_ONLY.

## 8. Implementation and routing acceptance cases

The lead must apply the small actual executor change; this published document is not a code patch or proof tests passed. Reuse the existing runtime, FK/FCL, renderer, codec and review tools, not a new service/orchestrator.

Required CPU tests before new collection/admission:

1. Old DEV4/20 FAIL remains unchanged, but valid H0+R2 registry permits prospective production.
2. Correct TRAIN>=16 andVAL>=6 complete post-codec families admits both arms without Agent V; incomplete/invalid data does not.
3.15TRAIN or5VAL complete families blocks, irrespective of a high producer summary score.
4. More than minima but below40/10 is a truthful COMPACT tier, not false STANDARD or an automatic STOP.
5. OldTEST, MT-TEST_COMPOSITION andMT-TEST_GEOMETRY aliases all enter numeric protected exclusions; union overlap is checked independently from string classification.
6. UNUSED slots cannot migrate between prescribed destinations or replace an invalid VAL output. OldDEV never enters fitting or validation.
7. New VAL-reserve language template mapping is correct; task/source IDs stay out of model conditioning.
8. Failed geometry/visibility/codec stays failed with original counters; no post-codec refill or statistics refit.
9. Variable-size cohorts construct all1760 logical evaluation slots with documented correlated windows, not counterfeit independent counts.
10. Existing valid arrays/jobs are adopted once; hash change or duplicate owner blocks conflicting execution. Helper absence is not a correctness blocker.

Pure engineering defects within the defined algorithm may be repaired, narrowly tested and exactly resumed without arbitrary retry-count limits. A source-proven bug that changes data semantics requires quarantining affected outputs and explicit versions; do not silently call fresh altered data an exact replay. No scientific redesign under the label of a recovery.

## 9. Cumulative resources and specific budget amendment

This is the SAME PI0260 family with a prospective data allocation extension, not a fresh grant of model training. First reconcile actual counters; the reported stopped usage is84calibration pairs,80sequence attempts,11,399IK solver calls,2,134production pairs,3,520active raw windows,0model updates/queries.

| Category | R2 cumulative limit including existing PI0260 work |
|---|---:|
| OldDEV families |20 already executed, no new retries|
| New prospective families |up to184:132TRAIN+52VAL, fewer after protected exclusions|
| Total distinct family proposals |204 maximum|
| Kinematic attempts |80 old +736 possible new +24 strictly no-valid-result technical =840 maximum|
| Additional calibration render pairs |0; keep existing84 and selected rig|
| Active H40 data windows |90000 maximum, raised from60000|
| Production render pairs |120000 unchanged, minus existing use|
| Main model updates |12000 unchanged:6000 per arm|
| All completed model updates incl.32canaries and64recovery reserve |12096 unchanged|
| Raw training action-head calls |109120 unchanged|
| Planned offline logical tuples |1760 unchanged|
| Offline attempts / inference heads incl. strict reserve |1824 /7296 unchanged|
| WBC, physics integration, hardware, TEST, oldDEV/F |0|

The extra44family slots add at most176sequence attempts to the prior664 ceiling. They do not add an extra model arm, a new seed or extra optimizer steps. The90000 active-window ceiling covers the old3520 plus at most184*440 new active raw windows (84480 at the declared M1/M3 timing); unused allowance is not permission for another population. Record all geometry states, solver calls, recompute work and partial failed attempts. Do not continue drawing after a target is reached.

PC is the sole coordinator. Corelab handles CPU/geometry, zero-integration prescribed-pose rendering and no-gradient evaluation. Hopper handles admitted real-model canaries/training on a compatible lawful GPU allocation. Do not pin H200 by host name or debug ordinary Python/IO on a GPU node. No duplicate qsub, killing unrelated jobs, peer wait, large-scale cleanup, mail or monitor.

Preserve G and current complete recovery/evidence sources. Use compact task-local arrays and fixed-version HF public archival with truthful anonymous readback scope. No private URDF/mesh/WBC/tokenizer or credentials are published. Separate df from quota; quotaUNKNOWN remains UNKNOWN. Use inherited next-write overlap plus10GiB reserve, not a perpetual100GiB-free requirement. Reuse unchanged archived checks instead of rehashing all old files for each stage.

## 10. Completion and immediately executable lead instruction

Success of EXECUTION requires actual accepted TRAIN/VAL with its size tier, real codec/input/recovery evidence, BOTH6000 complete model endpoints, all fixed evaluation results and saved-array QA, durable archive and a readable results index.

Terminal: `PI0260_R2_MOTION2_MATCHED_PILOT_COMPLETE — STOP_FOR_PHYSICAL_TRANSFER_ADMISSION`, with separate model-quality PASS/FAIL. A data-only outcome, one model arm or missing evaluation is not complete. Possible blockers include protected-split ambiguity, actual dataset shortfall below16/6, codec/input/resume integrity, resource/raw-budget exhaustion. `selected_checkpoint=NONE` and `deployment_selected_checkpoint=NONE` remain; any candidate is research-only.

Lead instruction: preserve the old4/20 scientific failure; freeze the R2 metadata overlay and protected split reconciliation; reuse exact H0 without new camera/IK settings; inspect original untouched TRAIN/VAL plus their preassigned UNUSED extensions; keep original per-example validity and reject incomplete families; use the predeclared STANDARD/COMPACT dataset rule; once actual minimum complete data, evaluation-readiness, processor, recovery and resources pass, finish BOTH Base/G6000 and fixed1760 evaluation autonomously. No second-agent signoff, oldsettling, M2 fix or six-family oldDEV redo is needed. Do not keep designing new infrastructure or expanding the hypothesis tree instead of conducting the first credible matched two-behavior model experiment.

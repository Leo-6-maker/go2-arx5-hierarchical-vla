# PI0270 — Reliable task-space feedback through bounded learner-state recovery

Authority: **PI0270_LEARNER_STATE_RECOVERY_WITH_REUSED_CONTROL_GO_V1**.
Date: 2026-10-03. User request: review the actual experiment and publish the next goal to advance the main scientific line.
Review base: `d508c8782ec169b0f23c023d6d1a9020177a51d2` on `pi0267/executor-decoupled-recovery`.
Predecessor: PI0269 release `e77ad092406555645fb8c68ba34cc8594ef18ee6`, canonical Issue #11 comment `5964897978`.
This document is a **serialized, conditional successor**, not permission to interrupt PI0269 or create a competing run. Its immutable release commit is supplied by the source-bound Issue #11 publication.

## 0. Scientific objective and immediate action

Learn a shared RGB/language/actual-EEF-state policy that completes M1 and M3 under repeated feedback, including reliable terminal retention, using task data not collected by teleoperating Go2/ARX5 and not filtered by the target WBC's success. Keep embodiment constraints in a separate execution layer. Virtual success is an intermediate scientific result, not WBC, hardware or cross-robot qualification.

**Immediate action: adopt and finish the healthy PI0269 evaluator and preservation work under its existing counters. Do not restart its training, repeat completed valid cases, change its cards, or cancel PREFIX because legacy retention has already failed.** The same single Codex owner proceeds through this document's gates. Helpers are optional; no reciprocal owner/signature dependency is introduced. Do not install a second automatic goal or monitor.

Only after section 2 passes, one new learning arm, AGG4096, is authorized. Its matched control is the already-trained PI0269 PREFIX4096. Reuse that control's actual training and evaluation evidence instead of training it again. This is a prospective data intervention against a reused historical control, not a concurrent randomized or independent-seed replication.

New implementation and outputs are isolated under `scripts/pi0270_execution/`, `reports/PI0270/`, `docs/protocols/PI0270_*`, and corresponding PI0270 runtime roots. Continue the same execution branch; no new general-purpose Issue, PR77 pointer, permanent branch, force-push or merge. Never synchronize the moving Git branch into a sealed live runtime. Preserve concurrent commits.

## 1. Findings reviewed, and limits that remain

The following were actually read at the review base, not inferred from the handover:

- `reports/PI0269/admission/MAIN_TRAINING_AUDIT.json`, `ACTUAL_TRAINING_CLOSEOUT.json`, and `ALL_TRAINER_HF_CLOSEOUT.json`: UNIFORM and PREFIX each completed4096; raw8224 updates =8192 main +32 canary,148080 recorded training heads, zero technical recovery updates; four complete18-file Trainer HF readbacks closed. Actual input/random-packet/S4/mode and224 FIT consumption bindings are recorded.
- `reports/PI0269/offline-only/PRELIMINARY_RETURN.md` and `DONOR_UNIFORM_MAIN_CPU_PRELIMINARY.json`: all2904 offline logical rows are complete. FIT median prefix XYZ donor/UNIFORM/PREFIX =3.760/2.622/2.322mm. Paired PREFIX-minus-UNIFORM prefix medians are -0.220427mm for M1 and -0.295505mm for M3. The registered temporal fit contrast passes; this is not a task qualification.
- Legacy440 median XYZ/SO3: donor15.121mm/.046878rad, UNIFORM17.085mm/.070897rad, PREFIX17.587mm/.067752rad. Legacy REJECT counts0/8/0. Both new endpoints fail the donor-relative retention condition. Do not relabel this as a global normalization or prefix-weighting win.
- Complete UNIFORM main64: M1 original31/32 (E0:15/16, commands7/8 and8/8; E1:16/16), M3 original0/32, reliable0/64; full64 safe. This does not establish that every first-hit episode has the same late-failure mechanism.
- Latest read ledger is comment5968958021 at2026-10-03T12:01:29Z; review-base commit is dated12:06:07Z. The most recent scheduler snapshot is11:57:51Z: PREFIX main1/64, components0/96, evaluator PID3611282 live. These counts are dated observations, NOT current SSH telemetry. No full PREFIX/96-component/native-replay/result-archive terminal is assumed.
- The public main CPU summary contains aggregate verdicts and128 source identities. Its detailed11,144,986-byte per-case receipt is referenced on Corelab, not reproduced in Git. The reviewer did not replay those raw arrays or download all weights.

Code review covered the current PI0269 training entry and offline summary implementation, inherited temporal/S4/processor and feedback/scoring paths, and `pi0267_expert.ArclengthCorrection`. A correctly anchored first point does not determine the direction at0.4s. Teacher endpoint projection, zero-displacement hold and a sample drawn from the last nominal anchor are different labels. The existing teacher already defines progression and terminal correction.

Working hypothesis, not established root cause: replacing artificial nominal-neighborhood recovery states with states the fixed learner actually visits will improve recovery/progression/retention. Alternatives include remaining approximation error, weak visual/task conditioning, stochastic action bias, missing useful history, and terminal policy dynamics. No claim of unique covariate-shift causation is made. This is ONE bounded aggregation round, not full iterative DAgger or a convergence guarantee. Methodological context: Ross, Gordon and Bagnell, AISTATS2011, https://proceedings.mlr.press/v15/ross11a.html .

Local review arithmetic for counts/query-grid design is recorded in `reports/PI0270/design/REVIEW_AND_BUDGET.json`; it involved no neural, teacher, render, optimizer or physics call.

## 2. A0 — predecessor completion and deterministic successor gate

Before any PI0270 model query, new teacher label or render:

1. Refresh actual remote receipts and the existing serialized evaluator/lock. If PI0269 has progressed, adopt those results. Complete only its still-missing authorized work: original192 logical episodes,96 component proposals, all2904 offline rows, complete saved-array replay, reference/achieved/target analysis, failure-preserving videos, result archives/readbacks and cumulative accounting. Reuse all completed valid results and all four closed Trainer readbacks. No PI0269 update or inference budget reset.
2. Bind a complete PI0269 final report/requirement closeout and its source-bound canonical return. The owner publishes that return directly when the evidence exists; no extra helper or PI signature is needed. Explicit technical failures and valid quality failures remain separate. A real missing resource pauses only the affected operation; do not pretend a prepared script is an executed result.
3. Independently summarize the final ORIGINAL, RELIABLE, safety and legacy verdicts, plus component entry/midarc/terminal diagnostics. Preserve all original thresholds and opened-development labels. Do not use a partial PREFIX sample to select a donor or recipe.
4. **If PREFIX4096 passes BOTH original and reliable main64 qualification with full-point safety, skip sections4–7 learning/collection.** Return `PI0270_LEARNING_NOT_NEEDED_PREFIX_MAIN_RELIABLE` with the actual legacy/identity status and stop for a retention or independent-transfer decision. Do not train merely because this successor exists. If a qualifying virtual candidate exists, report it within the original scope; selected/deployment still NONE.
5. Otherwise, if the completed results are valid and the fixed control/initializer identities are recoverable, automatically continue to the one-arm recovery experiment below. There is no need to prove a unique neural root cause first. A newly demonstrated label/frame/time/processor/scorer correctness defect instead yields `PI0270_BLOCKED_SPECIFIC_CORRECTNESS_CONTRADICTION`, with exact evidence and preserved completed work; this release does not authorize changing the defect's scientific semantics.

CPU-only source preparation may overlap A0 in isolated PI0270 files, without touching active processes or consuming new scientific queries. The stage gate is actual predecessor completion, not a date or an elapsed wait. Do not poll tightly.

## 3. Fixed policies, contracts and counterfactual

### Common initializer for the existing control and the new arm

PI0268 ROT2048 complete Trainer:
- HF repo `leoyuer/pi0246-gate-pass-campaign`;
- revision `97d8bf027f07241e2206b05e07597e4719c891e4`;
- prefix `arms/PI0268/ROT/checkpoint-2048`;
- complete tree `3db80d35d5e476fe5a90d540ff8959fb5edf3da43a988fde7f88ce9f54b9b147`;
- effective statistics SHA256 `0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9`.

### Reused control and fixed collector

PI0269 PREFIX4096:
- revision `953bb358f37e4d788188c9bedcbeef1780f33e5e`;
- prefix `arms/PI0269/PREFIX/checkpoint-4096`;
- complete tree `12aff760d05dd05e0e11a024be3353fb45a5cb98a8b110ae7c7a0da7199609a5`;
- complete readback receipt SHA256 `08773cd8e8a85c4da439a9519be78b3c6a5291c3f35ca6b31fdb8eaa52480a7f` at the review base.

PREFIX is fixed now as the collector/control; it is not declared the best task policy. Do not switch to UNIFORM or an intermediate checkpoint after viewing remaining results. AGG starts from ROT2048, NOT from PREFIX4096: it is the counterfactual same4096-update run with a changed recovery-state source, not an extra4096-update continuation advantage.

Reuse the exact qualified PI0269 PREFIX recipe, including H40 on0,.05,...,1.95s, native BF16/FP32 boundaries, actual trainable-name set and frozen backbone, row-rot6d, relative SE(3), no-clamp ROT maps, gripper, input fields and deterministic camera preprocessing. Model identity includes weights, processor/statistics, embodiment mapping and decoding. Verify parsed statistics as well as hashes; serialization whitespace is not a semantic statistics change.

Loss stays exactly:

`J = L_native + .4*L_position + .1*L_increment + .1*L_rotation + .1*L_peak_speed`.

Use the inherited PREFIX weights: pose0..8 mass.8 /9..39 mass.2; transition0..7 mass.8 /8..38 mass.2. Native fullH40, full39-transition peak-speed term, physical scales, masks and S4 gradient path remain unchanged. No new loss, stopping head, smoothing, action projection, geometry input, history field, normalizer, precision or WBC change.

Compare the actual prior training fileset and loaded helper `__file__`/SHA values, not only the current Git filenames. Existing PI0269 `train.py` explicitly imports the qualified PI0267-local paired trainer and R3 physical decoder. Do not substitute a historical clamping helper. If control comparability cannot be established, stop with `CONTROL_NOT_COMPARABLE`; do not silently authorize a second new control training run.

## 4. A1 — fixed TRAIN learner-state collection and exact task labels

### 4.1 Collection, without teacher control

Use ALL24 original `FEEDBACK_TRAIN` geometry families from the unchanged PI0267 scene card, in its manifest order. Do not regenerate the10752-proposal corpus or create new geometry families. For each family use M1/M3, both original commands and E0/E1: **192 original-start virtual episodes**, each40s/100 replans maximum, exactly the existing initial-pose/camera/tool contract.

Run the fixed PREFIX4096 collector with fresh achieved RGB/StateV1, native S4, the unchanged fullH40 gate and actual0..0.4s mathematical execution. M1 collection language is original T0; M3 language is the original direction/endpoint instruction. No online teacher actions, target/path projection, snapping, smoothing, output stitching, reset after a quality failure, oracle stop-on-success or score-based data collection extension. Valid unsafe outputs stop the episode and remain valid failures. Every actual query/input/prediction/trace and actual achieved/installed trajectory is saved. Renderer graphics do not count as dynamic robot physics; retain the existing sim-time assertions.

Freeze all collection conditions and actual noise tensors before the first query. For `(family,task,command,backend,k)` use namespace `PI0270:COLLECT:27101:{family}:{task}:{command}:{backend}:{k}`, SHA256 first8 bytes big-endian modulo `(2**63-1)`, CPU generator, FP32 noise `[1,40,132]`; bind actual BF16 initial noise separately. No alternative noise seed is allowed. This namespace is separate from both old evaluation and training.

Also reuse ONLY the completed valid PREFIX4096 component trajectories from PI0269's first TWO TRAIN families: up to32 episodes and640 saved queries. Do not rerun those conditions. Their initial placements remain explicitly `VIRTUAL_DIAGNOSTIC_PLACEMENT`, not learner achievements. Prequery exclusions and missing queries remain exclusions/missing. This release explicitly permits these TRAIN-only observations to become new training material after A0; it does NOT permit any original8 development-EVAL family, factor panel, legacy evaluation row or WBC attempt to enter training.

### 4.2 Label proposals and correctness

For each new original-start episode propose labels at32 fixed query indices `q_j=(99*j)//31`, j=0..31. These are query indices, and the physical time is `.4*q_j`; never query at fictitious35s or substitute nearby successful timestamps. There are6144 original-start proposals. For the reused component episodes propose labels at each saved query0..19, at most640 additional proposals. Total at most6784 proposals. A scheduled query absent after a valid stop is `MISSING_AFTER_STOP`, not a replacement opportunity.

For each available observation invoke the unchanged ArclengthCorrection on its EXACT measured world position and rotation, correct family/task/command. Preserve world/anchor/TCP transformations and its original futureH40 label; never take a nearest existing TRAIN label as the label for the new state. Use its alias test and original teacher formula, including terminal exponential correction. Compute current/predicted/teacher same-state prefix discrepancy and direction from the saved collector prediction without an extra model call.

Eligible training labels require: finite coherent current RGB/StateV1 and timestamp; original necessary visibility for that task/command; no current tool collision; unambiguous teacher label; unchanged raw-label fullH40 gate and virtual-tool collision checks; actual processor round-trip within inherited tolerances. A model's poor prediction, reversal, task failure or high loss is NOT an exclusion. No WBC feasibility or successful target-WBC execution is a label requirement. No clipping/refitting of statistics to make new states appear in-range.

Keep every proposal and its reason in the denominator, including valid-but-unusable labels. Preserve independent CPU checks of first-pose consistency, noncommuting rotations and label/decode parity on this NEW support; reuse old fixtures rather than replaying all old teacher qualifications. Distinguish source last-anchor metadata, projected-terminal metadata, zero-displacement hold and nonzero terminal correction. Record actual normalized ranges, initial/terminal errors and coverage, not merely a total sample count.

A finite data-admission floor is8 eligible distinct processed StateV1 values per `(family,task,command)` cell across the fixed sources. Report all missing strata. If any cell lacks that support or cannot furnish the fixed new-fit card, return `PI0270_DATA_COVERAGE_INSUFFICIENT`, preserve the192 attempts and evidence, and stop; no extra family, reset, sample search or repeated collection is authorized. Ordinary technical recoveries reuse exact saved packets and remain within all hard caps.

## 5. A2 — one frozen state-source intervention, not a new curriculum search

Start from original PI0267 R `PAIRING_PLAN.json`, SHA256 `fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef`. Preserve all4096 unit identities, family/task schedule, command order, M1 T0–T7/M3 text, per-unit seeds, microbatch1/accumulation2 and total8192 microbatches.

- All2048 `COMMON_NOMINAL` units /4096 microbatches stay bit-identical to the actual PREFIX control inputs and labels.
- Replace ONLY the2048 `RECOVERY_INTERVENTION` units /4096 microbatches with eligible A1 learner-state records of the SAME family/task/command. RGB, actual state and corrected task label move together. Do not reuse nominal images with a changed state. The original scheduled anchor/variant is provenance, NOT a claim about the learner state's actual projection.
- Preserve the original unit's language template when encoding the selected state. The task/command semantics do not change. Record any difference from the collector's T0 input identity; do not falsely call a different text the exact same processed condition.

Freeze deterministic selection BEFORE new-arm training: within each cell, bucket original-start records by query-time quartile0..3, then component records by their declared start type. Use this fixed bucket order: ORIGINAL_Q0,Q1,Q2,Q3,M1_TERMINAL,M3_ENTRY,M3_MIDARC,M3_TERMINAL, restricted to nonempty valid buckets for the cell. Sort each bucket by `(source_episode_key,query_index)`; for the n-th replacement occurrence in that cell select bucket `n % B`, item `(n // B) % len(bucket)`, starting n=0. Empty buckets are reported, not silently populated by other families. No score, teacher progress, model error or successful endpoint determines the sample order. This is one fixed combination of learner-visited and explicitly placed TRAIN recovery support, not a claim isolating each axis, terminal-only samples or pure natural visitation.

Keep an immutable source-to-training mapping, distinct condition counts, actual terminal/zero-motion/correction counts and used/not-used proposals. Do not add factor/legacy evaluation rows to repair forgetting. Legacy retention is still measured; a separate retention intervention is NOT smuggled into both arms.

**Important evaluation terminology:** the old FIT224 includes old recovery conditions now potentially absent from AGG training. Keep it unchanged as an old-support panel; report each row's ACTUAL AGG-consumed status. Do not continue calling all224 rows seen TRAIN fit for the new arm.

Build `NEW_FIT192` before training from the FIRST TWO distinct processed replacement conditions in each of96 `(24 families x2 tasks x2 commands)` cells, in the frozen plan order. All192 must subsequently be proven actually consumed, with exact text/state/image/label identities. This is training-support diagnosis, not blind test. Freeze one FP32 `[1,40,132]` noise per row under `PI0270:NEWFIT:27102:{input_id}` with the same SHA/CPU-generation convention. Query PREFIX4096 and AGG4096 on this card only after AGG's final is frozen:384 maximum new queries. Reuse a prior prediction only for exact full policy/input/noise identity, never just the same pose.

## 6. A3 — real canary and AGG4096 training

Use a narrow isolated data-loader override of the qualified PI0269 PREFIX trainer. Preserve the exact inherited actual code/precision/frozen scope through a source-bound execution manifest; the data source is the sole major experimental change.

Fresh matched AdamW, LR1e-5, warmup10, cosine4096, weight_decay1e-5, max_grad_norm1.0; seed26741 and the original big-endian per-unit draw schedule. AGG starts from the fixed ROT2048 WEIGHTS with fresh optimizer, exactly like the reused control. No new trainable parameters, backbone unfreezing, capacity change or altered S4 count.

Run one real AGG canary: continuous8 versus4/full-save/fresh-process-resume4,16 total optimizer updates. Match model/optimizer/scheduler/cursor, actual data and random packets, modes and Python/NumPy/CPU/CUDA RNG. Retain the DataLoader RNG lifecycle and checkpoint metadata repairs. Capture the already-required first-backward gradients at all four S4 nodes without an extra diagnostic backward. Discard canary weights after evidence is durable; do not repeat the old PREFIX canary.

On correctness PASS, automatically run AGG4096. Save complete2048 and4096 Trainers; evaluate ONLY4096. Preserve processor/config/statistics/mapping/RNG/optimizer/scheduler and full inventories, and perform fixed-revision HF full-file readbacks for these TWO new artifacts only. Existing donor/control inventories and model-only views may be reused with exact bindings. No optimizer payload on Corelab.

Audit actual common-input bit identities against the old control for all common units. For ALL units, including changed-state units, compare actual native epsilon, tau and stock initial-noise tensors/hashes against the old control; same seed alone is insufficient. Assert exact scheduled family/task/command/template/unit order and current labels/masks. Inspect full-S4 loss application, gradient path, clipping, restored modes and frozen-name/weight scope. Changed states and labels are intentionally NOT expected to have identical condition hashes. If a required stochastic/control contract differs, record `CONTROL_NOT_COMPARABLE`, rather than calling unmatched training a data effect.

Huge finite losses are recorded. A real nonfinite optimization/corrupt-input event is an affected-path correctness stop, not authority to silently change LR, precision or statistics. No checkpoint selection, extra healthy updates, third model, data-bank refresh or second aggregation round.

## 7. A4 — fixed complete evaluation and mainline qualification

Freeze all cards and code before AGG quality queries. Reuse the original input/noise/geometry/scorer contracts and exact closed control/donor outputs. New-model inference is label-free RGB/StateV1/language; no teacher/geometry/phase/target oracle enters the worker.

### Existing offline, original-start and component cards

- Standard offline:968 rows per point =224 old-FIT +304 factor +440 legacy. Points ROT2048, PREFIX4096, AGG4096 give2904 logical rows;1936 donor/control rows reused and968 AGG queries new. Preserve rejected/nonfinite rows and all subgroup denominators. Distinguish genuinely consumed old-support rows from replaced recovery rows.
- Original-start main: same8 opened development families x2 tasks x2 commands xE0/E1,64 per point.192 logical episodes,128 exact donor/control reuse,64 AGG new. No geometry, goal, command or noise substitution.40s/100 replans, full gate and all-history original V3; V2 side labels remain.
- Component: same32 proposals per point from the first2 TRAIN families,8s/20 replans. Reuse64 ROT/PREFIX proposals; execute32 AGG proposals. Since some PREFIX component states became A1 training data, this is explicitly a TRAIN-supported component diagnostic, NEVER independent transfer or replacement main qualification. Preserve exclusions; no favorable replacements or shortened hold episodes.
- NEW_FIT192:384 new logical comparison rows (PREFIX versus AGG), exactly matched new-support conditions/noise, with actual AGG consumption proven. Report paired median deltas by task/command/cell, not differences of marginal medians. Direction on zero teacher displacement stays undefined/hold, not success.

### Separate questions and unchanged thresholds

Report separately: old/new-support fitting, main task first completion, persistent reliability, safety, legacy retention, and component behavior. The primary experiment endpoint is AGG's unchanged original AND reliable main64 qualification, compared with the exact PREFIX control; no lower substitute task-pass threshold is introduced.

Original/reliable subgroup thresholds remain at least13/16 for EACH task/backend and6/8 for EACH command, with original full-point safety. Reliable means original task completed by35s, safely run through40s, every one of251 saved35..40s ticks within35mm/.10rad. M3 retains full ordered270-degree/12-bin traversal, original entry and1mm reverse tolerance. No oracle stop-on-success, target snap or gate relaxation.

A scoped `RELIABLE_VIRTUAL_RESEARCH_CANDIDATE` additionally requires complete identities and no increase against the ORIGINAL ROT2048 donor in legacy440 REJECT or median XYZ/SO3. Do not reset the legacy baseline to the already-regressed PREFIX endpoint. If main reliability improves but legacy still fails, report `MAIN_RELIABILITY_IMPROVED_LEGACY_FAIL`; it is useful scientific progress, not candidate qualification. If only fit improves, report it without claiming the mainline is solved. Formal selected/deployment remain NONE in every branch of this virtual study.

Task-space reference correctness, installed-reference tracking and achieved task completion must be shown separately at the same timestamps, including terminal and post-entry behavior. A zero reliable count alone does not identify which reliability subcondition failed. Report those subconditions and all completed failures. E0/E1 are mathematical executors, not two robots; the8 families have informed development and are not blind TEST. Repeated windows/noises/backends do not multiply independent scene count. Publish family-level paired changes; do not use tick-level pseudo-replication for significance.

CPU replay must cover the NEW saved S4 traces/decoding, physical metrics, achieved/installed arrays and reliability. Reuse old closed replay receipts. Original task scoring still shares the frozen V3 path; disclose that limitation, and independently recompute geometry/physical metrics where already supported. Component angular-bin diagnostics never override original task labels. Make one fixed-first-family, all-eight-case AGG video from saved observations, with complete failures and VIRTUAL/NOT_WBC/NON_REALTIME labels; no new renders/queries for presentation.

## 8. Budgets, resources and technical accounting

PI0269 completion retains its OLD ledger and caps; the following is a separate bounded PI0270 budget, not a reset or permission to repeat old work.

| New consumption | Planned maximum | Hard ceiling |
|---|---:|---:|
| Official AGG optimizer updates |4096|4096|
| Canary optimizer updates |16|16 planned|
| Completed updates incl. technical reserve |4112|4368 =4096+16+256|
| Recorded training heads, incl. parity/recompute/recovery |existing18-head/update convention|80000|
| Original TRAIN collector episodes / queries |192 /19200|included in global caps below|
| Standard AGG offline queries |968|968 scientific conditions|
| NEW_FIT192 two-point queries |384|384 scientific conditions|
| AGG main episodes / queries |64 /6400|same fixed conditions|
| AGG component proposals / queries |32 /640|same fixed conditions|
| New neural inference attempts total |27592|30720|
| New inference S4 heads |110368|122880|
| New virtual starts |288 =192+64+32|304|
| New observation frame pairs |26240|28672|
| Teacher-label proposals |6784|same frozen proposals|
| Actual teacher calls incl. technical repeats |<=6784 planned|7168|
| New dynamic WBC / physics / hardware starts |0|0|

No new geometry family, external-model API, paid service, language template, controller or robot. Saved-array arithmetic, packet hashing, gate checks and metadata mapping are not model calls. Technical reserve cannot buy scientific conditions, another training seed or another aggregation round. Every new inference attempt is journaled before execution; record actual partial heads and distinguish technical-incomplete attempts from completed valid predictions. Do not require a crashed attempt to have four heads or fabricate an output to satisfy a reconciliation assertion. All completed valid outputs are immutable and reused; retries preserve their separate attempt consumption.

Hopper: admitted real training only on matching hardware/software/precision. Corelab: serialized no-gradient collection/evaluation/rendering. CPU: metadata, teacher labels and replay, low-priority bounded threads. Reuse the original privacy and asset restrictions. Check actual user-fileset/quota conditions, unique payloads and overlapping full-save/staging needs plus10GiB margin; no arbitrary300GiB target or inference of quota from a global mount. Protect initializer/control, current recoveries and unique evidence. No broad cleanup or repeated historical HF downloads. Archival may overlap other stages only when immutable identities and durable complete sources exist and resource separation is safe.

Same-semantic import/path/serialization/cache/lifecycle/attempt-accounting repairs are autonomous, with evidence and narrow tests. Changes to teacher, input, labels beyond the declared new states, loss, precision, masks, scorer, normalizer, controller or scientific sample selection require a separate decision. No healthy job cancellation or extra sign-off gate.

## 9. Deliverables, stop and next scientific decision

Deliver a single source-bound research return containing: predecessor actual closure; fixed donor/control identity and control-reuse proof; collection/card/noise/attempt receipts; all6784 label proposals with exclusions/missingness and coverage; frozen changed-vs-common plan; actual consumption and actual-noise comparison; real AGG canary; actual4096 training audit; two complete Trainer HF readbacks; standard2904 offline, NEW_FIT384 comparison rows,192 main and96 component logical proposals with exact reuse accounting; full new saved-output CPU replay; subgroup/reference/tracking/task/terminal analysis; fixed failure-preserving video; complete result archive/readbacks and all budgets.

No stop at CODE_READY, CPU_PASS, DATA_READY, JOB_SUBMITTED, TRAINING_DONE or UPLOADED when the next authorized stage has its dependencies. Execute automatically through the fixed complete evaluation and preservation, unless one of the explicit correctness/resource/data-admission/skip branches applies. Quality failure can be a complete bounded study.

Terminal: **PI0270_LEARNER_STATE_RECOVERY_STUDY_COMPLETE — STOP_FOR_RETENTION_OR_EXECUTOR_TRANSFER_DECISION**.

Interpretation routes, NOT additional launch permission:
- Reliable main task plus unchanged legacy qualification: prepare a frozen policy-to-executor identity handoff, then separately authorize genuinely new scenes and a validated physical backend. No physical run is authorized here.
- Reliable main task with legacy regression: preserve the positive task result and negative retention verdict; choose one future retention intervention, without silently accepting a weaker candidate rule.
- New-support fitting improves but main task does not: use observed component/transition/terminal errors to choose a finite representation/terminal-dynamics or bounded direct-policy baseline, not unbounded additional aggregation or temporal-weight search.
- Even actually consumed new-support prefixes remain inaccurate: prioritize finite fitting/optimization feasibility over larger collections. Do not label this alone a generalization failure.
- Low-level transfer later fails: separate observation shift, feasibility, installation and tracking. Do not train the high-level weights into an undeclared single-WBC compensator.

PI0267 secondary WBC remains PARTIAL with its original8 valid experts,1 engineering-incomplete model attempt/3 queries and23 deferred conditions. These are not23 quality failures and this protocol does not resume them. No merge, PR77 pointer, mail or monitor. The mainline goal remains same task data/shared high-level weights with explicit replaceable embodiment execution, not a selected video or a relaxed PASS.

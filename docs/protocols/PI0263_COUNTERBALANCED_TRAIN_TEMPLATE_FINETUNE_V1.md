# PI0263 — Counterbalance TRAIN wording without changing the Motion2 control problem

Publication state: AUTHORIZED_SCIENTIFIC_EXPERIMENT_NOT_EXECUTED.

Authority: `PI0263_TRAIN_TEMPLATE_COUNTERBALANCING_GO_V1`.

One lead owns implementation, source checks, real canaries, both training arms, fixed evaluation, saved-array replay, archival and presentation. Assistance is optional and never a signature dependency. This is a new bounded experiment, not a restart of PI0262 or an automatic physical admission.

## 1. Evidence and scientific decision

Read these immutable sources before execution:

- PI0262 canonical: Issue #11 comment 5912394938.
- PI0262 evidence commit: `2f93913efdc80b9480ea5fb27ebe3f27a54e6879`.
- `reports/PI0262/FOUR_CELL_REPORT.md`, `NEXT_INTERVENTION_DECISION.md`, the completed CPU receipt, crossed card, wording flips and target-choice decomposition.
- PI0261 evidence commit: `3e3983c8345d50eb322405084efd4f3c9381d97f`, canonical comment 5909947990.
- Its actual `scripts/pi0261_train.py`, model/input bindings, `MAIN_TRAINING_AUDIT.json`, `MODEL_QUALITY_REPORT.md`, `PAIRING_PLAN.json` and final model receipts.
- Prior paired-training authority: `6f2353a718cc877dddb3124029de76007eaa7df3`.

PI0262 completed 1920 rows, reusing 504 and making 1416 new queries / 5664 S4 heads, with zero retries or training. Its diagnosis is MIXED, not language-only or proven visual failure. P's original TRAIN wording bindings scored 59/60 and 60/60; replacing only the wording with the other already-supervised template scored 40/60 and 46/60. Thus strong original TRAIN-entry performance did not establish interchangeability even between T0 and T1.

P's actual crossed counts, kept separate by noise:

| Panel | B0 | B1 |
|---|---:|---:|
| TRAIN T0 |55/60|57/60|
| TRAIN T1 |44/60|49/60|
| TRAIN V0 |53/60|52/60|
| TRAIN V1 |33/60|36/60|
| KIN-VAL T0 |17/20|14/20|
| KIN-VAL T1 |12/20|14/20|
| KIN-VAL V0 |14/20|16/20|
| KIN-VAL V1 |11/20|12/20|

These are correlated family measurements. The table neither proves that the backbone cannot understand V1 nor that scenes no longer matter. V0 can outperform supervised T1. Color/command frequencies differ between splits; the split is not a randomized pure-geometry intervention. Preserve the report's counterevidence.

The direction/bias analysis often finds positive instruction ordering but a common offset large enough to leave one target choice wrong. That is an output-space description, not a proven neural mechanism or authority to add a new loss. The first intervention is the source report's single recommendation: counterbalance T0/T1 against target, color, family and state within TRAIN.

The publisher read repository evidence and training source, and checked the experiment arithmetic. The publisher did not SSH, download/replay the full remote raw archive, run models or implement the proposed sampler. Reuse unchanged source receipts; verify actual newly materialized bytes. Publication is not an execution or correctness certificate.

## 2. Fixed scope and initialization

Both arms start from the SAME PI0261 P2000 weights:

- HF repo: `leoyuer/pi0246-gate-pass-campaign`.
- Revision: `875abdde2cc7c9e6858fac008e53f2f4b3375255`.
- Prefix: `arms/PI0261/P/checkpoint-2000`.
- Reported complete Trainer tree: `882931d754cb25653105a9213fe0dbeb4d2b14e463efb27e49c6101fd5fc5e4b`.

This is a research initializer chosen prospectively for retained geometry/M3 and strong TRAIN command choice, not a qualified deployment selection. Do not substitute C, an intermediate endpoint, G, or the original Base after observing new results.

Preserve COMPACT_30_10 membership, original images/state/labels/masks, statistics, processor/tokenizer, embodiment, frame/TCP convention, H40/dt=.05, S4 and native inference. No new scene, rendering, IK, camera, dataset-statistics fit, label averaging/retiming, target coordinate, phase/source token, segmentation/depth input, prompt canonicalizer or output repair.

Only these original M1 TRAIN template strings may enter the intervention:

- T0: `Approach the {color} target and align the tool with its axes.`
- T1: `Bring the tool to the {color} marker with the pose shown by its axes.`

Color is the original requested target's semantic color, cross-checked against the original registry and sequence. Command 0 is not a fixed color. Both templates preserve the same target and orientation requirement; XYZ/SO3/gripper labels are unchanged.

V0 (`Move to ...`) and V1 (`Reach ...`) stay out of optimization, normalization and training-side quality selection. They are held out from this project's supervision, not necessarily from base-model pretraining. All M3 training text remains exactly its original sequence instruction. No new paraphrase generation.

## 3. Matched F2000 / X2000 experiment

- **F2000**: fixed original per-sequence wording, the continuation control.
- **X2000**: counterbalanced T0/T1 for M1, original wording for M3.

Each arm runs 2000 optimizer updates with two sequential supervised microbatches at the same weights and one mean-loss update. Each arm therefore computes 4000 supervised examples: 2000 M1 and 2000 M3, with both commands equally represented. Half the units per task are same-observation common-start pairs, half ordinary phase-balanced replay, preserving the P-style entry/replay mixture.

Only M1 instruction assignment differs BETWEEN these arms. Frames, states, action labels, masks, source visit sequence, number of examples, update count, initial weights, loss, optimizer and actual random packets match. Common changes versus old PI0261, including another 2000 updates and the repeat-paired visit design below, cannot be attributed uniquely to template counterbalancing.

### 3.1 Deterministic source-unit bank; no new sample search

Reuse original `PAIRING_PLAN.json`, SHA256 `208f0bfbc97095369ce84329899bbbde67eb5f694ca8c9022f0566dc61f28477`, and its actual referenced source records. Resolve the actual existing plan builder through the source manifest; do not invent filenames or regenerate a plan from a guessed implementation.

Construct 1000 base units from the archived 2000-unit plan, BEFORE any new quality query:

1. Partition existing units by (M1/M3, PAIR_SLOT/REPLAY_SLOT), four strata.
2. Within each stratum and each of the 30 TRAIN families, retain the lowest original unit ID. This provides 30 coverage units per stratum.
3. Add the earliest remaining unused units in that stratum until exactly 250 are retained. No selection uses model errors, margins or visibility scores.
4. Merge all retained units and sort by original unit ID. Bind their original P-selected pair: common-start bank for PAIR_SLOT, ordinary pair for REPLAY_SLOT. Verify all original records, indices and labels; no sequence/window replacement.

The result is 500 M1 and 500 M3 base units, 250 per task/slot stratum, with all TRAIN families covered. Each unit has both original commands. Family/phase frequencies inherited by this deterministic subset are reported, not falsely declared globally uniform. A missing source or schema discrepancy is an implementation/provenance issue to resolve from the original plan, not permission to select new examples.

### 3.2 Two visits per base unit: exact template crossing

For base unit k in the above order, create consecutive new optimizer units 2k-1 and 2k. BOTH arms repeat the same selected observations/labels on those two visits.

- F uses each sequence's original instruction on both visits.
- For X's M1 units, one visit uses T0 for BOTH commands with their respective correct colors; the other visit uses T1 for BOTH commands. Thus a template itself does not identify a side/command within the unit.
- For X's M3 units, both visits retain the exact original sequence text, identical to F.
- Order c0,c1 on the first visit and c1,c0 on the second in both arms. Accumulate the two half-loss gradients before one optimizer/scheduler step; no weight update between commands.
- Determine M1 template-visit order prospectively: sort the 500 M1 base-unit IDs by SHA256 of `PI0263:26301:template-order:<original-unit-id>`, using original ID as tie-break. The first 250 use T0 then T1; the other 250 use T1 then T0. Freeze the rendered schedule.

Every selected M1 (family, command, color, state/frame, reference index, label) observation receives both templates equally in X, including repeated records. Per arm there are 1000 M1 updates / 2000 examples; X has exactly 1000 M1 T0 and 1000 M1 T1 examples. This guarantees template balance within each observed target/color/family stratum; it does NOT equalize the original frequencies of different colors or families. No resampling to force equal RGB-color frequencies is authorized.

The F arm may retain different overall T0/T1 frequencies. This comparison tests the entire counterbalanced assignment policy, including marginal frequency changes; do not claim it separately identifies pure decorrelation at fixed template marginals.

### 3.3 Actual conditioning and random-packet checks

At the real processor/collator boundary, exclude only labels/masks and audit metadata when computing conditioning identity. For corresponding F/X microbatches, all nonlanguage inputs and all supervised labels/masks must match. M1 language tokens can intentionally differ; M3 full conditioning must match. Within each PAIR_SLOT, opposite commands have identical nonlanguage observations. Ordinary replay is NOT required to have equal state across commands.

Use unit-keyed RNG namespace `PI0263:26301:<new-unit-id>` independent of labels and template correctness. Within a unit and across arms share the actual native epsilon/tau and free-S4 starting noise; retain arrays or sufficient existing raw audit plus digest. The two consecutive visits may have different noise, but BOTH arms use the same corresponding packets. Do not assume equal seeds suffice under differing token lengths or branching. Reuse the proven paired-random-input mechanism, testing the newly crossed text lengths and padding.

No audit field, unit ID, template ID, target metadata or label may leak into model conditioning. Shape differences caused by real language text are expected, not an integrity failure.

## 4. Finite preflight and autonomous continuation

One source-bound CPU pass performs: donor/data identity checks; source-unit selection; color/text semantic checks; exact exposure-count verification; cross-arm label and nonlanguage equality; no V0/V1 TRAIN leakage; valid masks; and the complete fixed evaluation manifest below. Reuse accepted geometry, codec, temporal-alias, noise and old raw-output receipts. Do not re-audit the whole history, regenerate H0/H1 or run PI0262 a second time.

Use negative tests for wrong color, wrong template assignment, leaked metadata, duplicated/skipped unit, incorrect half-loss scale and false cache reuse. The actual source-specific checks must be run, not inferred from a generic CPU PASS.

If these concrete checks pass, proceed directly to real canaries and BOTH arms. The MIXED diagnosis, known same-command temporal alias, lack of helper signature or inability to name a unique neural mechanism are not blockers. A true source mismatch, label damage or private/held-out data leak blocks only the affected descendants until resolved. Repairs that change scientific meaning require a new decision, not an engineering label.

## 5. Frozen optimization and recovery

For each arm: fresh AdamW, LR1e-5, betas(.9,.999), eps1e-8, weight_decay1e-5, accumulated-gradient clipping norm1, warmup10, cosine2000. Microbatch1, gradient accumulation2. Preserve FP32 trainable action projector/diffusion and the original BF16 frozen vision-language/backbone scope. Do not unfreeze, add LoRA, quantize or change inference precision.

Keep the inherited objective exactly:

`J = L_native + 0.4 L_position + 0.1 L_increment + 0.1 L_rotation + 0.1 L_peak_speed_margin`,

using the actual full-S4 free-generation graph and the same physical decode/statistics. Each unit uses `(J_c0 + J_c1)/2`; no ranking, consistency, embedding or contrastive loss is introduced.

Initialize weights from P2000 but NOT its optimizer or exhausted scheduler. Exact restoration applies within a new arm. Each arm runs one real continuous-8 versus 4-save/fresh-process-resume-4 canary. Reuse old tests where identical, but verify the new repeated-unit/template cursor and actual token/random packets. The two paths must agree under frozen comparisons for source visits, modes, optimizer, scheduler and weights. No tolerance relaxation after seeing discrepancies. These 32 completed canary updates are discarded; main training restarts from the fixed donor.

Save only after complete two-microbatch optimizer units; interrupted partial units are not silently promoted to steps. Main complete checkpoints at 1000/2000; no intermediate quality screen or rescue selection. F then X is the default sequential order. A valid F quality FAIL does not cancel or change X. Preserve at least one durable complete resume source before retiring local copies.

Log already-computed per-example losses, task/template/color/family/index, source/condition hashes, actual RNG identity and accumulated update norms. Do not add separate unbudgeted model probes just for diagnostics.

## 6. Evaluation: preserve old panels and retain the full crossed diagnosis

Freeze all three points: donor=P2000, F2000, X2000. No C2000/old Base/G or step1000 new quality queries.

Per point, retain the UNION of:

1. PI0261's original 680 logical rows: original 440-row Motion2/O64 panel plus the 240 TRAIN-entry rows.
2. PI0262's full M1 640-row crossed panel: 40 families x2 commands x4 exact templates x2 actual noises.

The expected exact overlap is 168 logical rows per point (120 original M1 TRAIN-entry +48 original six-family M1 validation wording rows). Therefore 1152 unique logical rows per point and 3456 across the three points; both named panel views remain available with all original denominators. The 1320 panel memberships per point must not be described as 1320 unique queries. Verify the join from source/label/condition identities; distinct labels sharing a forward remain distinct scoring rows.

The donor's complete 1152 unique rows already exist across PI0261 and PI0262 and must be reused after identity checks. No new donor query is planned. Each new endpoint has at most1152 new queries before additional exact cache reuse: planned new <=2304. Hard cap2432 attempts /9728 inference heads, including128 strictly no-valid-result technical attempts. Cache savings cannot fund extra prompts, models, seeds, scenes or test rows. Missing archived donor bytes must be restored from their frozen public evidence, not recomputed simply for convenience.

All source labels, raw images, B0/B1 arrays, templates, state/frame conventions, processor, S4, gate and scoring remain unchanged. The full card must be frozen before the first new endpoint quality query. Training may not use any new validation output to adjust the other arm, stopping step or text schedule.

### 6.1 Separate old qualification from the new robustness panel

Keep the original four KIN-VAL numerical groups: median XYZ<=.020m, p95 XYZ<=.035m, median SO3<=.050rad, zero REJECT. Keep the original eight language groups: >=10/12 separately for each task/wording/noise, using the original strict own/opposite H40 XYZ margin. Ties are not correct; rejected/invalid outcomes stay in denominators with separate status. Report `legacy_full_quality_pass` without rewriting any historical FAIL.

Also preregister `crossed_M1_robustness_pass`: all eight M1 KIN-VAL template/noise groups must have >=17/20 correct, zero REJECT, and the same .020/.035/.050 numerical thresholds. This is a NEW prospective robustness criterion, not a corrected PI0261/PI0262 gate and not permission to remove their historical failures. TRAIN counterbalance results and all per-command/both-command family counts are mandatory diagnostics, not a reason to stop midway.

Only a final endpoint passing BOTH qualifications may be listed as `synthetic_research_candidate`. If both F and X pass, prefer F as the simpler fixed-assignment continuation, while report the full comparative effects. Formal `selected_checkpoint=NONE` and `deployment_selected_checkpoint=NONE` remain regardless. This pilot does not grant generalist or physical readiness.

### 6.2 Required interpretation, not post-hoc score repair

Report each split/template/noise separately, not only pooled four-cell totals. Reconstruct original-versus-other T0/T1 assignments from source metadata. For X-final, the previously unpaired TRAIN template combinations are now TRAIN-exposed; call them counterbalanced training support, not unseen-language tests.

For the same target/state/noise compare T0/T1 and V0/V1 correctness flips and signed margins. For the same template/state/noise compare correct target choice, fully correct opposing pairs and color/command strata. Reuse the existing s_c, g and b decomposition; do not replace correctness with output-difference magnitude. Keep M3 and O64 regression panels: improvement confined to M1 with broken M3 or geometry is not full success.

V0/V1 remain absent from optimization, but their validation families and scores have informed this research. Label all these results DEVELOPMENT VALIDATION. There are30 TRAIN/10 validation geometry families, not thousands of independent experiments. Preserve both noises separately and avoid binomial claims treating every template/noise row as independent. Broad unseen-task, real-world, full-contour or WBC success is not measured.

No nearest-of-many-label selection, DTW, time alignment, label averaging, floor subtraction, output smoothing, prompt routing/canonicalization or best-of-noise. Same-command temporal ambiguity stays a reported limitation, not a new blocker.

## 7. Engineering autonomy, resources and accounting

This is a NEW PI0263 allocation, separately preserving old PI0261/PI0262 counts:

| Activity | Budget |
|---|---:|
| Main optimizer updates |4000 total: F2000 + X2000|
| Main supervised microbatches |8000 total|
| Discarded real canary updates |32|
| Completed-update technical recovery reserve |256|
| All completed optimizer updates |4288 hard cap|
| Raw training action-head forwards, including recomputation/failures |100000|
| Logical unique evaluation rows |3456 across donor/F/X|
| Donor unique rows reused |1152|
| Planned new quality queries before exact reuse |<=2304|
| New query attempts / inference heads incl. technical reserve |2432 /9728|
| New scenes, camera renders, IK, labels, statistics fit |0|
| WBC/physics/TEST/old DEV-F/real hardware |0|

No mechanical qsub/IO retry count replaces actual compute bounds. The lead may autonomously fix paths, imports, serialization, staging, exact-recovery and resource configuration within frozen semantics, with narrow CPU tests and versioned receipts. No cancellation of healthy unrelated work, duplicate dispatcher or quality-driven retry. Failed/partial work is recorded; reserve cannot extend a completed2000-step arm or add a third arm. If durable state and remaining compute cannot complete the authorized experiment, report the precise deficit rather than resetting counters.

PC coordinates; Corelab handles CPU, no-gradient inference and saved-array presentation; Hopper handles actual model canaries/training on lawful compatible allocations. Ordinary debug tests do not require H200; GPU choice follows proven peak-memory/runtime needs rather than an arbitrary host name. Preserve precision and algorithms. Missing scheduler tooling means UNKNOWN, not no jobs.

Keep donor, source models, current full resume and unique evidence. Reuse accepted fixed-HF readbacks for unchanged assets; verify new local materializations and new complete Trainer archives. Preserve next-write overlap plus the inherited10GiB reserve, and report quota UNKNOWN when unknown. No broad cleanup, repeated old-data upload, private robot/WBC asset upload, credentials, email, monitor, daemon or PR merge.

## 8. Completion and scientific next decision

Required final deliverables: source-bound diagnosis summary; frozen F/X schedule and actual counterbalance checks; real recovery receipts; both complete2000 endpoints; the two original panel views and full3456-row union; independent-computation-path CPU replays; per-family/wording/direction/regression report; complete new Trainer and result HF public fixed-revision readback; compact donor/F/X comparison using saved images/predictions only. Same-owner replay must be labeled honestly, not an invented external signoff. Video cosmetics cannot delay substantive completion or create new queries.

Terminal:

`PI0263_TEMPLATE_COUNTERBALANCING_PILOT_COMPLETE — STOP_FOR_INDEPENDENT_GENERALIZATION_OR_PHYSICAL_ADMISSION`.

Execution COMPLETE and quality PASS/FAIL are separate. Both valid endpoints must finish even if one quality-fails. Missing an arm or mandatory panel is PARTIAL, not COMPLETE. The lead continues from finite preflight to training without another routine GO when actual correctness checks pass.

Interpretation after this fixed experiment (NOT extra authority):

- X beats F on crossed TRAIN and development-validation/held wording while geometry and M3 hold: evidence supports counterbalanced instruction assignment in this setup. Do not call it a unique neural mechanism or broad generalization proof.
- X improves only the now-exposed TRAIN combinations: recombination was learnable, but robust scene/wording transfer remains unresolved; next choose one controlled representation or language-diversity intervention rather than replaying old validation until it passes.
- Both arms improve similarly: additional optimization/shared sampling may explain common gains; do not credit X uniquely.
- Familiar T0/T1 improves yet V1 remains weak: language transfer merits the next control. Familiar-template scene gaps persisting across both arms motivate a separate grounded visual/scene-support study. MIXED results remain permissible.
- A research endpoint qualifies: prioritize genuinely new, prospectively sealed geometry/expression conditions and separately admitted frozen-WBC compatibility. Do not make another old-O64-only fitting run the mainline.

The immediate objective is to remove the fixed TRAIN wording-to-target/family association and measure its consequence, without restarting camera/IK/data engineering or changing four scientific factors at once.

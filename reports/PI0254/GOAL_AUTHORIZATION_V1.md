# PI0254 — Diagnose, then controlled support-expansion fine-tuning

Authorization token: **PI0254_DIAGNOSE_THEN_CONTROLLED_FINETUNE_GO_V1**.

Authority: the user's explicit instruction on 2026-09-29, “授权下一goal，先仔细分析失败原因，再尝试新的微调训练”. This document authorizes the bounded dependency chain below, not just another retrospective report. It is a new research experiment; it does not amend PI0253's scientific result or authorize deployment. Publication of this document is not evidence that a remote job has started.

Base evidence commit: `59ae38d99faf76d8c856cdb757de030161effe6f`.
P0 canonical: https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5881992689
P0 report: `reports/PI0253/P0_REVIEW/REPORT.md` at the base commit.
Inherited training contract: `reports/PI0253/FULL_run_card.json` at the base commit.
G qualification identity: `reports/PI0253/G_qualification_index.json` at the base commit.
Model recovery identities: the existing PI0253 completion audit and fixed HF receipts. Local paths in historical run cards may have been retired; resolve immutable assets through those receipts, not an assumed existing directory.

## 1. Objective and evidence boundaries

Answer two questions in order:

1. Did FULL10000 improve representative TRAIN support outside the original64, or is poor fitting more widespread?
2. Can a deliberately repeated, nine-times-larger-than-M64 support be learned while retaining M64, and does explicitly increasing old-support rehearsal help at the same update budget?

Then produce two genuinely trained, comparable research arms, or a documented engineering/scientific stop at an explicit boundary below. A valid mixed/inconclusive P1 result is NOT a reason to stop at analysis; it is itself useful context for the already authorized controlled training. Do not wait for another human GO between these stages when the frozen prerequisites pass.

P0 already establishes label/admission conflict, broad old-support degradation, low old-window exposure, and a mixed-frame historical CALIB distance statistic. It does NOT establish zero model rotation error, a unique optimizer cause, universal mode collapse, model incapacity, or blind generalization failure.

The joined expert/model CSV contains, for example, expert-to-current rotation 2.0103091722 rad with a 1.5-rad admission limit. By the triangle inequality, any first pose inside that limit must differ from this expert first pose by at least 0.5103091722 rad. This is a local first-pose incompatibility, not a lower bound on the aggregate H40 median metric, and not proof that all model error is caused by labels.

## 2. Non-negotiable separation of research and deployment

- Preserve `PI0253.selected_checkpoint=NONE` and `NO_ADMISSIBLE_CHECKPOINT`; never overwrite its profiles, selector, score files, hashes, or source membership.
- Set `deployment_selected_checkpoint=NONE` throughout PI0254. A separately named `research_candidate` may be identified only under section 8 and confers no execution permission.
- WBC, SafetyGate and its thresholds, ABI, H40, dt=0.05 s, decoder, embodiment mapping, C_T processor/statistics and physical target authority remain frozen.
- No smoothing, clipping changes, rotation projection added to model output, retiming, prefix conditioning, expert/future/source-ID inputs, target-XYZ inputs, action-space changes, new teacher distillation, loss search, learning-rate search, extra arm, or backbone unfreezing.
- No new sim/WBC episode, hardware, DAgger, RTC, DEV-E/F/Final-F read, automatic mail/monitor, PR merge, or broad historical cleanup. PR77 remains unmerged; it is a pointer only.
- Existing expert admission conflicts do NOT become an impossible prerequisite requiring zero expert REJECT before diagnostic TRAIN fitting. Keep and report those rows. Such conflicts DO prevent a claim of deployment readiness and must not be silently fixed by changing labels or discarding difficult windows.

## 3. S0 — Narrow source diagnosis and pre-query materialization

Use P0 receipts and existing arrays; do not redo the completed 704-row postmortem, entire history seals, or old model screens.

### 3.1 Initialization/rotation traceback, CPU/read-only

Trace the 13 already exposed expert/model-union windows back to the existing raw collector and expert/reference construction. Compare, in the same documented frame and time, achieved pose, desired/reference pose, first H40 label, observation timestamp, reset/settling order and the actual quaternion/rot6d convention. Check the early windows explicitly, including `obs-0000`, `obs-0006` and `obs-0013` where present. Reuse matching historical gate results. Follow only the implicated call chain and records, not an open-ended search of all old campaigns.

Deliver a table linking each claimed mechanism to an actual array/source statement. Where raw evidence is absent, state `UNRESOLVED_SOURCE_EVIDENCE` and stop that particular traceback; do not invent the missing pose or prevent independent fitting research. Propose, but do not implement, a versioned label/initialization contract repair if required. No additional CALIB predictions or CALIB-driven sample/loss design.

### 3.2 Coordinate statistic

The historical CALIB `min ||native-relative XYZ - world target||` is not a physical world distance. Preserve it as historical/invalid-for-physical-interpretation, not a quality objective. An additive CPU-only sidecar on existing arrays is authorized only where an actual, verified per-window native-to-world transform can be recovered from source evidence. Use that full documented transform; do not assume every transform is translation-only because the fixed live bridge added an anchor. Missing transform means `WORLD_REFERENCE_DISTANCE_UNOBSERVED`. Never call reference distance achieved TCP, rerank PI0253, or exclude REJECT rows to improve a summary. This sidecar is optional and is not a training prerequisite.

### 3.3 Freeze executable bindings before the first new model query

Publish a commit containing the P1/train/eval implementation, CPU tests, resolved run cards, immutable model/processor/runtime identities, cohort manifests, the two exact consumed-update schedules, actual evaluation-noise tensor hashes, failure accounting and this authorization reference. This is mechanical materialization within this authorization, not a fresh request for scientific permission.

All choices below must be materialized before observing P1 or any new baseline model output. No metric-dependent sample selection. Reuse original masks, units and metric definitions. Verify raw labels and the actual codec on the chosen training/evaluation support on CPU; log expert gate status rather than filtering by it. A confirmed wrong label/state/frame binding, non-finite data, missing required arrays, split violation or changed runtime/gradient contract is a real BLOCKED condition. Already known expert-versus-gate incompatibility with otherwise correct bindings is not such a block.

## 4. Fixed cohorts, all from the original TRAIN split

Do not read untouched evaluation splits. Use source-rollout membership, not only distinct window names, for disjointness.

- **O64:** original64, with the original old16 membership retained exactly. Original O64/B0/B1 G results and raw noise are reused.
- **P64:** 64 diagnostic probe windows outside the source rollouts represented by O64. Select 64 distinct TRAIN rollouts,16 in each A-D condition, and one window per rollout, allocating four windows to each preexisting phase quartile within each condition. P64 is excluded at source-rollout level from both new arms' training. FULL10000 has already been trained on these original TRAIN data, and C_T statistics are inherited; this is NOT a new blind test or an untouched global holdout.
- **N512:** select 128 additional TRAIN rollouts,32 per A-D condition, disjoint from O64 and P64 source rollouts. Select one window from each of the four preexisting phase quartiles of each rollout:512 windows. These windows and O64 form the exact **576-window training support** for both arms.
- **E64:** a fixed fitting audit subset of N512, from64 distinct N512 rollouts,16 per A-D and four windows per phase within each condition. It is trained on and must be labelled a fitting audit, not validation.

Deterministic selection: use SHA256 ordering with namespace `PI0254_COHORT_V1` separately for P64 then N512 then E64; record literal sorted candidate lists. For each condition, hash-order eligible rollouts; take the specified count; assign probe/E64 phases round-robin by selected-rollout order; choose the window nearest that phase's temporal midpoint with full-source-ID lexical tie-break. N512 uses all four phases. Eligibility depends only on original split, complete required source data and the disjointness rules, never model/target-distance/gate outcomes. Record jitter-family coverage descriptively; do not repeatedly redraw. If required cardinalities cannot be met, report the exact structural blocker rather than silently using shared rollouts or changing sizes.

These fixed design choices deliberately test repeated bounded support expansion, not another one-pass 9792-window run. They are not a claim that576 windows represent the entire original distribution or that the chosen rehearsal proportion is optimal.

## 5. P1 — 128 paired new-support diagnostic queries

Run only G2000 and the original FULL10000, on P64, with one presealed actual noise tensor per window shared between the two models: `64 x 2 x 1 = 128` queries. Both models use the exact inherited C_T/S4 inference contract, and outputs are saved at the same native and physical stages. Persist processed model-input hashes and the actual initial noise, not seed names alone.

Before training, write the signed P1 comparison: per-window XYZ and SO3 RMSE, first/last errors, mean-offset and centered-shape errors, all-row gate issues, and model-added versus expert-associated violations. Report paired differences, fractions improved, medians/p95 and A-D/phase/source-rollout aggregation. These are repeated TRAIN/development observations and a single training-family contrast; neither window count nor two noises is a count of independent training replications.

Interpretation:
- P64 improves while O64 regresses: evidence consistent with support transfer/interference, not proof of its unique optimizer mechanism.
- P64 and O64 both poor: broader fitting difficulty is supported; capacity, optimizer, sample exposure and observation sufficiency are still distinct hypotheses.
- Mixed results: retain strata and uncertainty instead of forcing a unique root cause.

P1 diagnoses the prior family. It does not pick a different initialization, alter the frozen U/R design, tune the rehearsal fraction, unlock a candidate, or stop the goal merely because direction is inconclusive.

## 6. P2 — Authorized two-arm fine-tuning

### 6.1 Shared numerical contract

Initialize **both U576 and R576 from the same original G2000 model weights**. Neither continues FULL10000 optimizer state, nor initializes from whichever old checkpoint looks best. Resolve G through its fixed receipts; its checkpoint tree is `cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226` and its original screen revision is `efecea485ae77000599d19171b02491f8d514f42`.

Both arms use fresh identical AdamW/scheduler states, lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5, max_grad_norm=1.0, batch1, accumulation1,10 warmup steps followed by cosine schedule to the fixed4608-step endpoint. Freeze the resolved scheduler implementation. This is an explicitly NEW4608-step schedule, not an exact continuation of the old10000-step schedule. Training RNG seed25402 in a new PI0254 namespace; each arm starts from the same RNG state. Schedule RNG is isolated from model/noise RNG. Couple actual sample-visit noise/time draws between arms where the same source/visit exists, while recording each arm's actual per-update identities. Do not assume a common seed alone proves coupled draws.

Keep original bf16/tf32/image settings, trainable parameter set (projector and diffusion model; no visual/LLM unfreeze), action/state mapping, statistics and raw-unit normalizations. Do NOT recompute C_T statistics on576.

Objective stays exactly:

`L_native + 0.4*ell_pos + 0.1*ell_delta + 0.1*ell_rot + 0.1*ell_speed_margin`.

Keep the original complete pure-noise S4 generation and gradients through all four Euler steps, all physical loss implementations/scales, original0.05-second timing and original speed margin. No new first-pose constraint, masking out label conflicts, endpoint weighting or coefficient tuning.

### 6.2 Exact budgets and exposure contrast

|Arm|O64 consumption|N512 consumption|Total actual successful optimizer updates|
|---|---:|---:|---:|
|U576 uniform repeated control|8 per window =512|8 per window =4096|4608|
|R576 rehearsal intervention|24 per window =1536|6 per window =3072|4608|

U576 is8 complete independently shuffled cycles of the576 windows. R576 is a fixed one-old/two-new interleaving, using24 shuffled O64 cycles and6 shuffled N512 cycles. Freeze permutations using the schedule namespace before P1. Do not place all old updates at the end, sample adaptively, or select hard examples from model errors.

Thus R576 allocates1/3 of updates to old support and2/3 to added support, versus1/9 and8/9 for U576. R has three times the old exposure and25% fewer N512 exposures. That trade-off is part of the intervention and must be reported. Both use the same support and same update budget; neither is a reproduction of the original9792-window FULL run. If both improve relative to FULL, do not attribute that shared improvement solely to rehearsal: support size, repetition and schedule also differ from FULL.

Maximum two scientific arms;9216 actual optimizer updates total, including any lost/repeated updates on engineering recovery. No third arm, opportunistic continuation, more epochs, ratio sweep or seed search. One initial submission and at most one science-neutral engineering resubmission per arm; exact available state resume only, with consumed work never erased from the raw budget. If recovery cannot complete the endpoint within that raw cap, retain the partial result and report the block rather than resetting the counter.

### 6.3 Logged training evidence and stops

At actual updates, record source/visit ID, native and physical loss components, learning rate, existing pre/post clipping gradient norms where accessible, non-finite flags, and trainable-group parameter update norms/dtypes at fixed steps1,64,256,1152,4608. Use the actual training computation; no extra model, per-component gradient-cosine, backward or optimizer query outside this ledger. Confirm the actual all-four-state gradient path on the first real update with hooks, not an extra smoke-training run.

Stop an affected arm on non-finite active outputs/loss/gradients/weights, wrong consumed identities, frozen-parameter drift, loss/gradient contract violation, forbidden input, or inability to save/recover within resources. Ordinary finite quality FAIL at1152 is an observation, not permission to retune or select that checkpoint. Do not continue an invalid arm just to exhaust budget. An independent valid arm may finish unchanged unless the error invalidates their shared contract.

## 7. P3 — Fixed evaluation and budgets

Evaluation noise: O64 uses original B0/B1. P64 and E64 use two new presealed noise families, generated with fixed namespaces `PI0254_EVAL_B0` and `PI0254_EVAL_B1` and seed25410/25411 respectively in the inherited runtime. Archive actual dtype/shape/bytes/hash before inference. P1 uses the P64 B0 family.

Before new training, supplement baselines with G on E64/B0/B1 (128 queries) and G on P64/B1 (64 queries). Together with P1, baseline/diagnosis costs320 queries. Reuse G/O64/B0/B1; do not rerun it. The original FULL10000/P64 comparison exists only for B0; do not pretend B1 was evaluated for it.

Evaluate each arm at exactly1152 and4608 on O64,E64,P64, each under both noises: `3 x 64 x 2 = 384` queries per milestone, `2 arms x 2 milestones x384 =1536`. Old16 results are derived from O64 with no new queries. No live side prediction or new CALIB model evaluation is authorized in this goal.

Planned new inference queries: `128 +128 +64 +1536 =1856`. Hard cap1920 including failures; the remaining64 are ONLY for clearly invalid/no-result engineering attempts, never another valid draw or additional scientific condition. At S4, the inference-head-forward cap is7680, accounting actual partial attempts. Track inference, native training forwards, S4 rollout forwards, backward recomputations and optimizer updates separately. No uncounted parity or sample queries.

Keep every fixed cell's validity and failure in the denominator. Report all-row metrics including REJECT trajectories where geometrically defined, alongside the explicitly labelled expert-nonREJECT stratum. No best-of-noise result, no prediction alignment in reported primary scores, and no world target metric without a verified transform. Centering remains decomposition-only. Perform independent CPU replay of saved arrays/metrics/gates, not additional GPU inference.

1152 is diagnostic-only. Only each arm's4608 endpoint may enter the research comparison. No intermediate winner, retry-until-pass or runner-up substitution.

## 8. Research interpretation and endpoint rule

Primary comparison is U576 versus R576 at equal4608 updates, with paired fixed-input/noise metrics. Show O64/old16 retention, E64 added-support fit and P64 not-replayed-this-stage performance separately. Report A-D and phase strata and all expert-associated versus model-added gate issues. Where intervals are shown, resample source rollouts, not raw windows; do not claim independent training-seed significance from this one paired experiment.

The optional label `PROMISING_RESEARCH_ONLY` requires all of the following, predeclared here:

1. Endpoint has all fixed valid evaluation cells and passes the exact inherited four O64/old16 x B0/B1 quality gates. Bind the original qualification code and full numeric thresholds before P1; do not reconstruct them from rounded report values.
2. For EACH noise independently, E64 median XYZ RMSE is at least20% below G, and P64 median XYZ RMSE is at least10% below G; each respective p95 XYZ and median SO3 is no worse than G. These are new practical research-screen thresholds, not historical criteria or statistically proven effect sizes. Compare unrounded numbers; only1e-6m/1e-6rad numerical comparison tolerance is allowed for the no-worse comparisons.
3. No model-only REJECT on expert-nonREJECT E64 or P64 cells under either noise. Keep expert-REJECT cells in all-row fitting/error tables and report their exact counts; this stratum is diagnostic, not a redefined deployment gate.

If both endpoints meet this research screen, choose the smaller worst-noise A-D macro-average of the four P64 condition-specific XYZ medians; exact tie goes to U576. If one meets it, that is the research candidate. If neither meets it, `research_candidate=NONE`. All endpoints and all failures remain in the report. This does not rerun or supersede the original CALIB selector.

Interpretation matters more than attaching a winner label: R preserving O64 by sacrificing new fit/probe is not a successful expansion; U and R both improving indicates the repeated bounded-support baseline itself matters; both failing E64 despite verified exposure keeps optimization/objective/observability open. More accurate expert matching can coexist with expert-associated gate rejects. No outcome proves physical task success, full9792-support competence, or multi-task generalization.

## 9. Artifacts, storage and closeout

Create a new PI0254 root and branch; leave old artifacts untouched. Download only the fixed G/FULL10000 model parts and small evidence required for this goal, not all six Trainer families or retired weights. Preserve the ability to distinguish model-only from full Trainer states.

Plan serial arm execution and task-local storage before jobs. Use current filesystem space and bounded write/checkpoint tests as evidence of what was tested, NEVER as proof of quota headroom. Quota may remain UNKNOWN; report that residual operational risk. No global purge or claim that df free is personal quota. Stop on quota/write failure with a durable last valid state rather than silently dropping optimizer/RNG state.

Keep a rolling complete recovery checkpoint while an arm runs. Retain exact1152 model snapshots for the fixed evaluation, plus the two final4608 complete Trainer checkpoints. A temporary1152 full recovery payload may be retired only after the same arm's final complete checkpoint has fixed-revision HF content/SHA readback and no process needs the temporary state. Keep all small failure evidence and training/evaluation records. Archive new payloads under a unique PI0254 prefix in the existing authorized HF project; no overwrite of old revisions. Only explicitly inventoried, newly created PI0254 temporary files may be cleaned after verified durable replacement. No old training data or P0 evidence deletion.

Final deliverables: SOURCE_CAUSE_REPORT; P1 paired report; sealed cohorts/noises/schedules/run cards; both arm training ledgers;1152 and4608 evaluation arrays and CPU replay; equal-budget comparison and research-candidate decision; immutable HF model/Trainer receipts; final resource/query/update counts and remaining limitations. If a real blocker stops execution, preserve a partial evidence-backed closeout and mark it BLOCKED, not training-complete.

Canonical report goes to Issue11. PR77 may receive one pointer to the canonical authorization/closeout, never a merge. Do not complete this goal merely by posting a plan or by finishing P1 when the shared contract is valid: proceed to the two bounded fine-tuning arms and fixed evaluations under this GO. Publication and remote execution remain distinct states.

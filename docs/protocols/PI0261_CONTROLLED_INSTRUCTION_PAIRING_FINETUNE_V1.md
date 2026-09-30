# PI0261 — Controlled instruction pairing after numeric Motion2 qualification

Status at publication: AUTHORIZED_SCIENTIFIC_EXPERIMENT_NOT_EXECUTED.

Authority: `PI0261_CONTROLLED_INSTRUCTION_PAIRING_FINETUNE_GO_V1`.

## 0. Objective, evidence and ownership

One lead owns read-only diagnosis, implementation, real canaries, both training arms, fixed evaluation, CPU replay and archival. Optional assistance is not a signoff dependency. Complete the authorized chain whenever its actual correctness checks pass; do not stop at a plan, a diagnostic report, DATA_READY or a queue entry.

Sources reviewed for this decision:

- PI0260-R3 canonical: Issue #11 comment5903045678.
- Result commit: `a5220086290a5c90d1440cfbfa6bb59c1babf78f`.
- `reports/PI0260/R3/MODEL_QUALITY_REPORT.md`, `FINAL_CLOSEOUT.json`, saved semantic replay rows, and `SECONDARY_SAVED_ARRAY_REPORT.json`.
- Actual source: `scripts/pi0260_r3_train.py`, `scripts/pi0260_r3_eval.py`, `scripts/pi0260_r3_eval_card.py`, `scripts/pi0260_r3_secondary_cpu.py`.
- Parent learning authority: `fd8e69b59a16aa90b1761d23b3a8f5a78cee8763`, PI0260-R3 alias-aware protocol.

The latest canonical and report agree on these final scores:

| Point | Numeric KIN-VAL task/noise groups | M1 instruction B0/B1 | M1 paraphrase B0/B1 | M3 instruction B0/B1 | M3 paraphrase B0/B1 |
|---|---|---|---|---|---|
| Base6000 | 4/4 PASS | 7/12,8/12 | 6/12,7/12 | 12/12,11/12 | 12/12,11/12 |
| G6000 | 4/4 PASS | 8/12,8/12 | 6/12,7/12 | 6/12,7/12 | 6/12,6/12 |

Base final KIN-VAL XYZ medians are5.3–8.6mm and p95s11.9–15.1mm; G medians7.3–8.9mm and p95s14.1–20.5mm. Both meet the original position/posture/zero-REJECT gates. The original complete qualification still FAILS because each instruction and paraphrase group requires10/12. G M3/B0 is command0 6/6 and command1 0/6, not the reverse. Base M3 has positive evidence for both commands and paraphrases. Do not describe all language conditioning as absent.

The semantic replay compares predicted H40 XYZ RMSE to the original own-command and opposite-command references: margin = d_opposite - d_own; strict margin>0 is correct. This is an entry/reference discrimination test, not full target arrival or completion of the270-degree contour. A numerically close prediction may still be closer to the wrong reference. Preserve the actual frozen rule; do not substitute a normalized XYZ+SO3 classifier or a newly invented endpoint predicate.

R3's empirical full-corpus position-floor RMSE is about1.47mm TRAIN/1.43mm KIN-VAL. No mixed-command input alias or processor merge was reported. Within-command temporal ambiguity remains admitted and measured; it neither identifies the cause of a cross-command preference nor proves that every task is fully observable. No floor is subtracted from primary errors.

Hypotheses to distinguish: insufficient training exposure where language is necessary; state/visual shortcuts in ordinary trajectory windows; command/wording bias; task interference; limited use of available language features. These are hypotheses, not established unique causes. The first experiment below changes sampling emphasis, not the loss, visual encoder or task definition.

Publication reviewed repository evidence, not the remote raw tar/video or live machines. No model/scene/physics execution is asserted here. Reuse unchanged accepted source receipts, and verify newly materialized bytes against them.

## 1. Scientific decision and immutable boundaries

Start BOTH new arms from the SAME PI0260-R3 Base6000 model, chosen prospectively as a research initializer because its M3 instruction AND paraphrase groups pass. It is NOT a deployed or globally selected model. This does not prove it dominates G in every metric; do not replace it after inspecting new results.

Keep the existing COMPACT_30_10 dataset, actual TRAIN/KIN-VAL membership, labels, processor/statistics, cameras, FK/reference/IK, active masks, H40/.05s, action interface, allowed observations, S4 sampler and physical G objective. No new rendering or data-generation program. No scene/TEST substitution, label averaging, temporal retiming, explicit or disguised phase/target/source inputs, or new instruction templates.

The new controlled factor is the fraction of updates supervised on two OPPOSITE commands with IDENTICAL nonlanguage observations. Both arms process two examples per optimizer update with the mean loss. This shared effective-batch change versus R3 is explicit; common improvement over R3 cannot be attributed uniquely to pairing.

- `C2000`: ordinary phase-balanced windows, both commands represented in each update, but their states need not match.
- `P2000`:50% of updates use a same-observation opposing-instruction pair;50% use exactly the ordinary examples of C for the corresponding update.

Each arm runs2000 canonical optimizer updates. There is no third arm, ranking-loss branch, prompt search or backbone-adapter sweep. Both begin from identical donor weights and fresh optimizers. R3 history and FAIL/NONE are immutable.

## 2. S0 — Finite saved-evidence diagnosis, not another indefinite prerequisite campaign

Refresh current Issue #11 ledger, actual working tree, source identity, process/GPU ownership and scheduler state. Adopt healthy unrelated work without interference. Use an isolated PI0261 execution/output directory, existing source tools and one dispatcher. No broad cleanup, new orchestration service or peer-wait state.

### 2.1 Reconstruct the actual semantic result

Reuse the four completed440-cell CPU receipts and actual saved arrays. Resolve the exact executable semantic checker from the sealed source/manifest; do not guess its identity from a filename. In one source-bound CPU pass:

- Confirm own/opposite pairing by family, behavior, command, actual shared state/image and reference index/time, and restore the unchanged margin sign and tie rule.
- Test the designated expert as prediction: its own reference should beat a distinct opposite reference. Swap labels/commands as a negative control; the corresponding margin changes sign. The same prediction against a shared opposing pair cannot receive two strict-positive nearest-reference wins. If references are indistinguishable, report that preexisting evaluation issue, not a model error or a reduced denominator.
- Check all semantic/paraphrase examples use the prescribed t0 H40 entry semantics. Do not impose full-contour completion on1.95s or use a future label for another phase.
- Report each final model by task, command, noise, wording, family: own/opposite XYZ errors, margin and expert reference separation, along with full gate/rotation information. Distinguish wrong-reference preference, exact tie and any genuinely invalid computation.
- On existing empty-language rows, compare outputs only to exactly matched observations/noise/reference. Unmatched empty-language and ordinary cohorts are not a causal ablation. Do not add calls to fill missing comparisons.
- Reuse actual condition audit and confirm opposite commands remain distinct token conditions. Same RGB/state with different instruction is intended, not a temporal-alias error. Confirm condition cache keys include language; do not infer a cache bug just from a biased result.

Wrong family/label/token/cache/scorer wiring is a true correctness problem. Repair only a demonstrated implementation defect with versioned affected descendants; never rewrite old conclusions silently. A defect changing the scientific meaning of the requested contrast blocks the affected new experiment for a specific decision. No such defect is asserted by this publication.

### 2.2 Audit actual R3 exposure, without asserting an unmeasured shortage

Use the actual6000-visit plan and sample audits to count per-task/command/family/phase visits, common-start visits, repeated nonlanguage conditions carrying both commands, and instruction-template exposure. Count a useful opposing pair by observed tensors, not just equal task totals. Report whether high aggregate command balance actually provided repeated same-state comparisons.

Check the full-view96x96 evidence already admitted in R3; review only source cases necessary to interpret a new discrepancy. No new H0 design or all-history rehashing. Absence of command-mixed aliases does not prove the network uses the language, and low ordinary-window error does not by itself prove a state shortcut.

The output is a small diagnosis table and the frozen PI0261 manifest. If no new integrity defect is found, proceed to this preauthorized intervention even if the diagnosis is mixed. Do not stop simply because a unique neural mechanism has not been proved. Accepted same-command time aliases remain accepted.

## 3. S1 — Construct a TRAIN-only paired curriculum, with an explicit control

### 3.1 Common-start pair bank

Use ALL30 already admitted TRAIN geometric families and BOTH tasks M1/M3. For each of the60 family/task combinations, bind both commands at the original stored common-start index0 using their original allowed TRAIN instruction text and original H40 labels.

The two examples must have byte-identical actual processed nonlanguage inputs: pixels/image grid, current state and embodiment. Language tokens must implement the two distinct requested commands. Verify equality at the actual conditioning boundary, excluding action labels/masks and command metadata. State/frame/label identities remain the original dataset identities. All60 pairs must be actual usable original records with meaningful original own/opposite XYZ reference separation; no generated state, invented labels, new images or VAL borrowing. If a purported index0 pair is missing or inconsistent, investigate the existing source; do not silently replace the family or index to improve scores.

No new vocabulary is trained in this first intervention. Retain the original TRAIN templates. Original reserved KIN-VAL/paraphrase strings, target coordinates and evaluation cases are not introduced into optimization. This intentionally tests transfer to the existing alternate wording rather than memorizing it.

This pair bank is a temporal reweighting of the existing dataset, NOT new independent trajectories. H40 starts do not represent full task episodes, and a successful paired-training probe is not unseen-task generalization.

### 3.2 Freeze2000 paired-update units per arm

Use one deterministic plan under namespace `PI0261_PAIRING:26101`, with seed26101 and recorded serialization. Build500 four-unit blocks, containing one each of M1/PAIR_SLOT, M3/PAIR_SLOT, M1/REPLAY_SLOT, M3/REPLAY_SLOT. Seeded order within blocks may be shuffled before outcomes. Thus each task has1000 units, of which500 are PAIR_SLOT and500 REPLAY_SLOT.

Assign families using deterministic shuffled cycles through the30 TRAIN families per task; visit counts differ by at most1 over each task's1000 units. Family/task/command membership is identical between C and P at the same unit. Alternate microbatch order c0,c1 versus c1,c0 in a predeclared balanced fashion; no first/second-command update privilege.

For every unit, prepare the C ordinary pair: one usable active window from each command's sequence in that same family. Cycle motion quartiles evenly, then draw a valid window using the frozen keyed RNG. Keep terminal-only padding/holds out of the ordinary training pool exactly as in R3. These two examples need not share the same state. Incidental common starts sampled by this rule remain; do not remove them to exaggerate the contrast.

C uses this ordinary pair in ALL2000 units. P uses the exact same ordinary pair on the1000 REPLAY_SLOT units, and substitutes the fixed index0 common-observation pair of the same family/task on the1000 PAIR_SLOT units. Nothing is substituted after training or evaluation begins.

Per arm:4000 supervised example evaluations,2000 per task and1000 per task/command. P has2000 common-observation example evaluations plus2000 ordinary examples. Both arms have equal updates, effective batch, family/task/command counts and total examples; their within-trajectory window distribution intentionally differs. P spends half its example budget on entry semantics instead of ordinary coverage: report this tradeoff, not a free augmentation.

All original examples/failures remain in the data archive. Sampling weights are changed prospectively; no data is deleted or relabeled. Neither errors on KIN-VAL nor alias statistics determine the schedule.

### 3.3 Same-weight paired loss and actual random inputs

Each unit processes two sequential microbatches at the SAME parameter values, and makes ONE optimizer step after accumulating the mean gradient:

`L_update = (J(example_c0) + J(example_c1))/2`.

Use microbatch1/gradient accumulation2, or an exactly tested equivalent explicit two-microbatch wrapper. Backpropagate each half-loss, clip the accumulated gradient once, then optimizer/scheduler step once. Do not update parameters between commands. Counting two microbatches as two canonical updates would change the experiment and is not allowed.

Both arms use identical named random packets for a corresponding unit. Within an update, use the same actual free-S4 initial noise and native tau/epsilon for both commands; label-dependent noised actions will legitimately differ. Draw packets independently of labels, task choice quality and hidden time, from frozen unit-keyed RNG. Record actual arrays/digests, not only seed names. Shared seeds alone are not proof of identical draws when token shapes or code branches differ. Keep other model modes and dropout behavior consistent with the pinned runtime, documenting what is shared.

Nonlanguage equality is required within P's PAIR_SLOT; it is NOT required for C or P's ordinary pair. On REPLAY_SLOT, full processed inputs and labels also match across arms. Audit the expected relationships separately so intentionally different inputs do not create a false engineering blocker.

## 4. S2 — Fixed donor, optimization and real recovery

Single donor for BOTH arms:

- Repo `leoyuer/pi0246-gate-pass-campaign`.
- Fixed revision `52abc8d151b67bb667e43fcf874dfe2cfa253364`.
- Prefix `arms/PI0260/R3/Base/checkpoint-6000`.
- Reported tree `24ed9d1b5a0e42e0aff6d62c54d8438d2f202d765b006d90c62d8623604e4ce5`.

Use the actual full16-file receipt and original processor/statistics/header reconciliation. Do not load the older unadapted Base, G2000 or G6000 as a substitute. Initialize weights from this donor but use a NEW optimizer and2000-step scheduler in each arm; this is not exact continuation of the old optimizer. Exact resume applies within each new arm.

Both arms:2000 canonical updates; fresh AdamW LR1e-5, betas(.9,.999), eps1e-8, WD1e-5, global accumulated gradnorm cap1, warmup10, cosine2000. Microbatch1, accumulation2. Same BF16 frozen backbone and FP32 trainable projector/diffusion scope as R3. Do not unfreeze the visual/LLM/vlln backbone, add LoRA, quantize, or modify action decoding.

Keep J exactly the inherited objective: native flow matching +.4 physical position +.1 physical increment +.1 physical rotation +.1 peak-speed margin over original fullS4 free generation. No contrastive/ranking term, temperature, auxiliary classifier or coefficient sweep is added. This experiment isolates semantic-entry exposure relative to its matched ordinary-window control; it does not test every possible language-loss design.

In each arm, real continuous8 updates versus4+full save+fresh-process restore+4 test the new two-microbatch cursor, pending gradients, random packets, loss scaling, model/optimizer/scheduler and mode state. Save only at unit boundaries. Freeze exact/numeric comparisons before the canary; do not relax them after a mismatch. These32 completed canary updates are discarded and main runs restart from the donor. Reuse existing data/header/norm checks, not an old batch1 recovery verdict as proof of new accumulation correctness.

Use narrow CPU tests before any GPU canary: two half-gradients equal the tested mean-loss gradient; exactly one optimizer/scheduler step per unit; interrupted mid-unit does not silently skip or double-update; paired packets really match; metadata stays out of conditioning. Publication is not a claim that these tests are implemented or passed.

Sequential C then P is the default. C results cannot change P's schedule, recipe or donor. A valid quality FAIL of one arm does not cancel the other matched arm. Shared integrity/nonfinite failures block affected descendants. Save complete Trainer1000/2000 and maintain a durable exact-resume source; no interim quality screen or checkpoint rescue.

Log individual already-computed losses, norm/update magnitude, task/family/command/index and paired-gradient accumulation. Do not add unbudgeted model probes merely to produce more diagnostics.

## 5. S3 — Fixed evaluation that separates entry learning from ordinary fit

### 5.1 Inherit all original case identities and noise

For donor, C2000 and P2000 use exactly the R3 common processor, all220 original cases (156 Motion2 +64 O64), and B0/B1 actual saved packets. Each model point has440 logical rows. Keep original source IDs/labels, semantic/paraphrase strings, timestamps, frames, raw masks, gate and score rules. Reuse the donor's complete R3 Base6000 raw outputs when every identity matches; do not run them again to get favorable noise.

The original KIN-VAL panel and semantic families have informed this new design; explicitly label them DEVELOPMENT VALIDATION, not blind or untouched generalization evidence. No old TEST/DEV-F or new scene is accessed. A new independent test/domain-robustness experiment would require its own future authority.

### 5.2 Add one complete TRAIN entry panel, not a new validation search

For each of the60 TRAIN common-start pairs, evaluate both commands at both B0/B1 for donor/C2000/P2000. This is240 logical rows per point,720 total. It measures whether the intervention can learn instructed choices on its actual TRAIN support. It is not independent validation, nor a new sample of60 unrelated robot episodes. If a tuple already exists with identical model/condition/noise/decode identity, reuse its forward but retain both display/index references and original denominators.

Total logical evaluation:3*(440+240)=2040 rows. Donor's original440 rows are reused, so planned NEW queries are at most1600 before further exact cache reuse:240 donor TRAIN entries +680 C-final +680 P-final. Extra strictly no-valid-result attempt reserve is128. Hard cap1728 new attempts/6912 inference head forwards. Cached rows do not create an allowance for new scientific cases. No new G6000 queries or step1000 screening.

Freeze this complete evaluation manifest before the first new quality query. The donor TRAIN panel may be run before training to complete diagnosis; it does not authorize subsequent schedule or recipe changes. A good TRAIN semantic score does not cancel the controlled trial or convert its VAL into training.

### 5.3 Preserve primary gates and add interpretation, not corrected scores

Each behavior/noise original KIN-VAL group still requires median XYZ<=.020m, p95 XYZ<=.035m, median SO3<=.050rad and zero REJECT. Each original instruction AND paraphrase group still requires at least10/12, separately. Reference-margin ties remain not correct. Do not change labels, use nearest-of-many experts, time-align/DTW, average aliases, choose a better noise or smooth predictions.

Report original four numeric groups and eight language groups, not only pooled accuracy. Report per-command wins, ties, negative margins, magnitude relative to original reference separation, and count of fully correct opposing pairs. Report both TRAIN entry performance and original validation; a higher TRAIN score alone is not the goal. Keep invalid/rejected rows in outcome denominators.

Use saved predictions to measure the change caused by swapping instructions at exactly the same nonlanguage inputs and noise. Reference contrast, not mere output inequality, determines correctness. For saved matched empty-language cases report the available comparison honestly, with no new calls. Shared-prefix H40 semantics remain entry/target-direction discrimination, not a full M3 trajectory or physical task completion.

Numeric performance must remain qualified while language improves. If P only improves entry accuracy by damaging ordinary KIN-VAL fit, that is a tradeoff/FAIL, not a success. O64 is a disclosed regression panel under the synthetic-domain processor; it does not inherit or regrant historical G qualification.

Existing accepted temporal aliases stay measured; use the same labels and ambiguity diagnostics. The small full-corpus floor is neither a semantic explanation nor a number to subtract from subset RMSE.

Complete saved-array CPU metric/gate/semantic replay for all new outputs and reuse matching donor receipts. Produce a compact side-by-side donor/C/P index with numerical scales, correct/opposite instruction results and explicit KINEMATIC/OFFLINE/NO PHYSICS labels. Use only existing arrays: no model or render call for video cosmetics.

### 5.4 Research candidate and stop

A final endpoint is research-qualified only if ALL inherited numeric and language groups pass. If only one passes, record it under `synthetic_research_candidate`; if both pass, prefer C as the simpler ordinary-window continuation. If neither passes, research candidate stays NONE. Do not select a separate model per task or an intermediate endpoint.

Regardless, `selected_checkpoint=NONE` and `deployment_selected_checkpoint=NONE` remain. Passing these heavily developed synthetic tests is not independent physical/general-purpose readiness. New learned-policy WBC, actual physics or hardware is NOT authorized here.

## 6. Engineering autonomy, accounting and resource allocation

This is a NEW bounded scientific experiment, not a reset of PI0260-R3 counters. Archive R3's reported12000 main updates/12064 total completed updates and1724 inference attempts/6896 heads separately. New allowances below do not extend the old experiment or permit a third arm.

| New PI0261 activity | Limit |
|---|---:|
| New scene/IK/render/data-statistics/label generation |0|
| Main optimizer updates |4000 total,2000 C +2000 P|
| Main supervised microbatch example evaluations |8000 total,4000 per arm|
| Discarded real canary optimizer updates |32|
| Extra completed-update engineering recovery reserve |256|
| All completed optimizer updates |4288|
| Raw training action-head forwards |100000, including partial/failed work|
| Total logical evaluation |2040, with440 donor rows already archived|
| Planned new quality queries |<=1600 before exact cache reuse|
| New quality attempts / inference heads incl. reserve |1728 /6912|
| WBC/physics/TEST/old DEV-F/hardware |0|

No arbitrary limit on the number of qsub/import/IO repair attempts replaces these actual resource bounds. Pure engineering work may be repaired autonomously with narrow tests, actual consumed-compute accounting and exact recovery. Valid scientific failures cannot be retried for a better score. Reserve updates restore interrupted valid work; they cannot extend a completed2000-step arm. Before spending reserve, confirm remaining steps/heads/space can finish the admitted work; if not, preserve state and report an exact budget gap.

PC coordinates one run ledger. Corelab owns CPU processing, no-gradient inference and presentation. Hopper owns the real canaries/training under a lawful compatible GPU allocation. Use tested memory/runtime requirements, not a host-name preference or blind propagation of a hardcoded100GiB assertion. No known-undersized allocation or new low-precision trick; any change to capacity guards must be justified by the actual unchanged workload and captured peak memory during the authorized canary. Ordinary CPU/debug tests do not go on H200.

Refresh actual scheduler/process state before submission. Missing qstat is UNKNOWN, not an empty queue. Keep one authoritative dispatcher, adopt existing healthy jobs, do not kill unrelated work or duplicate qsub. Optional helper can inspect sealed artifacts and return patches; it cannot own another scheduler or block the lead's substantive checks.

Preserve donor, R3 Base/G, current full resume and unique evidence. Download only necessary model views; preserve complete durable Trainer sources. Use inherited next-write overlap and10GiB reserve; df is not quota. Reuse fixed-HF verification for unchanged artifacts; upload only authorized new weights/metadata/results, not private meshes/URDF/WBC/tokenizers/credentials. No broad cleanup, mail, monitor, daemon or PR merge.

## 7. Completion and longer-range decision

Terminal only when both actual2000 endpoints, full planned evaluation, paired-input/resume integrity, CPU receipts and durable archives exist:

`PI0261_INSTRUCTION_PAIRING_PILOT_COMPLETE — STOP_FOR_GENERALIZATION_OR_PHYSICAL_ADMISSION`.

Quality PASS/FAIL is separate from execution COMPLETE. A single arm or a missing mandatory panel is PARTIAL. The lead proceeds without another routine PI GO from S0 to training once concrete checks pass; familiar temporal ambiguity, lack of helper signoff or video cosmetics are not blockers.

Interpretation after the fixed experiment (roadmap, NOT extra authorization):

- P improves TRAIN and validation language while numeric groups remain qualified, beyond C: evidence supports higher exposure to instruction-disambiguating conditions under this setup. It is not a proof that language is the only cause or a novel method claim.
- Both improve similarly: extra updates/shared minibatch design may suffice; do not attribute common gains uniquely to pairing.
- P improves TRAIN entry but not validation/paraphrases: likely generalization/grounding gap remains; next study could vary language/visual adaptation with a new control, not replay the same validation until passing.
- Both fail even on TRAIN entry despite expert positive controls: next diagnose actual conditioned features and gradient use, and preregister one limited adapter/objective intervention. No unfreezing is silently added here.
- Numerics regress: measure the semantic-versus-trajectory tradeoff; do not relax the original gates.
- A synthetic endpoint qualifies: next priority is independent new-family/appearance evaluation and a separately admitted frozen-WBC compatibility test, not yet another O64-only fit run. Proper feedback execution still needs measured observations and controller capacity evidence.

Immediate goal: determine whether direct, balanced same-observation language contrasts improve correct task choice while preserving the now-established geometric skill. No further camera, IK, corpus-generation or storage-governance campaign is required to ask this question.

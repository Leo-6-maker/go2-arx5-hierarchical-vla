# PI0264 — Matched language-diversity training and prospectively sealed transfer challenge

Authority: `PI0264_LANGUAGE_DIVERSITY_TRANSFER_GO_V1`.
Publication state: `AUTHORIZED_NOT_EXECUTED`.
One lead owns the complete experiment. Optional helpers do not sign off mandatory gates. This is a new allocation, not permission to resume or reinterpret PI0263.

## 1. Evidence, progress and the question

Read PI0263 canonical Issue #11 comment5920151156 and evidence commit `539c175ff9f955a8b49bcaf7eedb29c606d538eb`: `reports/PI0263/final/MODEL_QUALITY_REPORT.md`, `FINAL_COMPLETION_AUDIT.json`, actual plan, training audits, source manifests and `scripts/pi0263_train.py`. Preserve the old protocol `1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28` and all original results.

PI0263 is COMPLETE, with F/X valid quality FAIL. Both retain all four legacy numeric groups and M3 qualification. X improves legacy language groups to7/8 and crossed-M1 groups to5/8, versus F5/8 and2/8. X original TRAIN T bindings are60/60,60/60 and other-T bindings60/60,58/60. The newly supervised recombinations are not unseen-language tests. X's crossed KIN-VAL counts are T0 19/20,19/20; T1 16/20,17/20; V0 18/20,14/20; V1 16/20,17/20. X TRAIN V1 remains40/60 at B0 and56/60 at B1. These are correlated family/template/noise conditions, not independent trials.

The prior counterbalanced assignment helped; it did not universally improve transfer. F also improved with extra optimization, X is worse than F on KIN-VAL V0/B1, and familiar-template scene failures remain (including reported family142/148 failures). Do not claim a unique neural cause, that language is the only remaining problem, or that WBC caused these offline failures. The report's CPU scope/loss-reader errata preserve raw evidence and do not authorize changing model outcomes.

**Question:** with the same visual/state/action support and matched computation, does increasing TRAIN instruction diversity from two to eight balanced equivalent formulations improve target choice for supervision-held-out wording, including newly sealed formulations and noise packets, without losing geometric accuracy or M3?

This directly follows the completed T0/T1 recombination experiment. No further old-phrase weight sweep, image/camera/IK rebuild, new loss, backbone adaptation or prompt canonicalizer is authorized. Broad language or scene coverage may still be needed later. This is one controlled intervention, not a promise of a generalist/deployable model.

Publisher scope: repository reports and actual training source were read; headline sums and the new factorial/budget arithmetic were recomputed. The publisher did not SSH, download/replay the complete remote raw tar, examine every image, load models or implement/run this experiment.

## 2. Fixed donor, data and deployment boundary

Both new arms start from **PI0263 X2000** weights:
- HF repository `leoyuer/pi0246-gate-pass-campaign`;
- revision `a2f9f767a11e4af269377603f3da432f4931d623`;
- prefix `arms/PI0263/X/checkpoint-2000`;
- complete Trainer tree `4ec3b3a44ad40a057cfdcd7ad992a86b68889b4efba98be70ed109e571497694`.

It is a fixed research initializer, not a selected/deployable policy. Use weights with a fresh optimizer/scheduler; do not substitute F, P, G, or an intermediate checkpoint after observing results. Preserve donor files and actual shared dependencies from the storage subagent's cleanup.

Reuse the frozen COMPACT_30_10, all original images/states/actions/masks, processor/tokenizer/statistics, frame/TCP/embodiment contract, H40/dt=.05, S4 and inherited precision. New language variants are new conditioning, not new visual/action data. All M3 training text stays original. Only the same authorized TRAIN data enters optimization. No old or new validation labels enter optimization or normalization. No source/phase/target-coordinate/template-ID metadata is model input.

No new scene, camera render, IK, physical step, WBC episode, TEST/old DEV-F, robot or contact task is authorized. New text/noise challenge uses only the already-authorized ten KIN-VAL families; it is NOT a new geometry split or blind physical TEST. Formal selected/deployment remain NONE.

## 3. Fixed language partitions: no prompt search

Only substitute `{color}` using the original requested target's semantic color from the registry, not a guessed command-color mapping. Every instruction preserves the original position-plus-orientation goal; no speed, approach-side, force, gripper, timing or extra motion-stage requirement is introduced. Keep punctuation/case/spacing as frozen below and use the unchanged tokenizer.

### 3.1 TRAIN language pool

T0: `Approach the {color} target and align the tool with its axes.`
T1: `Bring the tool to the {color} marker with the pose shown by its axes.`
T2: `Position the tool at the {color} target and match the orientation of its displayed axes.`
T3: `Guide the tool to the {color} marker and align it with the marker's axes.`
T4: `The {color} target is the destination for the tool; match the orientation shown by its axes.`
T5: `Match the tool's position and orientation to the {color} marker and its displayed axes.`
T6: `Bring the tool into alignment with the displayed axes at the {color} target.`
T7: `Use the {color} marker as the tool's target position and use its axes as the target orientation.`

T2-T7 are the six newly authorized supervised formulations. They are predetermined by this PI, not generated or selected by model scores. The producer checks target/color semantics and tokenization in CPU preflight; genuine contradictions are reported, not fixed with result-driven wording replacements.

### 3.2 Old development formulations

The exact archived V0/V1 (`Move to ...`, `Reach ...`) remain excluded from supervision/statistics and remain in the old development evaluation. Do not silently count lexical overlap or prior base pretraining as proof of project exposure. They have informed research design; they are not independent blind tests.

### 3.3 New prospective wording challenge

Q0: `Place the tool at the {color} marker, oriented to agree with the axes shown there.`
Q1: `The tool should go to the {color} target with its orientation matching the displayed coordinate frame.`
Q2: `Take the tool to the {color} marker and give it the orientation represented by the marker's axes.`
Q3: `Make the tool coincide with the {color} target in position and with its shown frame in orientation.`

Q0-Q3 never enter training, normalization, schedule tuning or a prompt rewriter. Seal their exact rendered text, source observations, labels and noise arrays before any new canary. Do not view ANY Q-panel model quality, including donor quality, until both final checkpoints have been frozen. Tokenization/shape/hash and semantic meaning checks without model calls are permitted. No replacement for awkward or poorly performing wording after results.

These four PI-authored texts are a small prospective challenge, not representative natural-user language or necessarily unseen foundation-model language. The same KIN-VAL geometries have already informed development. Report the novelty along the text/noise axes only.

## 4. Matched arms and compact exact-factorial source schedule

- **B2048:** balanced two-template continuation control, T0/T1 only for M1.
- **L2048:** eight-template diversified supervision, T0-T7 for M1.

B is the NEW two-template control; do not confuse it with historical raw Base. Both start from X2000. All nonlanguage data, labels, source visits, model parameters at initialization, sample/update totals, two-command grouping and actual randomness match across arms. Only M1 text assignment differs. Diversity necessarily reduces individual T0/T1 exposure; the comparison estimates that whole policy at fixed compute, not diversity at fixed per-phrase counts.

### 4.1 Source bank

Use PI0263's actual frozen `PAIRING_PLAN.json` (reported plan SHA256 `602dc213ca91f2dcd5099bd97a45587fe5fe1e0d205243596d48e07c5f662617`). Resolve actual files from its source manifest. Collapse its two visits into the 1000 original base units; verify repeated units have identical source/label bindings.

Partition base units by task(M1/M3) and slot(PAIR/REPLAY): four strata. In each:
1. include the lowest original unit ID for each of the30 TRAIN families;
2. for every original phase represented in that stratum but not yet covered, include the lowest unused original ID in that phase;
3. fill with lowest remaining original IDs until exactly64 units.

No score, margin, failure or validation information is used. Phase metadata must be resolved from existing records, not guessed. This fixed bank has256 units,128 per task,64 per task/slot; all30 TRAIN families and existing global phases are covered. It does not promise each family's every phase, equal color frequencies or 64 distinct images: common-start records may repeat. Report actual unique observations, replay-stage coverage and family frequencies. If the real schema contradicts the asserted existing coverage, resolve the provenance/schema from original files rather than select new data.

Each unit retains both original command records and their expert labels; common-start observations match across commands, ordinary replay need not. Reuse the original valid P/X-selected records; no new timestamps/windows.

### 4.2 Eight visits, interleaved over the whole bank

Sort bank by original ID. Execute eight full passes v=0..7 over it, not eight immediate repeats of one state. There are256*8=2048 optimizer units per arm. Unit identifier u=256*v+bank_index+1.

For each M1 base unit, define offset r as integer(SHA256(`PI0264:26401:text-order:<original-unit-id>`)) mod8. Use least-significant modulo of the full digest; freeze the rendered plan.
- L uses T[(v+r) mod8].
- B uses T[(v+r) mod2].
- Within a unit BOTH commands use that template with their respective correct colors.
- M3 retains original sequence text in both arms.
- Microbatch order c0,c1 on even v and c1,c0 on odd v, identical across arms.

Thus every chosen M1 observation/target/label gets each of the eight texts once in L and each of T0/T1 four times in B. Per arm:4096 supervised microbatches,2048 M1 and2048 M3, half PAIR/half REPLAY. L has256 M1 examples per template; B1024 each T0/T1. Task and command frequencies match. Counts and equality are checked at the actual processor/collator, not inferred from text manifests.

This new source subset/repetition and2048-step horizon are common changes versus old PI0263. Historical X-to-L gain alone does not isolate language diversity; B-to-L is the matched contrast.

### 4.3 Actual randomness and conditioning

Use the mature actual native epsilon/tau/free-S4 packet mechanism with a new fixed `PI0264:26401:<u>` namespace, independent of template, label, correctness and arm. Corresponding units share real packets across arms and both commands. Different visits may differ. Save arrays/identities; same nominal seed is not the sole evidence when text lengths differ.

Nonlanguage and labels/masks match across arms; new language token IDs/masks may intentionally differ. Strip all audit metadata before model forward. The real supported batch/input schema is unchanged. Deterministic image processing remains; no crop/flip/color augmentation.

## 5. Finite preflight, canary and optimization

Preflight is one source-bound CPU pass: donor/data identities; all rendered texts/colors; no V/Q supervision; unit/microbatch exposure counts; condition and label parity; actual token length/attention masks with no truncation of target/orientation clauses; source bank and sealed evaluation card. Negative controls cover wrong color, leaked template/target IDs, skipped or duplicated visits, false cache reuse and wrong half-loss scale. Reuse the accepted geometric/codec/temporal-alias audits; do not rebuild old H0/H1 or rerun PI0262/3.

Known same-command temporal ambiguity and MIXED interpretation are accepted limitations, not stop conditions. A true identity, masking, coordinate or data-leak fault blocks only affected descendants until resolved. Failure to name one neural mechanism is not a blocker. The lead proceeds when actual checks pass without a helper signature or repeated PI GO.

Freeze per arm: fresh AdamW LR1e-5, betas(.9,.999), eps1e-8, weight_decay1e-5, clip accumulated norm1, warmup10/cosine2048; microbatch1/accumulation2. Original FP32 trainable action-projector/diffusion scope, BF16 frozen backbone and inherited inference precision remain. Preserve exactly the actual full-S4 free-generation objective and physical decode:
`J=L_native+.4L_position+.1L_increment+.1L_rotation+.1L_peak_speed_margin`.
Update once from `(J_c0+J_c1)/2`. No consistency/ranking/classification loss, adapter/LoRA/unfreeze, quantization, CFG scaling, prompt routing or output repair.

Real canary per arm: continuous8 versus4-save/fresh-process-restore4, checking new8-pass/template cursor, source visits, actual packets, weights, optimizer and scheduler under unchanged tolerances. Total32 completed canary updates discarded. Main arms restart from the fixed X weights. Preserve failed canary records; technical fixes cannot silently change scientific settings.

B then L sequential by default, unless an already frozen safe allocation dictates otherwise. Save complete checkpoints1024 and2048 only, plus the canary's necessary saves. No quality query at1024. Do not inspect either new endpoint's quality until BOTH endpoints are frozen; source/gradient/integrity logs remain available. One endpoint quality FAIL never cancels or alters the other. Partial-unit interrupted work is accounted, not promoted to optimizer completion. Exact in-arm recovery is allowed, with counters retained.

Reuse the mature training mechanism, adapting the hardcoded2000/4000 limits, plan constants, seed namespace, donor path and checkpoint times in an isolated PI0264 implementation. Do not change old scripts or rename an unmodified old constant into a false new ledger. Record all actual heads including backward recomputation and failed work.

## 6. Fixed evaluation: old regressions plus actual prospective transfer

Points: frozen donor X2000, B2048, L2048. No other donor or intermediate endpoint search. Seal the complete manifest, text and actual noise arrays before canaries. Evaluate after both endpoints are frozen. Donor new panels are not an early recipe-tuning signal.

### Panel A: preserve the full old union

Each point: PI0263's1152 unique logical rows (legacy680 plus crossed640 minus168 exact overlap). Preserve both named view memberships and original labels, B0/B1, raw inputs, S4, gates and metrics. Donor's1152 are already available and MUST be reused after identity checks. Missing archived bytes are restored, not conveniently regenerated. No new F/P/old Base/G queries.

### Panel B: newly supervised text fit, NOT generalization

On all30 original TRAIN M1 common-start families:2 commands*6 added templates(T2-T7)*2 old actual noise packets =720 rows per point. T0/T1 fit remains in Panel A. All 8 supervised formulations therefore have a transparent TRAIN fit view. These are now training support for L, not held-out success.

### Panel C: prospectively sealed text-and-noise challenge

On all10 existing KIN-VAL M1 common-start families:2 commands*(Q0-Q3 plus T0/T1 anchors)*2 NEW noise packets N0/N1 =240 rows per point.

Generate N0/N1 once per family using the mature standard-normal noise generator/shape/dtype with deterministic seed from SHA256(`PI0264:fresh-eval:26441:<family>:<N0|N1>`), truncated to the seed width required by that existing generator. Record the exact conversion and arrays. Within a family/noise, all commands/templates/points share the same actual initial packet. The generator/distribution itself is not changed. Record accidental byte coincidences honestly; do not resample based on model quality. These packets are separate from training randomness. No best-of-seed, averaging or choosing N0 versus N1.

Anchors on the same NEW packets help distinguish wording failure from a new-noise boundary crossing. Different panels with different text/noise should not be called a pure wording causal comparison; within Panel C each anchor/Q comparison fixes actual observations and noise. Ten geometries remain previously developed scenes, not new scene generalization.

### Counts and reuse

Per point:1152+720+240=2112 unique logical scoring rows. Three points6336. Planned donor reuse1152. Planned new quality queries<=5184 before further exact forward reuse. Hard cap5440 attempts and21760 inference heads, including256 strictly no-valid-result engineering attempts. Cache savings do not buy extra prompts/noises/model points. Distinct labels remain distinct scoring rows even when one forward is reused. Invalid/rejected cases stay visible with original denominators.

### Metrics and qualifications

Continue strict original own-versus-opposite H40 XYZ margin>0. Ties not correct. Numeric thresholds medianXYZ<=.020m,p95<=.035m,medianSO3<=.050rad andzeroREJECT unchanged. No DTW, timestamp alignment, nearest-label replacement, label averaging, alias-floor subtraction, output smoothing or best-of-noise.

Report THREE qualifications separately:
1. `legacy_full_quality_pass`: original4 numeric groups and8 language groups each>=10/12.
2. `crossed_M1_robustness_pass`: old four templates xB0/B1 on10 KIN-VAL families, each>=17/20 plus numeric gates.
3. `prospective_text_noise_pass`: Panel C's six formulations xN0/N1, all12 groups individually>=17/20 plus numeric gates.

The third is a NEW prospective criterion, not a retroactive correction of PI0263. Results are not independent-binomial tests; families are clustering units. Also report both-command-correct families, every color/command/template/noise group, signed margins, wording flips, original g/b decomposition, TRAIN versus KIN-VAL, and M3/O64 regressions. Average performance cannot mask a failed direction or poor wording group.

A final endpoint can be listed as a synthetic research candidate only if all three qualifications pass. If both pass, list both qualified research endpoints; do not select one by a post-hoc favorable seed or weakest-error slice. `selected_checkpoint` and `deployment_selected_checkpoint` remainNONE; the next PI decision chooses any transfer candidate. Execution completion does not require quality PASS and does not permit extra updates.

## 7. Resources, engineering autonomy and storage coordination

New PI0264 allocation (keep old ledgers intact):
- Main updates4096 (B2048+L2048); supervised microbatches8192.
- Discarded canary updates32; completed-update technical recovery reserve256.
- Total completed updates hard4384; raw training heads hard110000 (including recomputation/failures/parity).
- Logical scoring rows6336; planned new queries<=5184; attempts/heads hard5440/21760.
- New model/image/action data generation, IK, rendering, physical/WBC/hardware/TEST/DEV-F0. New TRAIN strings6, prospective strings4; no on-the-fly expansion.

Do not use a mechanical qsub/IO retry count to stop correct work. Path/import/CRLF/serialization/staging/resource/exact-recovery repairs are autonomous within frozen meaning and total actual compute. Never rerun a valid outcome to improve it; reserve cannot extend a2048 endpoint or add a third training arm. If remaining budget cannot support necessary recovery, preserve full state and report the exact deficit.

PC orchestrates; Corelab CPU/no-gradient evaluation; Hopper real model training/canaries on lawful memory-compatible allocations. Do not reserve H200 for ordinary CPU debugging or change precision to fit a smaller GPU. Query current jobs/processes; unavailable tooling isUNKNOWN, not no jobs. No duplicate healthy job, interruption of unrelated work or new monitoring daemon/email/PR merge.

The separately authorized storage assistant may continue its inventory/retirement work, but must now protect X2000's designated donor and all active PI0264 inputs/outputs/resume/transfers. Do not require a cleanup completion/signature for training. Notify through the existing ledger/protected-asset record, not a new coordination system. Preserve all current source dependencies; no broad new cleanup authorization here. Reuse fixed-HF receipts for unchanged assets. Archive complete new Trainers to the already authorized project HF PUBLIC area after content/permission checks; no private robot/WBC/credentials. Maintain next-write overlap and10GiB minimum reserve, explicitly count temporary staging/readback copies and report quotaUNKNOWN when unknown. A300GiB management target is not a scientific admission gate.

## 8. Deliverables, terminal and the next scientific decision

Deliver: sealed source/text/schedule/challenge card; concrete input and leakage checks; two real restore receipts; four complete1024/2048 checkpoints with fixed-revision full-file readback;6336-row union and panel views; independent-computation-path CPU replay (same-owner clearly labeled); per-template/family/noise/regression report; compact fixed-array video including failures and new challenge coverage. No new query for video, no generated or relabeled physical success. Reuse PI0257-v2 tools; video cosmetics must not block substantive closure.

Terminal: `PI0264_LANGUAGE_DIVERSITY_TRANSFER_PILOT_COMPLETE — STOP_FOR_NEW_SCENE_OR_PHYSICAL_TRANSFER_ADMISSION`.
Keep PI0263 COMPLETE/FAIL and all old selection states. Both valid training endpoints and all panels are required forCOMPLETE; otherwise namePARTIAL/BLOCKED and actual missing work.

Decision interpretation, NOT extra authority:
- L beats B on Q with fresh noise and old qualifications retained: supports this balanced multi-formulation training for text/noise transfer. Do not call reused scenes independent geometry or full-task generalization. Next prioritize genuinely new scene-family coverage and separately authorized WBC compatibility, not another old-template score chase.
- Only the newly supervised T2-T7 improve, but Q/V remain poor: the training combinations are learnable but paraphrase transfer is insufficient; stop adding arbitrary old-scene template schedules. Propose one explicit representation/visual-grounding or wider-data experiment next.
- Both improve similarly: added optimization/common schedule may suffice; no unique diversity credit.
- T0/T1 and Q all fail on the same new-noise/family conditions: pure wording is insufficient; examine source-bound scene grounding and stochastic action ambiguity without deleting hard families.
- Geometry/M3 regress: report a tradeoff, not a semantic win. O64 remains its own processor-domain regression, not an unqualified weight-forgetting claim.

The goal is a real paired learning experiment with a forward-looking evaluation, not another audit-only task. Necessary finite checks followed by canaries/training/evaluation are already authorized; helpers, known alias, oldWBC settling or a request for a unique neural cause must not create extra waits.

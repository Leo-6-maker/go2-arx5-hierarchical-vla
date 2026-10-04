# PI0272 — Matched direct trajectory readout on consumed recovery support

Authority: **PI0272_MATCHED_DIRECT_TRAJECTORY_READOUT_FITTING_GO_V1**.
Date: 2026-10-04 SGT.
Review base: `90489ca895d1ea1ad2371efbb174d72058bf1250`, branch `pi0267/executor-decoupled-recovery`.
Predecessor: PI0271 release `cbdcaf85ab6c25999fe054bf5da5285fadc394a3`, terminal Issue #11 comment `5978753654`, evidence commit `91330241c13b083b4077a012e00394f9d1df624d`.
The immutable release commit and protocol blob are supplied by this goal's source-bound Issue #11 canonical.

## 0. Decision and scientific scope

GO for ONE bounded, two-arm TRAIN-support fitting experiment: **S4FIT1024 versus DIRECT1024**. Both start from the same complete PI0270 AGG4096 weights, use the same 296 previously consumed conditions and 1024 fresh-optimizer updates. This is not another full data campaign, not a new DAgger round, and not permission to restart PI0270/71.

The intervention is the generated-trajectory readout path. S4FIT retains the qualified stochastic four-step generator. DIRECT uses the SAME action-head modules and parameters as a deterministic one-pass normalized-trajectory readout from a fixed zero query at diffusion bucket zero. Crucially, **both retain the unchanged native flow loss as a regularizer and the same physical losses, scales and temporal weights**. DIRECT is therefore a native-regularized direct readout, not an assertion that a newly initialized MLP, pure regression objective, or all deterministic policies outperform flow matching. No new head parameters, MSE/L1 auxiliary term or loss-weight search is authorized.

The questions are whether this fixed readout can learn the residual corrective-reference map more faithfully on already-consumed support, and whether any advantage exceeds additional fitting with the existing S4 path. An old frozen AGG result alone is NOT an adequate control for 1024 additional updates. The two new arms are necessary for that comparison; they do not repeat the earlier full-corpus training experiment.

This study cannot establish a repair of the separate M1 development/visited-state gap, reliable task completion, independent generalization, WBC performance or embodiment transfer. No new EVAL label enters training. The fitted diagnostic endpoints are not selected/deployment candidates. Those formal fields remain NONE.

One Codex main owner proceeds through implementation, finite correctness checks, real canaries, both training arms, fixed evaluation and preservation without helper-signature dependencies. Refresh actual runtime/locks before acting; do not cancel a healthy job or create a competing automatic goal. A closed predecessor needs identity reuse, not repeated inference or archival downloads. Platform/session limits do not reset scientific counters.

New source and outputs belong only to `scripts/pi0272_execution/`, `reports/PI0272/`, `docs/protocols/PI0272_*` and isolated PI0272 runtime roots. Continue the existing branch, preserve concurrent commits, no force-push/merge/new permanent branch/generic issue/PR77 pointer. Never synchronize moving Git files into a sealed live runtime.

## 1. Reviewed evidence and interpretation

The reviewer read PI0271's complete research report, terminal canonical, final closeout, artifact audit and required-source index at the review base, and PI0270's terminal canonical. PI0271 closeout is dated `2026-10-04T09:58:52.808347+00:00`; its exact canonical readback is dated `09:57:50.2202728Z`. Execution is COMPLETE, with no missing scientific requirements: 2400 new queries, 9600 S4 heads, 64 teacher calls, zero technical neural retries, and no new training/episodes/renders/physics. Its three result parts and publication receipts are closed. This is a repository-receipt review, not fresh SSH/PBS telemetry or a reviewer replay of every prediction.

PI0270 AGG's completed main result is M1 E0/E1 16/16 original each, reliable 1/16 and 3/16; M3 E0/E1 1/16 original and reliable each. Thus reliable6/64, full64 safe, still VALID_QUALITY_FAIL under the unchanged subgroup gates. Original ROT-relative LEGACY PASSES: AGG XYZ13.326mm/SO3.024064rad versus ROT15.121mm/.046878rad, zero rejects. Do not retain the earlier speculation that old-FIT SO3 regression established legacy failure. Only32/224 old-FIT logical rows were actually consumed by AGG.

PI0271 A1 Q3 prefix XYZ improves PREFIX-to-AGG M1 62.112 to7.401mm, M3 7.291 to1.680mm. M3 terminal consumed support retains35/48 opposite-direction two-noise outputs, versus46/48 for PREFIX; only24 input conditions/two TRAIN families underlie that number. M1 terminal support improves too (prefix9.874 to2.007mm). Do not call all late support unlearned.

The K8 Q3 empirical mean-error term contributes median90.510%/91.996% of AGG M1/M3 prefix MSE. Dispersion increases while total error decreases. This finite-bank observation is not a population bias estimate, not a proof of a representation fault, and does NOT transfer automatically to component or A3 states where K8 was not measured.

On A3's two development families, all31 usable M1 same-state comparisons worsen relative to PREFIX. This includes all8 observations at t=0, before action-induced divergence. Therefore the difference cannot be attributed solely to later learner visitation: geometry/conditioning, generalization and residual local fitting remain confounded. One of64 teacher labels is unusable under the original gate; all64 remain DIAGNOSIS_ONLY. Their labels are prohibited from this study's data and validation selection. Existing AGG first successes and late reversals retain their original scorer semantics.

Actual code review followed the R3 physical binding, R3 full-loss construction and the full `pi0247_u_rollout_overlay` forward/rollout/readout/parity paths. The inherited readout consists of state_encoder, action_encoder plus timestep/position embeddings, the conditioned transformer, and action_decoder. Runtime module paths and hashes, not same-name historical defaults, remain authoritative. R3 overrides the historical clamping defaults with the current unclamped ROT contract.

## 2. A0 — accept the closed predecessor and bind exact sources

Verify the already closed PI0271 `FINAL_CLOSEOUT_RECEIPT.json`, canonical body/hash and published terminal5978753654; verify PI0270's closed identities. Reuse valid receipts rather than repeating predecessor queries, canaries or all-member HF downloads. Record current actual lock/process/job state before the one serialized new run. A concrete newly discovered source/semantic contradiction blocks only its affected operation and is reported explicitly; an unknown unique neural root cause is not a blocker.

Common initializer: HF `leoyuer/pi0246-gate-pass-campaign`, revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`, prefix `arms/PI0270/AGG/checkpoint-4096`, complete tree `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`. Initialize from WEIGHTS with a fresh optimizer in each arm; do not resume the old optimizer/scheduler or change donor after results.

Bind effective ROT statistics `0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9`, original recovery mapping `53fe049e406f079e079d03af71e25c0ffd339b587fab7d8e5dc80bebe35fd633`, PI0271 LATE_SUPPORT_CARD `2899f730ef8f3fd1a76350662ec284237439ab7104192d9d6dfe2c30c503528e`, and original NEW_FIT192 `553d254cea5ad5c32d0d8acf615f2abfe222459ac10e9d0bb0e09b786884d226`.

Source archives already available: PI0270 training-data revision `d16394d8e32d535465828ef5d572b491a373e759`; PI0271 support/noise revision `5b5dc9bc65dec326c9f4d758a8920a4268d00a29`; summary revision `6004b2bce43961db8e0ca56ca4b9001b9ec44d8e`. Reuse exact local copies and archived member identities. Only restore a genuinely missing dependency from its fixed source. No historical model sweep or copying full optimizers to Corelab.

## 3. A1 — freeze the small shared fitting set and exposure schedule

Use exactly the296 designated PI0271 A1 conditions, not a newly mined set of worst failures. Each is bound to its original processed condition, language, raw teacher H40, masks and actual AGG consumption. Rows: M1 Q2/Q3 48/48, M1 terminal32, M3 Q2/Q3 48/48, M3 entry/midarc/terminal24/24/24. Total296 (M1 128, M3 168),24 TRAIN families overall; component rows remain from only two TRAIN families.

No relabelling, interpolation, state perturbation, new image, new scene, data augmentation or new teacher call. Preserve all296 logical source memberships, even if exact conditions coincide; report exact identities. No model-score rejection or replacement. Frozen teacher eligibility from the source remains; newly discovered byte/label correctness conflicts are explicit stops, not silently fixed labels.

Sort rows by numeric original `(unit,micro)`. For zero-based new microbatch ordinal n in0..2047, choose row `n mod296`. Each new update uses two consecutive microbatches with the existing accumulation2/mean convention. Thus272 rows occur7 times and24 occur6 times per arm. This deterministic small-set replay is COMMON to both arms; it is not the original4096-unit full-corpus plan or a claim of task-balanced sampling. Publish actual task/command/bucket exposures. Never rank rows by loss or iterate until one arm passes.

Use training seed27202, with unit seed = first8 SHA256 bytes, big-endian, of `PI0272:TRAIN:27202:{update_1based}`, modulo2**31. Preserve the qualified per-unit/micro RNG lifecycle, draw order and zero-worker DataLoader restore. Bind actual native epsilon/tau tensors across arms, not seed alone. S4's generated-path initial noise is the same native epsilon converted to the native BF16 dtype. DIRECT intentionally does not consume it for generated output; still record the identical native training packet.

CPU admission on the actual Hopper processor verifies the FULL2048-micro stream for each arm and exact pairwise input/label hashes, current affine ranges, padding/masks/dtypes, source mappings and all296 planned exposures. No learned forward or optimizer is permitted in this CPU stage. These are plan identities, not neural consumption until actual training audit. Retain the verified Prescott CPU encoder binding and all active scientific precision boundaries; do not silently refit or clip larger finite values.

Original NEW_FIT192 Q0/Q1 is an unchanged spillover guard, not newly added fitting data. Check actual identity overlap against the296 conditions and report any overlap; no substitution to force an independent holdout. All of these are already-developed TRAIN-support panels, not blind TEST. No A3/development-EVAL labels, factor rows or legacy rows are added to fitting.

## 4. A2 — exact generated-path intervention

### Shared architecture and objective

Both arms load identical AGG4096 tensors and keep the actual approved trainable-name set (currently471 tensors) and frozen set (559 tensors) unchanged. Bind names/shapes/counts to the initializer evidence; do not merely assert matching totals. No new parameters, reset readout, backbone unfreezing, pooling replacement, new positional encoding or state/history field. Model conditioning remains body_rgb, wrist_rgb, actual StateV1 and language, with the current image processing, embodiment mapping, relative SE(3), row-rot6d and gripper contract.

Both compute the unchanged native flow loss from the same actual epsilon/tau sampling and native mode lifecycle. Both optimize:

`J = L_native + .4*L_position + .1*L_increment + .1*L_rotation + .1*L_peak_speed`.

All physical functions, valid masks, .05m/.05rad/1m/s scales, native full-H40 target, coefficients, and full39-transition peak-speed term stay identical. Preserve PREFIX pose weights .8/9 at0..8 and .2/31 at9..39; transition weights .8/8 at0..7 and .2/31 at8..38. No new normalized-regression loss, stop loss, contraction loss, terminal bonus or tuning is introduced. The physical losses receive each arm's own generated trajectory defined below.

### S4FIT1024 — newly trained matched control

Use the exact qualified label-free S4 path: start from actual BF16 noise, four transformer/readout evaluations at native diffusion buckets corresponding to0,.25,.5,.75, Euler increments1/4, unchanged decoder and differentiable checkpointed graph. Do not alter native num_inference_timesteps=4, clamp behavior or inherited loss/mode semantics.

### DIRECT1024 — native-regularized one-pass direct readout

Use the SAME existing `state_encoder`, `action_encoder`, native position embeddings, conditioned transformer and `action_decoder`. Its generated-path call is precisely:

1. Obtain condition-only backbone features/masks through the exact native preprocessing used by the qualified S4 path. Encode normalized actual state identically.
2. Form an all-zero query tensor z of shape[B,40,132], in the SAME BF16 feature dtype/device as S4's initial query. The query is a latent input, not a physical zero pose or a substitute for actual StateV1.
3. Set the native diffusion timestep bucket to integer0. Apply action_encoder(z,t0,embodiment_id), the existing horizon position embeddings, concatenate the same state features, run the same transformer with identical conditioning/masks, and apply the same action_decoder; take the last40 outputs.
4. Interpret this ONE raw [B,40,132] readout as the normalized relative H40 trajectory. Do NOT Euler-integrate it, add noise/z residuals, average outputs, or add a new physical anchor/target correction. Decode with the unchanged processor and apply exactly the same physical losses/full-H40 gate.

The native action-head configuration remains S4 for the SHARED native training regularizer. Do not globally set inference timesteps to1, alter timestep bucket conventions or accidentally call native get_action for DIRECT evaluation. DIRECT's execution mode must be an explicit versioned sidecar; loading its weights through stock S4 is a different, unauthorized effective policy.

Use the generated-path eval-mode context with gradients enabled and the same restoration checks as S4; one non-reentrant checkpointed direct readout permits bounded memory without changing precision. No teacher, action target, path progress, component start type or source ID reaches this generated-path function. Direct quality inference is condition-only and never invokes native forward with labels. At evaluation dropout is off and generation consumes no RNG. Native training is still stochastic; do not call the entire training formulation deterministic or pure direct regression.

This joint change of query initialization and one-pass output interpretation is the registered readout formulation. It does not isolate stochasticity from integration depth, and is not a proof about all direct heads. Equal initial WEIGHTS do not mean equal step0 effective policies: DIRECT0 is separately evaluated below. Preserving the native regularizer avoids simultaneously replacing the supervision objective; if this formulation fails, it does not rule out a different direct-regression objective.

## 5. A3 — finite correctness, matched training and preservation

Create a narrow isolated wrapper/hook around the ACTUALLY bound paired/R3/temporal implementation. Reuse its native training and physical-loss construction. Do not run the inherited S4 physical rollout secretly inside DIRECT in addition to the direct trajectory, and do not reuse S4-specific gradient/forward-count assertions as if DIRECT executed four nodes.

Before scientific runs, source/CPU fixtures must verify the input boundary, constants/shapes/masks, unchanged physical formulas, exact plan and seed conventions, generated-only function independence from labels, correct noncommuting rotation/frame round-trip, and that historical clamping helpers are not active. Constructed fixtures are not real model results.

Real canary PER ARM: continuous8 versus4/full-save/fresh-process-resume4,16 updates/arm,32 total. The schedule,1024-step scheduler and inherited RNG restore apply from the start; discard canary weights after durable comparison evidence. Compare complete model/optimizer/scheduler/cursor/modes and Python/NumPy/CPU/CUDA RNG. Record actual source packets. The first already-required backward proves S4 gradients through all4 nodes or DIRECT gradients through its single generated readout; no extra diagnostic neural backward.

During these existing canary/main first passes, perform bounded train-versus-label-free-inference parity on the SAME packets: stock S4 parity for the control, the explicitly defined direct function for DIRECT. Retain the inherited numerical/gate parity tolerances (raw1e-4, XYZ1e-5m, unchanged complete-gate issues) rather than loosening them after failure. At most256 extra head evaluations total are reserved for these parity/entry checks, charged in training-head counters; they cannot be used for learning or quality selection. Preserve first-pass output references for audit. A finite bad prediction/gate rejection is not a canary correctness failure if both matching paths agree; a genuine semantic/gradient/nonfinite-optimization contradiction stops the affected path with its evidence.

After real correctness PASS, train S4FIT then DIRECT serially, each1024 updates from AGG4096 WEIGHTS with a fresh AdamW: LR1e-5, warmup10, cosine1024, weight_decay1e-5, max_grad_norm1, microbatch1/accumulation2, same trainable scope and native BF16/physical FP32. No extra data, second seed, learning-rate search, stage extension or third arm. Both arms finish before new endpoint quality is examined. Valid quality failure of one arm does not cancel the other arm.

Save complete512/1024 Trainers per arm, four formal checkpoints total; only1024 is quality evaluated. Retain full optimizer/scheduler/RNG, processor/config/mapping/statistics, exact source/environment/kernel identities and PI0272 readout-mode sidecar in every complete checkpoint. File count may exceed the old18 because the new mode contract is required. A raw safetensors set without the readout contract is not a complete effective policy. Full fixed-revision file inventories/readbacks are required only for these new artifacts; keep actual recoveries until durable readback. Do not copy optimizer payloads to Corelab.

Audit both actual2048-micro streams, every paired condition/label/native-random packet, all296 actual-consumption identities, scope/updates, losses/modes and complete checkpoints. S4 records4 generated nodes, DIRECT1. Under the inherited head-count convention, normal two-micro updates cost approximately18 and6 head evaluations respectively, including generated-path backward recomputation. Count actual calls; equal updates/exposures are NOT equal FLOPs or wall time. Report both resources. Missing logs or unmatched packets cannot be labelled a direct-head effect.

## 6. A4 — fixed endpoint comparison, no new episodes

Reuse AGG's closed PI0271 A1/K8 outputs and closed PI0270 NEW_FIT outputs with exact policy/input/noise/decode bindings. Do not rerun them or load PREFIX for new calls. Freeze all evaluation cards and the direct mode BEFORE training; run the following only once both finals are frozen, so DIRECT0 cannot trigger a formulation change.

### Fitting support and initialization

- S4FIT1024: all296 inputs with original PI0271 j0/j1,592 new queries; on the96 Q3 inputs also reuse the existing frozen j2..7 tensors for576 additional queries. Total1168 S4 queries. This matches AGG's existing two-noise A1/eight-noise Q3 panels.
- DIRECT0: the original AGG weights with the NEW direct readout, each296 input once,296 queries. It is an explicit readout-change baseline, not the old AGG effective policy or an additional training arm.
- DIRECT1024: each296 input once,296 queries. Do not call it eight times and count identical copies as independent samples. For logical comparisons, join its one deterministic output to matching conditions, with explicit reuse multiplicity.
- Repeatability: rerun only the FIRST Q3 input in each task/command stratum (four fixed inputs) once on DIRECT1024,4 extra calls; require exact output bytes and unchanged RNG versus the saved corresponding direct output. Do not repeat other direct draws. This checks execution determinism, not accuracy or stability.

### Unchanged early-support spillover guard

Use original NEW_FIT192's exact images/state/language/labels/noise without training on its rows except reported exact identity overlaps. Query S4FIT1024 once per input with the frozen original noise and DIRECT1024 once,384 new queries total. Reuse original AGG192 outputs. This tests spillover on already-developed Q0/Q1 TRAIN support, not new geometry generalization or LEGACY qualification.

Total maximum new quality calls2148, including DIRECT0 and repeatability; planned quality heads6228. DIRECT calls each count1 head, S4 calls4. All finite REJECT/WARN, nonfinite values and invalid rotations stay in outcome inventories with their fixed denominators. Do not substitute a better seed or silently discard a failed bank. Old finite AGG j6 REJECT stays in its existing Q3 bank.

For every condition report first-point bias, prefix0..8/fullH40/suffix errors, .4s endpoint error, SO3, continuous teacher/model displacement and signed alignment, gripper and complete original gate issues. Separate near-zero displacement and invalid/undefined direction; retain the existing reporting guard, never count undefined direction as success. A negative cosine is a local diagnostic, not the original M3 path verdict.

## 7. Predeclared analysis and permissible conclusions

Primary residual target: the24 consumed M3_TERMINAL conditions. Secondary target:96 M1 Q2/Q3 conditions. Always report all296 strata and the192 guard; do not hide M1 terminal/M3 entry/midarc tradeoffs.

For a condition i and S4 noises j0/j1, define prefix physical position MSE e(i,j) using the existing sum-over-XYZ/mean-over9-poses convention. Compare DIRECT1024's one e(i) with mean_j e_S4FIT(i,j), and with mean_j e_AGG(i,j); take medians of these PER-CONDITION differences within each fixed stratum. Also report corresponding RMSE and SO3 separately. These are not differences of marginal medians or equal-noise comparisons between a stochastic and deterministic generator. For Q3 use the complete8-noise banks as an additional analysis, retaining finite mean-error/dispersion decomposition and its limitations.

For M3 terminal, compare per-condition direct wrong-direction indicator with S4's fraction over the two fixed draws, and report both command strata and both families. Do not compare a /24 condition count unlabelled against a /48 noisy-output count. Small corrective displacements require the continuous vectors and magnitudes, not signs alone. Component results have two scene families; no artificial24-family or noise-as-independent-scene significance.

Interpret the four identities separately: AGG/S4 step0 (reused), DIRECT0 (new readout at the same weights), S4FIT1024, DIRECT1024. Improvement versus DIRECT0 alone is NOT evidence that direct beats S4. Improvement of both fits versus AGG may be extra exposure. A direct advantage over the matched trained control on the primary residual group, with lower wrong-direction rate and nonworse same-group SO3/gate and guard outcomes, supports a *local direct-readout feasibility advantage* only. Report the numeric conditions individually; mixed/negative results remain valid outcomes. No new task-PASS threshold or candidate selector is created.

A favourable result justifies a subsequent full-support and feedback experiment, NOT deployment of these small-set fitted endpoints. A poor result does not prove the VLA cannot represent the task, or that no direct architecture/objective can work. If both small-set fits retain substantial errors on actually consumed states, the next decision concerns fitting/conditioning feasibility rather than automatically collecting more scenes. If extra exposure alone accounts for improvements, preserve that counterevidence to a head-specific story. The distinct two-family M1 A3 gap is NOT claimed solved here.

## 8. Hard budgets and execution boundaries

PI0270/71 keep their old ledgers and closed results. New PI0272 budget:

| Consumption | Planned | Hard ceiling |
|---|---:|---:|
| Official updates |1024 per arm /2048 total|1024 per arm|
| Real canary updates |16 per arm /32 total|32 planned|
| All completed updates including technical recovery |2080|2208 (128 technical reserve)|
| Actual training head evaluations, including recomputation/parity |about24960 + bounded parity|32768|
| New quality calls |2148|2560 attempts|
| New quality head evaluations |6228|7168|
| New teacher labels, collection episodes, rendered frame pairs |0|0|
| New WBC/dynamic physics/hardware starts |0|0|

The256 parity-head allowance is WITHIN32768, not extra budget. Technical reserve cannot extend a healthy1024 endpoint, buy scientific conditions, a new seed, loss or third arm. Journal all attempts before execution, retain partial heads and actual technical failures. Resume only missing work from the exact state; never retry valid poor outputs for quality. No additional training-performance probes beyond recorded training passes and the fixed final panels.

Hopper: one compatible serialized training allocation; Corelab: no-gradient model-only inference after both finals; CPU: bounded-thread metadata/replay. Check current actual user quota and save/staging overlap plus10GiB; protect AGG and new recoveries, no arbitrary300GiB requirement, broad cleanup or repeated historical downloads. Optional helpers never block owner progress. Same-semantic import/path/serialization/cache/kernel/lifecycle/counter fixes are autonomous with narrow receipts. Changing the registered readout, loss, input/data/teacher/normalizer/precision or scorer is not an engineering fix.

This goal has zero new motion episodes, so no new motion video is required. Reuse predecessor videos for historical context only. New static plots may use saved outputs without new renders or model queries; an optional plot is not a launch or completion bottleneck. No mail/monitor/new paid API/merge.

## 9. Required terminal deliverable

Deliver a source-bound research return with predecessor closure/reuse,296-source card and actual exposure, mode-labelled step0/final identities, exact objective/path diff, full CPU admission, both real canaries, both1024 actual training audits, four complete Trainer fixed-revision readbacks, all fixed2148-query results and reuse counts, per-condition/stratum/direction/rotation/gate/guard analyses, correct head/cost accounting, CPU numerical replay and new result archive/readbacks. Direct replay validates its saved raw readout and unchanged decode, not fictitious S4 Euler steps; it does not recompute the learned network. Disclose the shared metrics/gate functions and independent arithmetic actually used.

Publish a single Issue #11 terminal canonical and exact readback. Do not stop at CODE_READY, CPU_PASS, CANARY_PASS, TRAINING_DONE or UPLOADED while the next authorized dependency is available. A real correctness/resource block records precisely what cannot continue and preserves all completed evidence; no indefinite responsibility loop. Effective scientific failure can complete this bounded trial.

Terminal: **PI0272_MATCHED_DIRECT_READOUT_FITTING_COMPLETE — STOP_FOR_FULL_SUPPORT_FEEDBACK_OR_FITTING_DECISION**.

No new original-task or reliable-task qualification is inferred from these fitting panels. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`. PI0270 original/reliable VALID_QUALITY_FAIL and its positive6/64/legacy findings, PI0269/68 FAIL, and PI0267 WBC PARTIAL (8 valid experts,1 engineering-incomplete model attempt/3 queries,23 deferred conditions) are all retained unchanged.

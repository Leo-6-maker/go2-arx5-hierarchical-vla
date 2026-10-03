# PI0269 — Executed-prefix supervision for reliable task-space policies

Authority: **PI0269_EXECUTED_PREFIX_RELIABILITY_LEARNING_GO_V1**
Date: 2026-10-03.
Basis: final PI0268-R1 evidence `7dc82e4f82513932f68875f927c208ee51418ded`, final-canonical binding `2684dbcd3243f23ff698760e327d5a19525d29c2`, Issue #11 comment 5964778423.

This is a bounded launch authorization for ONE matched temporal-loss comparison, including actual fine-tuning and fixed feedback evaluation. It is not a claim that this loss will succeed or that the previous normalizer alone caused the failures.

## 0. Owner, precedence, and research scope

The agent accepting this goal is the single main owner. PI0268-R1 is COMPLETE / VALID_QUALITY_FAIL; do not reopen it, extend its training, rescore it with a new rule, or reactivate a competing automatic goal. Adopt any genuinely healthy preservation process and preserve its exact inputs. Helpers may do isolated read-only QA or storage work, never mandatory sign-off. A concrete unavailable resource can delay that operation, not create a reciprocal owner-approval loop.

Continue `pi0267/executor-decoupled-recovery`; new files go only to `docs/protocols/PI0269_*`, `scripts/pi0269_execution/`, `reports/PI0269/`, and isolated PI0269 runtime/output roots. Publish a compact source-bound Issue #11 return; no generic new issue, PR77 pointer, additional long-lived branch, force-push or merge. Serialize writes and preserve concurrent commits. Do not sync a moving branch into a sealed healthy runtime.

The scientific mainline remains a shared vision/language/EEF policy learned without target-WBC task demonstrations. This experiment tests high-level short-horizon control learning, not low-level WBC training. The previously deferred PI0267 secondary-WBC lane remains explicitly PARTIAL and is not a prerequisite; this release grants it no new execution. Formal selected/deployment remain NONE.

## 1. Accepted findings and unresolved explanations

Read FINAL_RESEARCH_RETURN.md and FINAL_CLOSEOUT_RECEIPT.json at the final evidence commit; relevant final CSVs and CPU receipts; the PI0268 preparation codec/exposure/closed-loop reports; and the actually imported training/decoder fileset, not a same-name historical helper.

Established:
- Both OLD/ROT arms completed 2048 updates, four full Trainer archives, 3872 offline rows and 256 logical virtual episodes. All three preregistered verdicts failed.
- ROT final has better seen-input fit than OLD final: XYZ medians 9.940 vs17.570mm, SO3 .036871 vs.048520rad. Legacy440 regresses in both quantities. This is not a global normalization win.
- Original M1 first hits: OLD0 13/32, OLD2048 30/32, ROT2048 22/32. Each successful episode subsequently exits the pose bounds. Every policy has 0/64 reliable episodes. Both final models have no virtual gate refusal or dangerous collision, but M3 is0/32.
- M3 OLD2048: eight broad-ready, six entered, all six one-bin/reverse, three also shortcut. ROT2048: three broad-ready, two entered, one/two bins, both reverse/shortcut. These are overlapping scorer labels, not independent causal mechanisms.
- Initial first-point position biases are sub-mm; first .4s displacement is several mm. Neither quantity alone explains direction, completion or stability.
- The implemented S4 physical position and rotation losses average all40 poses; the increment loss averages all39 transitions. The actual executor repeatedly uses only0..0.4s, indices0..8. Native flow loss and physical loss are distinct; gradient rollout through four sampling steps is not backpropagation through the environment.

Inference to test, not an established cause: allocating more of the physical supervision to the actually executed prefix may improve local progression and holding under repeated feedback. The current teacher already defines progression and terminal correction. This experiment changes emphasis, not teacher semantics. Equal perfect-fit minima remain; weighting alone cannot guarantee stability, recover missing observations, teach unseen states, or solve cross-embodiment transfer.

Alternatives retained: insufficient decision-state coverage, remaining TRAIN approximation error, weak visual/task conditioning, stochastic policy bias, unmodelled history, and more appropriate action-head/optimization choices. Do not equate a low whole-window median with accurate policy dynamics or diagnose all failures as WBC problems. Original first-hit, V3/V2, reliable-window and safety scores remain unmodified.

## 2. P0 — finite existing-evidence and supervision check

Do NOT regenerate the24-family corpus, redo64 teacher qualifications, repeat whole HF archives, re-fit statistics or re-run PI0268 predictions.

Verify the exact closed inventories and reuse their identities. Examine saved per-horizon residuals/short-prefix errors by task, command, nominal/recovery and terminal status; means and marginal medians do not replace paired rows. Required quantities at a query are actual current pose, predicted pose at0/.4s, and corresponding teacher pose at the SAME current state when an unambiguous label exists. Compare frames and timestamps correctly; a nearest TRAIN label is not automatically the correct label at an encountered state.

For a bounded additional numeric diagnosis, use the first two completed development-EVAL families in frozen metadata order, OLD2048 and ROT2048, both tasks/commands and E0/E1, at scheduled query times0,10,35s: maximum96 saved-state label calls. Use only existing RGB/state/prediction arrays, no new policy queries or renders. If a time is absent after a valid stop, mark MISSING_AFTER_STOP, never substitute a favorable time. Apply the unchanged ArclengthCorrection to that measured pose; preserve invalid/ambiguous-label flags. Compute signed alignment with the teacher's executed displacement, prefix discrepancy and near-terminal contraction only where defined. Zero teacher displacement is a hold case, not an undefined directional cosine turned into success. These labels are diagnosis only: no EVAL state, label or sampled target enters training.

Use the new constructed CPU fixture only as a mathematical sanity test. Its synthetic path has4.517mm whole-H40 XYZ RMSE while its .4s endpoint goes8mm backwards instead of8mm forward. It is not a replay of project outputs and has NOT passed the project's complete gate.

If supervision, time/frame or processed-label correctness is newly contradicted, preserve the exact contradiction and pause the affected training path. If no new correctness defect exists, a lack of unique neural-root-cause proof is NOT a stop condition: continue the fixed two-arm experiment below. Do not make completion of every optional plot a training gate.

## 3. Common donor and numerical contract

Both arms start from the complete **PI0268 ROT2048** weights:
- repo `leoyuer/pi0246-gate-pass-campaign`
- revision `97d8bf027f07241e2206b05e07597e4719c891e4`
- prefix `arms/PI0268/ROT/checkpoint-2048`
- bind the18-file inventory in `reports/PI0268/R1/ROT2048_FULL_READBACK.json`.
- effective statistics SHA256 `0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9` as serialized in that checkpoint; verify actual processor parsed values and inputs as well.

ROT is a numerically bounded research platform, NOT a qualified candidate or the best historical M1 checkpoint. This choice does not overturn its retention regression. Do not switch to OLD after inspecting new results. No old-R inference-only decoder replacement is allowed.

Retain BOTH state and relative-action rotation maps, raw/relative pose definitions, no-clamp behavior, XYZ/gripper maps, native BF16 and existing full-S4 FP32 boundaries, H40=.05s grid, actual input fields, padding/masks, teacher, cameras and virtual tool. Same effective policy is shared at step0, so one donor baseline can be reused with exact identity. Do not merely compare weight-file names and ignore processor identity.

## 4. The sole intervention: temporal weights in physical supervision

Arms: **UNIFORM4096** and **PREFIX4096**.

For each microbatch, obtain the SAME complete differentiable S4 sample already used by the production auxiliary losses. No second sampling trajectory or extra neural pass is needed. Use the existing physical decoding and scales:

- point error a_k = ||p_hat_k - p_star_k||^2 / (.05m)^2;
- rotation error b_k = ||R_hat_k - R_star_k||_F^2 / (2*(.05rad)^2), exactly the existing chordal convention;
- transition error d_k = ||(p_hat_(k+1)-p_hat_k)-(p_star_(k+1)-p_star_k)||^2 / ((.05s)*(1m/s))^2, k=0..38.

UNIFORM retains the original production reductions: pose weights1/40, transition weights1/39.

PREFIX uses fixed, positive weights:

    w[k] = 0.8/9   for k=0..8
           0.2/31  for k=9..39
    v[k] = 0.8/8   for k=0..7  (transitions ending by0.4s)
           0.2/31  for k=8..38

Replace ONLY the reductions of these three physical components with their weighted masked means. Preserve original masks, valid-coordinate definitions and sum-over-XYZ / matrix chordal errors. Normalize by the sum of valid weights, never reuse the action mask as a time-weight carrier. Production microbatch size remains1; pair averaging remains1/2. All weights remain positive, including the unexecuted suffix.

Both arms retain:

    J = L_native + .4*L_position + .1*L_increment
                 + .1*L_rotation + .1*L_peak_speed.

The native generation loss, coefficients, .05m / .05rad /1m/s scales, and worst-speed-margin over ALL39 transitions are unchanged. No weight is fit to an EVAL result. The80/20 split is a new preregistered experimental choice, not a theoretically optimal or literature-guaranteed ratio. It increases point-prefix mass from22.5% to80%, approximately3.56x per prefix point, while reducing suffix physical weight; native loss still supervises the complete H40. It does not make H40 equal H9, mask the tail, smooth or retime an output, or make safety inspect fewer points.

This is ONE combined temporal weighting of physical pose/increment supervision, not a pure position-only, orientation-only, terminal-only or gradient-scale ablation. A constant error at all horizons has equal loss under both reductions, but real error distributions need not; altered gradient allocation is the intended treatment. Both losses have the same zero-error optimum. No additional terminal class, stop token, geometry oracle, progress scalar, history feature, second head, new teacher or sample curriculum is introduced.

## 5. Matched data, finite correctness tests, and actual training

Use the full original4096-unit PI0267 R plan with its original R_indices, commands, frames, raw labels, nominal/recovery slots and M1 T0-T7/M3 text. The source plan SHA256 is `fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef`. Confirm against the sealed source before use; do not alter source identities to match a guessed path. Each new arm consumes8192 microbatches. The common donor previously consumed only the first2048 units; this shared continuation expands exposure equally in both arms. Report overlap/new actual conditions and terminal counts, not a claim of an all-new dataset.

No FACTOR304 or opened EVAL examples are added to training. No synthetic new pose, dataset refresh, WBC data or sample rejection for poor model prediction. Existing valid nominal/recovery training labels remain fixed.

Reuse the qualified trainer with isolated narrow loss hooks. UNIFORM uses the original reductions without gratuitous algebraic rewrites. Verify weighted-reduction unit sums/masks, zero-error cases, matched constant-error scale, early-vs-late counterexamples, invalid-mask rejection, and independent numeric/finite-difference gradient checks on NON-LEARNED tensor fixtures. Real CPU decoder checks must include noncommuting rotations and the current unclamped ROT contract; do not accidentally import a historical clamping decoder. CPU array tests are not a model canary.

Real canary per arm: continuous8 versus4/full-save/fresh-process-restore4. Compare model, optimizer, scheduler, cursor/mode, actual data/noise and Python/NumPy/CPU/CUDA RNG. Retain the DataLoader RNG lifecycle and checkpoint-sidecar fixes. Save first-update S4 gradient evidence immediately after the already-required backward pass; no extra diagnostic backwards on the neural model. Canary weights discarded. A real gradient path through all four generation steps is required for the weighted physical losses; do not detach endpoint/prefix tensors.

After correctness/canaries, automatically train both arms from the common donor with fresh matched AdamW, LR1e-5, warmup10, cosine4096, weight_decay1e-5, max_grad_norm1.0, microbatch1 / accumulation2, exact existing trainable-name set and frozen backbone. Retain seed26741 and actual original unit-seed byte order/draw schedule; bind actual noise/tau packets across arms. Do not introduce new precision, optimizer or model capacity choices. Preserve the existing raw accounting conventions.

Save complete2048/4096 Trainer endpoints per arm (four total); ONLY final4096 gets quality evaluation. Never choose2048 after seeing4096, add steps, switch donors, or change PREFIX after seeing UNIFORM. Huge but finite losses are recorded; genuinely nonfinite optimization or corrupted input is an affected-arm correctness stop, not an excuse to alter LR secretly. A valid failure of one arm does not cancel the other preapproved arm.

Log per-task/phase raw uniform and weighted component values, gradient norms/clipping and actual sample identity from the existing pass, without extra model queries. Loss and quality are different. Unsaved internal activations remain UNOBSERVED.

## 6. Evaluation freeze and learning measurements

Before new model quality calls, freeze all evaluation identities/noises and the exact cards. Finish both endpoints before observing their new comparative quality. Reuse published donor outcomes rather than recomputing them. Three policy points: donor ROT2048, UNIFORM4096, PREFIX4096. No checkpoint selection and no best-of-noise.

### A. Existing968-row offline card per point

Reuse PI0268's224 FIT112x2,304 factors,440 legacy rows. Confirm that the FIT112 conditions occur in the FULL new4096 visits, with actual consumed identities under the unchanged ROT contract. All112 were chosen from its first2048 range, but actual new consumption must still be verified. Factor and legacy are evaluation-only. Three points give2904 logical rows, donor968 exact reuse expected,1936 new model queries. New policy binding includes processor/decode, not only weights.

Report fullH40 and indices0..8 physical errors separately, .4s endpoint error, per-task/command/variant signed predicted-versus-teacher displacement where nonzero, and terminal-label residual motion. Do not convert direction cosines on zero target displacement into favorable scores. Prefix error is evidence of fit, not completed task or proof of stability. Retain all rejected rows and all per-group denominators, paired changes, suffix errors and original legacy retention. Do not fit thresholds or weights to these values.

### B. Original64 closed-loop conditions per point

Three points give192 logical episodes. Reuse exact donor64, execute128 new final-policy episodes. Keep the same eight opened geometry families, M1/M3, commands and E0/E1, .4s replan interval,40s/100-query horizon, original noises, all-history semantics and native lifecycle repairs. These are DEVELOPMENT conditions, not renewed blind TEST or a second robot.

Each step must be fresh achieved RGB/StateV1 -> GR00T S4 -> unchanged gate -> raw model reference -> actual mathematical prefix execution -> new observation. No teacher/target/path projection repairs the policy; no snapping, online smoothing, oracle stop-on-success, trajectory stitching by an expert or removal of late failures. Full gate still checks H40. Mathematical execution and rendering do not become robot dynamics merely because GPU is used.

Keep original V3 qualification and V2 side results; retain the existing PI0268 reliability rule (task completed by35s, full safe40s, every saved tick35..40s within35mm/.10rad; M3 requires original complete ordered270-degree/12-bin traversal). Require at least13/16 for EACH task/backend and6/8 per command and original full-point safety. No changing the1mm reverse tolerance, entry acquisition or final5s after results. Zero reliable episodes is a valid possible result, not a mandate to retune.

### C. Bounded component-start diagnostics, not replacement task qualification

Purpose: separate inability to acquire a task from inability to follow/hold after reaching the relevant state, using only TRAIN geometries and no new training labels.

Use the first TWO original24 TRAIN families in manifest order, both commands and E0/E1. At each, create FOUR start types from the unchanged nominal teacher/geometry: M1 exact target pose; M3 exact entry pose; M3 midpoint of the270-degree arc (135 degrees from its required start); M3 exact terminal pose. This is32 cases per policy point,96 new diagnostic episodes total. Initial placement is explicitly a virtual diagnostic reset, not a claimed policy achievement, WBC reset, or replacement of the main64-case initial state. Show that the pose belongs to the correct direction/command and original camera/tool contract. No hidden start type or expert phase enters the policy.

Run exactly8s /20 replans, without online teacher actions or stop-on-success. Save original gates and all actual motion. For terminal starts report dwell/exit and final pose; for entry/midarc starts report correct local direction, actual ordered coverage from the initialized point and reversal, WITHOUT calling a midarc rollout a complete original M3 task. Original V3 may report missing approach/entry; preserve that label and separately show diagnostic progress. This panel has no selection/qualification authority. Fresh starting RGB must match the new initialized pose; never reuse inconsistent imagery. Static invalid visibility/geometry cases are explicit pre-query exclusions, not replaced. Reuse existing source/camera verification, do not start a new H0 search.

Bind actual diagnostic noise tensors before querying, identical across policy points for the same case/replan; use a declared PI0269 namespace distinct from training. No noise search. Both finals remain frozen even if donor diagnostics are unfavorable. Each new rendered state is observation-only and never enters training in this goal.

### Results and what they can establish

Report four separate items: executed-prefix fit, old-domain regression, original full task, persistent reliability/safety. A predeclared temporal-treatment contrast requires lower paired-median0..0.4s XYZ error on FIT in BOTH tasks and no increase in factor/legacy REJECT; report all SO3/suffix/legacy regressions even if this contrast holds. This contrast alone never creates a candidate. Compute terminal and midarc diagnostic effects independently.

A scoped RELIABLE_VIRTUAL_RESEARCH_CANDIDATE additionally needs BOTH original and reliable main64 qualifications, zero dangerous collisions/refusals, complete identities, and no increase relative to donor in legacy440 REJECT or its median XYZ/SO3. All models must be shown; do not average away a failed task or pick a favorable noise. No credible independent-generalization or physical claim follows from repeatedly developed scenes. Formal selected_checkpoint/deployment_selected_checkpoint remain NONE. Do not insert a new unrun qualifier after seeing results.

## 7. Resources, budgets, and engineering autonomy

This is one new budget, not a reset of PI0267/68. New official training8192 updates; planned real canaries32; technical completed-update reserve256; hard8480 completed updates and160000 recorded training heads (verify the existing18-head/update accounting before launch). Technical reserve cannot extend a healthy4096 endpoint. No third training/diagnostic fit run or extra backward experiment.

New inference: offline1936; main virtual <=12800; component-start diagnostics <=1920. Planned total <=16656 new queries, hard18432 attempts /73728 S4 heads. Exact reuse saves compute but cannot buy extra scientific conditions. Main+component new virtual starts224 planned,240 hard attempts. Up to17000 new virtual observation frame pairs, including technical reserve; no newly collected TRAIN, factor regeneration, new geometric family or external-model API call. Donor reuse gaps are explicit identity problems; do not silently add a complete baseline rollout under a different contract. A missing donor comparison must be resolved from exact stored identities, not concealed.

New WBC/dynamical-physics/real-hardware starts0. CPU teacher/saved-array analyses and per-output gate computations are not neural queries. Original simulation-asset privacy restrictions remain. No paid new service, credentials exchange, monitor/email task or broad storage cleanup.

PC coordinates one finite owner; Hopper performs actual admitted fine-tuning on lawful compatible resources under the same precision/runtime for both arms; Corelab performs no-gradient evaluation and rendering. Do not use H200 for ordinary CPU arithmetic. Verify actual user-fileset save space and full-checkpoint/staging overlap plus10GiB reserve, not a global mount's misleading capacity or an arbitrary300GiB target. Preserve full current recoveries until exact fixed-revision HF readback; archive pipelines can overlap evaluation when identities and durable sources are established. Only new/changed products need re-verification; do not repeatedly reread all historical archives.

Import/path/CRLF/JSON/serialization/camera-cache/lifecycle and exact-resume fixes within identical scientific semantics are autonomous; preserve attempts and narrow tests. Do not kill a healthy process, repeat completed valid episodes or fail the whole goal because a helper has not replied. New scientific loss/teacher/sample/controller/representation/precision/scorer choices beyond this document are not engineering fixes.

## 8. Completion and next decision

Deliver: immutable approved C/P card and loss-weight diff; reused source/donor/statistics identities; bounded diagnosis with missing/ambiguous rows; actual CPU/gradient-fixture and real-canary receipts; actual8192-update matched audit; four full2048/4096 Trainer HF readbacks;2904 offline logical rows,192 original-start logical episodes and96 component-start diagnostic proposals with explicit valid/excluded/attempt counts; independent numeric/prefix replay with shared-scorer limitations; full same-time reference/achieved/target analysis; source-linked virtual videos retaining failures; final budgets and a single canonical Issue #11 return.

Do not stop at CODE_READY, CPU_PASS, JOB_SUBMITTED, FIRST_ARM_DONE or UPLOADED. Valid quality failure can complete the bounded experiment. A real correctness/resource shortfall preserves all completed work and names the specific missing dependency; no indefinite signature loop. Keep old COMPLETE/FAIL/NONE and old partial WBC status intact.

Terminal: **PI0269_PREFIX_LEARNING_STUDY_COMPLETE — STOP_FOR_STATE_COVERAGE_OR_WIDER_TRANSFER_DECISION**.

Next directions are NOT extra launch permission:
- Better prefix fit plus safe reliable main task: test independently sealed new scenes and then separately validated execution backends.
- Better local terminal/midarc behavior but poor original-start task: examine decision-state coverage and staged acquisition, not another80/90/95% weight search.
- Prefix fit improves yet reliable task remains poor: investigate learner-visited TRAIN states with an explicit bounded expert-label/coverage comparison; do not import already-evaluated trajectories into TRAIN and call them new tests.
- Same genuinely consumed TRAIN prefixes still fail: bounded direct/low-dimensional policy or optimization feasibility baseline before collecting more scenes. Do not declare generalization failure from failed support fit.
- Numerical/short-horizon behavior improves but visual/WBC transfer fails later: separate observation-domain, feasibility, installation and tracking; do not make high-level weights a hidden compensation model for one WBC.

Publication review scope: final report and canonical, current source/loss/teacher and donor receipts were read. The reviewer ran only the supplied constructed CPU arithmetic fixture, not original model-array replay, a neural gradient, a renderer, WBC, SSH/PBS or training. Script success is not an experiment result. The goal authorizes execution after its limited correctness checks; it does not claim that it has already run.

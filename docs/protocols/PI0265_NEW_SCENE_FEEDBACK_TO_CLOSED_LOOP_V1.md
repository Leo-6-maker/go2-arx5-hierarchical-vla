# PI0265 — New scenes and measured feedback to a scoped closed-loop EEF policy

Authority: `PI0265_NEW_SCENE_FEEDBACK_CLOSED_LOOP_GO_V1`.
Publication: AUTHORIZED_NOT_EXECUTED. One lead owns implementation through actual simulation, data, matched training, evaluation and archival. Optional helpers never form a signature dependency. The storage assistant remains separately responsible for storage, not scientific execution.

## 0. Decision and claim boundary

This is an explicit NEW scientific authorization for bounded simulator dynamics/WBC execution and conditional new-scene training. It overrides the prior zero-physics/new-scene prohibition ONLY within this protocol. It does not unlock old TEST/DEV-F, hardware, WBC retraining or unrestricted DAgger. Historical FAIL/NONE and the PI0264 terminal remain unchanged.

Immediate question: can a frozen low-level controller execute the reference interface on new M1/M3 scenes, and does adding actual controller-feedback demonstrations improve complete receding-horizon task execution versus matched old-data continuation?

This is a change of mainline from old-scene wording fit to scene/state-support and execution. Do not add another old-template schedule or treat a 2/160 Q difference as proof that B universally beats L. PI0264 shows learnability, not generalist qualification: L720/720 on added TRAIN wording; Q X131/160, B141/160, L139/160. Its prospective group counts are X7/12, B9/12, L11/12. Thus L improved group coverage without a superior aggregate Q count. All full qualifications remain FAIL.

Read immutable source commit `dd816b24443f9ae4c0358305aeea91c91c4200c2`, canonical5925913074, `reports/PI0264/final/MODEL_QUALITY_REPORT.md`, `FINAL_COMPLETION_AUDIT.json`, `reports/PI0258/CONTROL_A0.md`, `reports/PI0256/CLOSEOUT.md`, `docs/ACCEPTED_BASELINES.md`, actual reference/scene code and their runtime manifests. Accepted Stage41 demonstrated reference-only repeated handoffs on limited conditions, not current-model task success. PI0258 A0 records column-packed 6D actor conditioning in a compatibility shim versus row-packed PyTorch3D semantics, but actual checkpoint training-time imports and a complete preview replay are not established. It also records a bounded quaternion-vector error proxy, not the full SO3 logarithm. These are distinct facts: do not silently replace the native error proxy, change gains, or declare a unique cause.

Publication review read repository sources and reports, not all raw HF archives, current SSH state or videos. None of the new control/data/model stages has been run by publication. Resolve actual source paths from repository/runtime manifests rather than guessing a runner name. Reuse the accepted Stage41 lifecycle and PI0257-v2 evidence tools; the latter's GUARD_STUB_ONLY is not an implemented physical backend.

## 1. Fixed research donor and ownership

Use PI0264 **L2048** as the sole new-training initializer:
- HF `leoyuer/pi0246-gate-pass-campaign`;
- revision `e4290e6ca387abd2de4337b5f12b9c72a5f85c99`;
- prefix `arms/PI0264/L/checkpoint-2048`;
- complete Trainer tree `2f61d8ed8c8fe8444b7ef23164cb4df2bcacd8a73c93f19be77fcdf459ef6fae`.

L is chosen prospectively for broad supervised-wording fit and11/12 prospective groups, not because it won every metric or passed qualification. Do not switch to B/X after observing new-scene outcomes. Both training arms inherit these weights, with fresh optimizers. The initial donor itself is one of the final physical comparison points.

Keep the PI0264 processor, statistics, tokenizer, H40/dt=.05, structural132/active10 action convention, inference precision and full-S4 objective. Training text is the eight already authorized M1 T0-T7 forms, balanced within each chosen target/observation; M3 retains original text. No new wording generation, Q/V supervision, prompt rewriting, phase/source/goal-coordinate tokens, output repair or new statistics fit. New observation data are real simulator RGB and measured FK state, not a new input channel.

Notify the storage ledger that designated L2048, its actual dependencies and PI0265 working/checkpoint/transmission roots are protected. Do not wait for storage cleanup completion, delete active sources, or require300GiB free. Reuse valid fixed-HF identities and accepted runtime/source receipts.

## 2. Finite G0: bridge contract and independent metric implementation

Before a simulator start, pin actual WBC actor weights/configuration, URDF/TCP, native simulator/control timestep/decimation, reset path, camera calibration, native gates/termination and stage41 reference-only handoff. CPU-test frame/time/rotation conversions on nonidentity rotations, translations and the recorded states. Preserve the low-level state/history schema and native quaternion-vector error feature; do not replace it with a SO3 logarithm merely because its function name suggests one.

Trace requested world SE3 -> canonical/native table -> future preview -> current/history actor conditions -> arm/body/dog action. Independently decode the stored physical reference and current TCP. Required coordinate roundtrip residuals: position<=1e-5m, SO3<=1e-4rad; source identity and complete finite/quaternion/orthogonality checks are mandatory. Tests include identity,90degree rotations about each axis and mixed noncommuting rotations. No fitted offset from a model failure.

Two prospective, isolated low-level representation versions only:
- LEGACY: actual recorded shim and native conditioning, after the already proved camera-cache isolation fix.
- ROW6D: replace only the implicated matrix-to-6D compatibility boundary by the first-two-ROWS packing, and make every corresponding consumer/preview decode consistent with that packing. Matrix/quaternion world semantics, native error proxy, weights, gains, action scale/clipping, dynamics and timing stay unchanged.

The second version is a scientific controller-conditioning hypothesis, not a proved universal correction. First inspect available checkpoint training/config/import metadata. If training convention is still unknown, record UNKNOWN and perform the bounded comparison; this unknown alone does not require another authorization. If the actual checkpoint requires column packing, document this as counterevidence; do not silently transpose weights or convert unrelated features. The production version must have an internally consistent physical reference/preview representation. A demonstrated implementation defect inside either specified version may be fixed prospectively; adding a third representation, altered reward, gains or actor is not authorized.

Camera observation must not mutate root/DOF/raw rigid state, derived TCP, actor history, target cursor or RNG. Test exact saved-state capture before/after rendering. Additional diagnostic overlays/masks must never enter policy RGB. Capture pre-autoreset terminal state; when unavailable mark it unavailable, never display the post-reset pose as a failure pose. Record active gripper drives accurately: the prior fixed-zero drives are not freely passive joints.

Implement the concrete task and safety metrics in section4 from saved analytic poses, with negative tests for wrong target, endpoint shortcut, reverse contour order, clock/frame mismatch, stale observation and post-reset telemetry. Freeze their code/thresholds and all scene/reset/noise cards before new physics. This finite source-specific work is required; do not restart full-history audits or old language diagnostics.

## 3. New scene population and protected splits

Keep the common asset topology, dimensions, marker pack and synthetic sensor calibration from PI0260; use its real scene factory and offsets. M2 remains deferred, with its obstacle/distractors and original collision geometry present. No grasp/push task is claimed.

New numerical geometry tuples, in meters:
- x shift {-0.03,-0.01,0.01,0.03};
- y shift {-0.03,-0.01,0.01,0.03};
- z shift {0.0075,0.0225};
- contour radius {0.040,0.050}.

These64 tuples are new samples within the earlier local workspace envelope, NOT new rooms, broad object categories or extrapolation beyond the workspace. Exclude actual numerical overlaps with all previously used/protected geometries using metadata only, including MT-TEST aliases. Sort surviving canonical JSON tuples by SHA256(`PI0265:26501:scene:`+canonical_json). Assign split BEFORE outcomes: first4 CONTROLLER_DEV, next24 FEEDBACK_TRAIN, next8 NEW_SIM_EVAL; the rest UNUSED and NOT a refill pool. At least36 disjoint tuples are required; report a real shortage instead of enlarging the grid.

Within each split assign the six ordered distinct color pairs from red/green/blue cyclically by the frozen rank; neither command is globally one color. Preserve source target/yaw and contour semantics. Freeze initial root/DOF and camera transforms, scene geometry, reset seed and text for every episode. No special per-task camera zoom, image highlighting, hidden goal coordinates, robot teleport after reset, visual-only obstacle removal, or failed-scene replacement.

Reference semantics must come from the actual source, not a guessed earlier summary: `pi0259_kinematic.reference` uses M1 quintic approach over6s; M3 approach over4s followed by12s along the270degree arc. Retain those curves/times, target yaw and measured-reset anchor, extending final target hold to a20s episode horizon. H40 future references beyond the motion endpoint remain terminal hold, not clipped-time jumps. The three coordinate layers (world task, native WBC, GR00T StateV1/decoded reference) have separate explicit transforms.

All physical trials use normal simulator reset to the registered initial state. The source's initialization/pre-roll may take at most1s and is included in counts; do not add a new unscored5s settling search. The old0.5s stationary preparation requirement is NOT reused as a gate before a purposeful dynamic task. This does not reclassify old settling failures; task tracking and endpoint holding are now evaluated directly. If native startup cannot be made valid within the defined initialization, record failure and do not improvise a teleport/pose search.

FEEDBACK_TRAIN and NEW_SIM_EVAL remain disjoint. No model quality on NEW_SIM_EVAL, including donor, is examined before both final new checkpoints are frozen. Expert evaluation on that split is likewise deferred to final evaluation. Data feasibility/visibility failures stay in the fixed episode denominator; do not select the eight evaluation families by controller or model success.

## 4. Actual physical task definitions, not H40-only proxy scores

All task metrics use independently reconstructed WORLD achieved TCP; requested/installed references are scored separately. Gates retain their actual pinned configuration from the current model and native runtime. This protocol changes the experiment, not those frozen gate thresholds.

M1 success: achieve the requested target position within.035m and orientation within.10rad for a continuous.25s before20s, with no episode safety failure. Also report final error and time-to-success; a minimum-distance hit alone is not success. A wrong-color target never counts.

M3 success: enter the instructed start neighborhood, follow the actual270degree arc in the requested direction, then reach the other endpoint within.035m and orientation.10rad for.25s. For ordered path coverage, divide the analytic instructed arc into12 ordered angle bins. A bin can be credited only from measured TCP within.015m of the corresponding arc, in the correct monotonic traversal order after start admission; record all bins and progress. Require all12, no reverse traversal or crossing the unrequested90degree gap as a shortcut. Current TCP near the final endpoint alone is insufficient. Freeze the geometric implementation and test a direct endpoint shortcut as FAIL.

Report reference-to-target/path error, achieved-to-target/path error, same-tick reference tracking and SO3 error; do not subtract independently minimized distances to claim tracking. Report M1/M3 and both commands separately, including all invalid, rejected and terminated cases.

Safety failure: original gate REJECT, native failure termination/fall, NaN/Inf, forbidden self/obstacle contact under the existing collision exclusions, stale/invalid reference or unsafe limit penetration. Do not disable termination/collision/limit checks. Soft-limit penalties alone are reported per joint, not treated as proof of dangerous contact or silently erased. For a prospective simulator numerical-penetration criterion, angular hard-bound penetration above1e-3rad or translational hard-bound penetration above1e-5m terminates that episode; smaller penetration is still recorded. Commanded actuator targets retain the original clipping/scale and limits. This is an explicit new simulator admission criterion, not a retrospective waiver of PI0256 soft/hard flags. Missing units/limits must be resolved from the model before execution.

A rejected candidate reference is NEVER installed. Log it, end the scored episode, and use only the native safe shutdown mechanism; no expert takeover or repaired trajectory is counted as policy success. A valid failed episode is not retried. An individual failure does not stop the remaining independent fixed cases unless it reveals a systemic integrity fault that makes further results invalid.

## 5. G1: bounded controller diagnosis and qualification

This stage runs expert references only, with no GR00T queries.

A. Four C0 diagnostic episodes: LEGACY/ROW6D x direct-native/reference-provider. Restore the original sealed C0 reference/reset identity from its manifest; preserve source attitude, do not rigidly transport it to an arbitrary reset orientation. Use original moving reference plus terminal hold, at most5s. Compare requested vs installed reference at identical native timestamps; provider/direct matrix/position parity and valid initial actor conditions are mandatory. C0 task hit and tracking are descriptive, not proof of general6D holding.

B. On the first two CONTROLLER_DEV families, run both commands of M1/M3 under each of the two representation versions:8 episodes/version,16 total,20s maximum. This is a prospective comparison of representations, not selection of the best noise or arbitrary pose. Freeze all inputs before the first episode.

A version is eligible for confirmation only with valid G0 reference semantics, at least7/8 complete task successes, at least3/4 for each behavior, and all successful episodes having tracking XYZ RMS<=.030m, p95<=.050m, SO3 RMS<=.10rad. These are NEW dynamic-controller criteria; original offline quality and old preparation verdicts are unchanged. Preserve all raw metrics and fails. At most one episode may fail; no unsafe output is continued to seek a hit.

If both qualify, prefer the version matching independently documented checkpoint training convention; if that is unknown or both match, prefer LEGACY only if it has a fully consistent reference/preview interpretation, otherwise ROW6D. Do not rank by a favorable tiny metric difference. Confirm the chosen version once on the third CONTROLLER_DEV family, all four task/command episodes. Require4/4 successes and the same criteria. The fourth DEV family is reserved/unexecuted, not an adaptive rescue.

G1 has24 planned starts:4+16+4. If no version qualifies or confirmation fails, stop downstream physical collection/training at `PI0265_CONTROLLER_ENVELOPE_BLOCKED`; still deliver the full fixed controller comparison, actual video, and identified reference-vs-actuation boundary. Do not resume another old-scene language run or run an unbounded controller search. If it passes, automatically freeze one runtime and proceed; no new routine GO or helper signoff.

## 6. G2: actual-feedback expert data, not kinematic pose playback

Execute all24 FEEDBACK_TRAIN families x M1/M3 x both commands:96 expert episodes under the admitted WBC/runtime. Existing analytic task references are the expert commands, installed through the same native provider/clock used by models. The native actor, simulated dynamics and real camera feedback must actually run. No setting DOF/rigid pose every recorded frame, prerecorded camera substitution, expert-prefix injection into a model output or hidden simulation resets.

Capture every native control tick's measured root/DOF, independent TCP, installed reference/preview, actual actions and termination. At times0,.4,...,19.6s capture fresh body/wrist RGB and StateV1, paired with the analytically correct futureH40 reference at that physical timestamp. Simulator inference-time pauses do not advance the clock. Expert phase/target/world geometry may be used to construct supervision and score, but NEVER enter policy conditioning. Desired reference and achieved TCP are separate columns. This is teacher-controller state support, NOT DAgger learner-state aggregation or a proof of on-policy robustness.

Do not require a fixed-base IK solver to reproduce every actual WBC pose as an admission gate. Verify measured states by independent FK and native joint/collision checks. The WBC may move the base; the state/image/relative-reference transform must use that actual root, not the nominal fixed renderer root. Validate actual processor roundtrip and current-frame/future-label alignment on moving-base examples.

Keep all valid expert data and all failures. For training admission require at least8 complete new families in which all four episodes succeed and all required source/codec/phase checks pass; retain EVERY eligible family among the24, not just the easiest8. No refill. At least95% of the declared50 observation windows per eligible episode must pass original raw gate and codec checks, with the initial observation and each actual movement phase represented. Terminal-hold windows alone cannot satisfy phase coverage. Episodes ending early for success may continue expert terminal hold for data only if no safety termination, explicitly logged; policy evaluation may stop on success.

Use one sample manifest fixed after this deterministic eligibility rule. Keep unsuccessful episode records in the collection denominator and out of supervised successful-trajectory claims. Use existing TRAIN-only processor/statistics without refitting; expose out-of-range values rather than clamp. Do not relabel measured actual trajectories as desired labels or average away hidden-time aliases. A known same-command temporal alias remains a declared limitation, not a new proof-of-unique-future gate. Cross-command mislabels or real input leaks remain blockers.

If eligible family count<8 or actual codec/state integrity fails, close `PI0265_FEEDBACK_DATA_SHORTFALL` or the concrete integrity blocker. Do not silently fall back to synthetic-only data and call this feedback training complete. Where valid, raw recordings and actual videos remain useful partial deliverables.

## 7. G3: matched old-only versus mixed-feedback training

Two arms start from the identical fixed L2048 weights:
- R4096: old COMPACT_30_10 TRAIN only;
- D4096:50% old TRAIN replay +50% new eligible FEEDBACK_TRAIN observations/labels.

This tests the package of new-scene and actual-controller-state data at fixed compute; it does NOT separately identify camera-domain, geometry, state-noise and temporal-feedback causes. No new labels from a learned-policy rollout are collected in this protocol. A separate DAgger intervention would require another authority.

Freeze a512-base-unit bank with128 units per (M1/M3)x(SHARED_OLD/TRANSFER_SLOT). Each unit contains two supervised microbatches, one per command. Cover all eligible families and active reference-phase quartiles by deterministic family round-robin, then lowest valid reference index not yet used; complete the bank by cycling the frozen valid lists. For SHARED_OLD, R/D use identical original samples. For TRANSFER_SLOT, R receives a separately frozen old-data list, D the new feedback list, matched for task/command/phase bucket, not falsely claimed identical images. Initial paired observations are called identical only when actual byte checks support it; dynamic opposite-command observations need not be identical.

Traverse eight full passes over the512 units:4096 optimizer updates/arm,8192 supervised examples/arm. Task balance50/50, command balance50/50. M1 template T0-T7 cycles once per base observation over eight passes using an original-unit-hash offset; M3 original instructions unchanged. R/D share matching text template slots, actual native epsilon/tau/free-S4 packets and microbatch order. No V/Q text or EVAL data enter optimization. Save source/template/frame/label identities and actual random packets.

Use the mature two-microbatch mean-loss full-S4 mechanism, fresh AdamW LR1e-5, betas(.9,.999),eps1e-8,weight_decay1e-5,clip1,warmup10/cosine4096. Original FP32 action-side trainable/BF16 frozen backbone scope and inference precision remain. Objective stays native+.4position+.1increment+.1rotation+.1peak-speed margin. No LoRA/unfreeze/quantization/new auxiliary objective.

Real restore canary for each arm: continuous8 versus4-save/fresh-process-restore4, checking new data/mixed-slot cursor, source identities, actual noise, weights/optimizer/scheduler under existing tolerances.32 completed canary updates discarded, main reinitializes from donor. Save complete2048/4096 checkpoints. No interim quality screen. R then D sequentially; both endpoints and identities freeze before any NEW_SIM_EVAL model or expert result is opened. A valid R quality FAIL never changes D.

## 8. G4: fresh-observation receding execution and final evaluation

Implement an actual episode backend by adapting the accepted native lifecycle; do not simply rename the PI0257 guard stub.

Loop: validated reset -> current body/wrist RGB and measured StateV1 -> native GR00T S4 inference -> physical decode and original SafetyGate -> reference-only native installation -> execute.4s of actual WBC dynamics -> new observations -> new inference. No physical/history reset at handoffs. H40 remains40x.05s; each execution interval advances8 reference samples, subject to exact inherited native-clock interpolation. Log real first/last sample timestamps and reference cursor. Do not replay expert-state images into this loop.

Simulator may pause while inference runs; label the experiment NON_REALTIME_LOCKSTEP and separately record inference wall latency and simulated time. No real-time claim. Up to50 GR00T queries in a20s episode. Freeze actual standard-normal packets for each `(PI0265, eval-family, task, noise-seed, replan-index)` before training. Use two new noise seeds per model, shared across matching commands/model points without encoding the correct label. Actual later observations differ naturally across closed-loop policies; do not describe their post-action inputs as paired-identical.

NEW_SIM_EVAL: all8 fixed families x2 tasks x2 commands=32 scenarios. Run expert controls once each (32 starts), and donor L2048/R4096/D4096 each with both frozen noise streams (192 policy starts). No geometry/visibility/teacher failure is discarded or replaced from this scorecard. Report (a) all32 fixed scenarios and (b) an explicitly secondary expert-executable subset; never hide the full denominator or call them32 independent real-world environments.

Collect complete per-episode query/gate/installed-reference/achieved-TCP/termination chains and actual body/wrist videos. Score full tasks using section4. An old offline semantic FAIL does not forbid this already authorized diagnostic simulation. A new raw reference REJECT still ends its episode without installing it. Do not require old wording panels to turn PASS before running these fixed safe diagnostic cases. Conversely, safe-looking videos cannot bypass native safety checks.

Offline retain the original440-row Motion2/O64 legacy panel for donor and each final arm; donor440 rows reuse existing exact results. Add a fixed64-row feedback TRAIN fit panel per point:16 source windows per behavior, both frozen old evaluation noise packets, all selected by family/phase/ID before model results. Do not pick the lowest-loss windows. The prospective physical outcomes are primary; these offline panels diagnose retention and input compatibility, not a new old-score hunt.

No main-training changes from evaluation. The evaluation data and outcomes are not folded back into TRAIN in this goal. No online expert rescue, retrospective prompt routing, output smoothing, nearest-expert selection, best-of-noise or failure replay to claim success.

## 9. Quality and scoped research admission

Report separately: controller-runtime qualification, data qualification, training completeness, offline legacy quality and actual closed-loop task performance. Passing one does not repair another historical FAIL.

Prospective `new_scene_sim_research_pass` for each new final model requires:
- all required scenarios, both noise streams and telemetry valid;
- per task and noise: at least13/16 full successes, and at least6/8 for EACH command;
- no fall/native safety termination, forbidden collision or hard-penetration failure; every rejected reference remains an episode failure, never applied;
- original four offline KIN-VAL numerical groups still pass, with M3 legacy semantic retention reported and all four M3 groups>=10/12.

Also disclose expert reliability and the full task/noise/command margins. If expert controls achieve fewer than13/16 successes for a task, that task's controller envelope is not confirmed; no model receives full Motion2 sim admission from a filtered subset. A policy's remaining old M1 language FAIL is still a restriction, even if these two fixed language conditions succeed physically.

At physical evaluation use M1 T0 for c0 and T1 for c1 on even family ranks, reversed on odd ranks; M3 uses the exact original direction instructions. These are familiar supervised forms. Thus any scoped pass is on new local geometries and actual feedback, NOT proof of arbitrary-language robustness. Keep old Q/V scores visible in the retained source record; no claim that they were repaired.

An endpoint meeting these prospective criteria may be listed as `sim_research_candidate` with exact scope/runtime/processor/scenario card. If both pass, list both without post-hoc cherry-picking. Formal historical `selected_checkpoint` and `deployment_selected_checkpoint` remainNONE; this goal does not issue real-robot deployment authorization or a universal model label. Failure is valid science and stops additional experiments, not a reason to extend training.

## 10. Budget, resources and engineering autonomy

This is a NEW allocation; retain all older counters.

- Physics/WBC:24 controller diagnosis/confirmation +96 feedback TRAIN expert +32 new-eval expert +192 model episodes =344 planned starts;352 hard cap including8 strictly no-valid-result technical attempts. Per start at most20s task+1s initialization (C0 at most5s); total simulator-time cap7392s. Before first start derive separate control-tick/physics-substep caps from the pinned timesteps, never silently change dt/decimation; include implicit resets and failed starts. Raw WBC actor calls and native step counters are all recorded. No spare budget becomes additional scenes/noises.
- Policy episodes:192x50 gives at most9600 in-episode GR00T queries. Offline new queries:880 final legacy +192 feedback-fit=1072. Total planned new<=10672; hard11000 attempted queries /44000 S4 inference-head calls, including technical failures. Donor old440 are reused. Cache savings do not fund new cases.
- Main training8192 updates/16384 supervised microbatches; discarded canary32; completed technical recovery reserve256; hard8480 completed updates and210000 raw training heads including recomputation/failures. No extra steps beyond4096 per arm. Save only complete two-microbatch steps.
- No WBC training, hardware, old TEST/DEV-F payload/execution, external model-generated dataset, new loss/architecture/template search, email/monitor/daemon or PR merge.

Corelab: source/CPU work, dynamics/rendering and frozen-model evaluation under actual compatible GPU/memory conditions; do not kill other projects. Hopper: admitted GR00T training/canaries, lawful compatible allocations; no H200 for ordinary CPU debugging. No blind duplicate qsub. A missing scheduler tool meansUNKNOWN. Simulation/controller support must use pinned actual runtime, not infer a new installation works from imports.

Within frozen definitions the lead autonomously fixes path/import/CRLF/serialization, reference-lifecycle implementation, finite geometry/scoring bugs, renderer-state aliasing, staging and exact recovery. Narrow tests and prospective version receipts suffice; do not rerun the entire history. No mechanical two-job engineering stop. No helper/cleanup signature or routine PI GO after a stage genuinely passes. Evidence of a new unsafe mapping or invalid data pauses affected descendants, while independent CPU/report work continues. Scientific changes beyond the two defined conditioning versions, scene/card design, loss or task thresholds require a new decision.

Coordinate low-I/O storage work: protect L donor, current full resume, raw controller/data evidence and upload/readback overlap. Stream verified archives where possible, avoid concurrent tar/unpack/full-download duplicates, keep next-write overlap plus10GiB reserve, quotaUNKNOWN if unmeasured. The separately authorized storage assistant may continue unrelated historical retirement. No new broad cleanup here.

## 11. Deliverables and completion

Required: source-bound rotation/reference diagnosis and negative tests; pinned two-version controller comparison and confirmation; new split/geometry/reset/noise cards; actual expert feedback dataset and all collection failures; two restore receipts and four complete training checkpoints; frozen-model192-episode scorecard and32 expert controls; exact query/tick counts; full legacy/feedback-fit panels; saved-array independent-path CPU recomputation; fixed-revision HF integrity/readback of permissible model/results; actual closed-loop videos with fresh-observation/query/handoff counters.

Use ACTUAL_WBC_CLOSED_LOOP, EXPERT_CONTROLLER or OFFLINE_PREDICTION labels correctly. No kinematic pose replay presented as actual execution. Geometry metrics and world transforms must be inspectable, with full failure videos kept. Hardware/private robot/WBC assets and credentials stay out of public archives. Same-owner replay is not an invented external reviewer.

Successful execution terminal (regardless of quality): `PI0265_NEW_SCENE_FEEDBACK_CLOSED_LOOP_PILOT_COMPLETE — STOP_FOR_WIDER_SCENE_OR_TASK_ADMISSION`.

Earlier scientific stops name the actual result, e.g. controller envelope blocked or feedback data shortfall; they are PARTIAL for the overall goal, not proof that training or universal EEF is impossible. Deliver completed actual physics and source evidence; do not fill missing stages with synthetic substitutes.

Next decisions, NOT additional authority: if D improves over R/donor on new-scene full tasks, expand prospectively sealed geometry/appearance and eventually carefully supported new skills. If references are right but tracking fails, prioritize low-level support/runtime in a separate control study. If tracking is adequate but target/direction wrong, prioritize grounded scene/action coverage. If both policies stall or drift from teacher support, a separately scoped learner-state aggregation/DAgger study becomes justified. Do not resume arbitrary old-scene template schedules. This milestone is a bounded two-behavior simulator bridge, not a claim of general-purpose or real-world manipulation.
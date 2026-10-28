# PI0274 — Pose equilibrium, rotational restoring behavior, and reliable feedback

Authority: **PI0274_POSE_EQUILIBRIUM_AND_ROTATIONAL_RECOVERY_GO_V1**.
Date: 2026-10-05 SGT.
Review base: `1a0bbf952c96bcff4f686dc39603bc3545ef5397`, existing branch `pi0267/executor-decoupled-recovery`.
Predecessor: PI0273 release `b6e4e436dced30445a014721e2c0df268989244f`, terminal Issue #11 comment `5990237456`.
The new Issue #11 canonical binds this protocol and `reports/PI0274/design/REVIEW_AND_BUDGET.json` to their immutable release. Publication is permission to execute the specified work, not a claim that it has started.

## 0. Decision, hypothesis, and scope

GO for **one new S4 arm, EQUIL5120**, compared with the already completed, precisely bound **PI0273 BROAD5120** control. Both start from the SAME original PI0270 AGG4096 weights and use the same fresh optimizer, 5120-update schedule, base data, context order, actual random-packet contract, loss, precision, and inference. Do NOT initialize EQUIL from BROAD or from S4FIT, and do NOT retrain the completed BROAD control.

The sole primary training intervention is the source of the existing 2048 supplemental microbatches: replace broad-context replay with fixed task-goal-centered examples consisting of the exact nominal goal pose and signed PURE orientation perturbations, with freshly consistent RGB/StateV1 and the UNCHANGED ArclengthCorrection teacher. The full 8192-micro base stream remains intact. This is new, bounded local state supervision, NOT another ratio search or a repeat of PI0273's existing 296-state selection.

A frozen-policy field panel measures pose fixed-point bias, rotational restoring response, and position/rotation cross-coupling before and after learning. The goal then runs the ORIGINAL-start task panels, not only fitted states. The principal repair hypothesis is reduction of M1 post-arrival SO3 drift without losing arrival, M3 path behavior, safety, or legacy. Full M1/M3 qualification is still required for a reliable virtual research candidate. This goal does not promise to repair the separate M3 entry/order problem.

No new action head, history/phase/stop token, learned completion latch, online teacher, goal snapping, rotation clamping, action averaging, reference smoothing, alternative normalizer, precision experiment, WBC, or physical run is authorized. In particular the ground-truth goal is used ONLY for offline TRAIN data construction and analysis; neither policy nor execution/termination receives it as a new input.

One Codex main owner continues serially to the specified terminal. Helpers are optional, not required signatories. Reuse valid predecessor artifacts, check actual locks/jobs before starting, and do not cancel healthy work, create duplicate jobs/automatic goals, reset old budgets, force-push, merge, add permanent branches, or revive PR77 pointers. Isolate new code/artifacts under `scripts/pi0274_execution/`, `reports/PI0274/`, and PI0274 runtime roots; never modify a sealed running predecessor runtime.

## 1. Reviewed evidence and limits

PI0273 is fully closed: terminal5990237456 at 2026-10-05 15:43:10 SGT, closeout commit1a0bbf9 at15:43:44. Both5120 endpoints, four complete26-file Trainers, 10272 completed updates including32 canary,184944 training heads,17152 quality calls/68608 heads,128 main+64 component trajectories, saved-frame videos and both result archives are complete. Technical completed recovery updates and quality retries are zero. Missing scientific requirements are empty. Detailed report SHA256: `ce4fbd502c7029b70d6ed94e5ea36bbc95417df50d828eacf1a63ea1c94db38d`.

BROAD: original48/64, reliable9/64, zero collisions; TARGET: original39/64, reliable3/64, seven virtual collisions. Both FAIL original/reliable point qualifications. Both PASS original-ROT-relative LEGACY. BROAD improves M3 first completion from AGG2/32 to16/32, but its9 reliable cases are all NEW: it loses all6 formerly reliable AGG cases. Do not describe the net+3 as retention of those old successes.

The previous narrow-fit explanation is insufficient: PI0273 improves early TRAIN prefix XYZ and SO3 versus AGG in BOTH arms. M1 early XYZ AGG/BROAD/TARGET=2.701/1.278/1.324mm; M3=2.120/1.115/1.390mm. Nevertheless M1 reliable counts are BROAD5/32 and TARGET0/32. Thus this is not merely the same early-panel regression seen in PI0272. TARGET is worse than BROAD on59/96 M1 and73/96 M3 early position comparisons despite better two-family terminal fitting. Terminal directions are0/48 for both, but task reliability remains poor.

BROAD M1:27/32 first-success exits;13/32 final-window XYZ failures and26/32 SO3 failures. In the32 actually read BROAD M1 diagnosis rows, disjoint final-window groups are12 both,14 SO3-only,1 XYZ-only,5 neither. TARGET:32/32 exits and SO3 failures,16/32 XYZ failures. These final-window facts do NOT establish that rotation drift causally preceded translation drift. A0 below reconstructs timing rather than asserting it. M3 no-entry and post-entry failure remain distinct; flags overlap. TARGET collision cases stay valid quality/safety failures.

The actual source teacher `ArclengthCorrection` has a fixed nominal endpoint and exponential residual correction. Where its projection is terminal, p(h)=g+exp(-2h)(p-g), with analogous left-SO(3) residual decay. At0.4s the teacher residual ratio is exp(-0.8), approximately0.449329. This is conditional on the ACTUAL teacher projection; it must not be imposed on other path states. Earlier teacher code already included terminal behavior; do not claim there were no terminal examples.

The original synthetic recovery generator perturbs translation AND same-axis/same-sign rotation together (10mm and.06rad). It does not systematically provide all pure rotational directions at a fixed position. Prior learner trajectories may contain such errors, but their presence and balanced goal-local coverage were not guaranteed. This motivates a specific falsifiable coverage hypothesis, not a proof that missing pure rotation is the unique cause.

Local reasoning, NOT an experimentally measured model: a receding-horizon pose-error map can behave as e_next=b+A e+noise. Small pointwise fitting error alone does not imply a zero-bias fixed point or a restoring neighborhood. Persistent bias can give an offset even for a contractive map; amplification/cross-coupling can drive further departures. Relative actions do NOT inevitably drift. The new field experiment measures this distinction with actual rendered inputs. Stable-BC (arXiv:2408.06246; authors' https://collab.me.vt.edu/Stable-BC/) provides related error-dynamics motivation; this protocol does not implement its stability loss or import its guarantees.

Reviewer scope: live branch/latest ledger, complete PI0273 report and closeout, first70 lines of the diagnosis CSV including all32 BROAD M1 rows, A0 source manifest, actual PI0273 train/virtual code and source teacher/generator. No SSH, raw full-trajectory replay, new model/teacher query, training or job operation was performed by the reviewer. Container download of a small HF CSV failed; a large GitHub same-tick CSV returned empty content. Those files are NOT claimed independently reread. All-source numerical replay remains supported by the predecessor's receipts, not a new reviewer execution.

## 2. A0 — bind closed sources and reconstruct drift timing without new neural work

Accept and reuse the closed PI0273 report, `FINAL_CLOSEOUT_RECEIPT.json`, canonical5990237456, fixed whole-file/member archives and their hashes. New source identities:
- PI0273 summary HF revision `5cd59ada4e39eaeac57734817f74ee58207a1acd`, prefix `evidence/PI0273/full-support-replay-20261005/summary`, tree `7644c534cea4ef0c7842d72ddfd8af48b58132ea4a67f5797b8919ff24dc33a7`.
- Predictions revision `eb52b2915bd865066964cad71700f5cfebac30ed`, corresponding predictions prefix, tree `62697e20d984e2851985da5c42f3884a42be09945838598fc48d0680c3a2a991`.
- PI0273 FITTING_CARD SHA256 `0f455c3f0ebf7528b95cbb4f4a4d53fa671f27e987697176cc65ce1f610344ac`; base8192 occurrences/6716 unique processed input-label pairs; old BROAD main sample audit `c939cfdd80d0ccf62e77fcb3e07b9d0e492eb57f8bd28bfb828554fa8b44d624`, step_rng `c7306e4bd9f14dcaee9203e12f6274f612b4d731b1e0aaddd3e963186d13bdb3`. Bind the actual closed source logs, not an identically named stale helper.

Using the ALREADY SAVED128 main episodes and their original query/trajectory/debug records, add a bounded descriptive onset table: first original success, first subsequent XYZ exit and SO3 exit separately, ordering of exits, per-query reference0/.4s versus actual pose, same-tick tracking, and later visibility loss under the original visibility function. Analyze all cases, not just extreme drift. If saved segmentation/camera evidence is absent for a field, record UNOBSERVED; do not rerender or alter eligibility retrospectively. Existing table/replay calculations can be reused, with only new onset/visibility joins computed.

Report whether SO3 error begins while XYZ is still small, whether visibility loss precedes or follows departures, and whether reference or tracking is responsible at the matched instant. These are temporal associations, not automatic causality claims. Preserve actual references, no threshold-derived stop/success behavior. For collisions, retain the original last accepted reference and actual swept path; do not call a generic gate a physical/full-body safety certificate. No new teacher labels, model calls or renders in A0. A missing unique-root-cause proof is NOT a launch blocker.

Control policy for new field queries: original BROAD5120, HF `leoyuer/pi0246-gate-pass-campaign@5bed9a0d209ccaf23cfcd8bb28fb5a4c41a699c4`, prefix `arms/PI0273/BROAD/checkpoint-5120`, tree `038a9b53de8d455a6a5a5c844a18538e14cd43f727ca3566929b694b880ac188`.
Common training initializer: ORIGINAL AGG4096 weights, HF revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`, prefix `arms/PI0270/AGG/checkpoint-4096`, tree `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`.
Use the original ROT statistics SHA256 `0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9`. No full historical checkpoint redownload or repeat closed readback unless a genuinely required local dependency is missing/corrupt.

## 3. A1 — fixed goal-local field data, with consistent RGB and unchanged teacher

Use ONLY the original24 `FEEDBACK_TRAIN` families, M1/M3 and two commands:96 cells. Do not collect new EVAL labels or use any A3 development labels in training.

For each cell, construct the existing pinned ArclengthCorrection teacher and take its NOMINAL final curve pose (p_star,R_star)=at_length(ell_end). This target is offline label-generation geometry, not policy input and not online success feedback. Preserve the teacher's actual runtime source binding; inspected repository teacher blob `a6d934f900d680afc137f67442d3bae40ef21f07`, original generator blob `b590e8fea14401b69af16489a92c60dfc4b63b41`. Source-sealed loaded files take precedence over a guessed same-name import.

Predeclare13 physical proposals per cell, all in WORLD axes:
- center: (p_star,R_star);
- rotations rx+/rx-,ry+/ry-,rz+/rz-: p=p_star, R=Exp(plus/minus .05*axis) R_star;
- translations px+/px-,py+/py-,pz+/pz-: p=p_star plus/minus .005*axis, R=R_star.

There are1248 proposals total. ONLY center and six PURE rotation proposals (672 physical pose proposals) may supply new training examples. The576 pure-translation proposals are diagnosis-only and must not be introduced into the supplemental training stream. These are poses, not automatically distinct processed/language examples or independent scenes.

At every proposal use the original renderer to obtain consistent body/wrist RGB at the actual pose, original StateV1 construction and TCP/camera extrinsics. Do not alter StateV1 while keeping an image rendered at a different pose. Keep original RGB resolution, deterministic processor, masks, gripper. Physical query time for these static samples is0; it is not an added model input. Reuse an old exact packet only when pose, camera/scene/renderer identity, label and complete source evidence match; count and preserve reuse explicitly.

Call the UNCHANGED teacher.label(actual p,actual R) once per new proposal, retain fullH40, aliases, actual ell/projection/terminal metadata, original label gate, tool collision and original visibility checks. Do not overwrite a label with zeros/goal poses and do not force `terminal_hold=True`: pure rotations may move the teacher's projection, especially in M3. Preserve continuous residual/teacher-endpoint quantities; the old exact-zero1e-12 category is not evidence that near-hold behavior is absent. Floating-point input roundtrip differences must be measured, not snapped away.

Data admission: all96 cells must have the center and all six rotation samples valid under the existing label uniqueness/visibility/gate/codec checks. Verify their expected use in the frozen schedule before training. A genuine failure yields a literal DATA_INCOMPLETE stop with exact proposals/reasons and no replacement radius/family/seed; it does not authorize inventing labels. Invalid diagnosis-only translations remain registered as excluded; do not replace them or manufacture a full6D derivative from missing columns. No output quality is used for data eligibility. Repeated valid generation is prohibited.

Freeze pose/label/member hashes, all outcomes, field card/noise, exact processed inputs under all scheduled languages, the complete training schedule and inherited final-evaluation cards before any new neural work. Archive/read back new training materials; reuse original B materials and old proofs. One-time/cached CPU source encoding is allowed when exact dependencies match. Report actual new render/teacher counts (maximum1248 each), not invented full counts when a packet was reused.

## 4. A2 — one EQUIL5120 training arm; precise reuse of BROAD control

Copy the source-bound PI0273 BROAD10240-micro schedule, its global unit/micro order and exact context sequence. Every one of its8192 BASE occurrences is copied bit-for-bit: RGB, StateV1, language, label, mask, source row and processor. Same4 base updates then1 supplemental update; no base displacement.

For the2048 supplemental slots, preserve the original slot's family/task/command/exact language and full global order. Maintain a separate zero-based counter per (family,task,command), deliberately not per language. Select from the ordered seven training poses [center,rx+,rx-,ry+,ry-,rz+,rz-] by counter modulo7 and increment. Copy the entire matching actual new RGB/state/teacher-label packet, using that slot's original language. No splicing a label from another pose. Verify all672 permitted physical poses are used at least once; disclose exact language/condition multiplicity and overlaps with B and historical C. Do not assert all field T0 inputs were actually consumed: the exact hash+label+language test decides.

Both the reused BROAD and new EQUIL are initialized from AGG4096 and have10240 microbatches/5120 optimizer updates with fresh optimizer. The SAME data/initialization seed27303, SAME `PI0273:TRAIN:27303:{global_update}` seed derivation and native two-micro RNG lifecycle are intentional to reproduce the matched-control contract, not an accidental forgotten version update. Require actual native epsilon/tau/generated initial-noise identities at every global unit/micro to match the closed BROAD logs; same seed alone is not sufficient. Shared8192 base input/label hashes and every supplemental context must match their prescribed control counterparts.

Use actual PI0273 S4 Trainer/native helpers, 471 trainable and559 frozen tensors, unchanged AdamW LR1e-5,warmup10,cosine5120,weight_decay1e-5,max_grad_norm1,microbatch1/accumulation2. Loss remains `Lnative+.4Lposition+.1Lincrement+.1Lrotation+.1Lpeak_speed`, original full nativeH40 supervision, PREFIX temporal weights and full39-transition speed term. NO new contraction/Lyapunov penalty is added. No DIRECT hooks or backbone unfreezing.

Preserve verified R1 precision: training native BF16 epsilon/autocast, original mixed parameter storage, FP32 processed VL/query/Euler and physical loss; quality original model-only BF16/query with FP32 decode. Explicit phases, not `.training`, define the boundary. Same-phase parity only, original tolerances. Preserve actual Prescott encoder/library identity and frozen statistics; no clamp/refit. Reuse current actual source bindings, not obsolete V1 BF16-training wording.

One real EQUIL continuous8 versus4-save/fresh-process-resume4 canary (16 completed updates) under the5120 schedule is required; do NOT redo BROAD's passed canary. Check exact model/optimizer/scheduler/cursor/RNG recovery, actual first-backward four-node gradients, scope, modes, and matched control random packets. After PASS, execute EQUIL5120 without quality-driven early stopping. Save complete2560/5120 Trainers with provenance, source dataset/card hashes and readout/precision identity; evaluate5120 only. Reuse old BROAD checkpoint/results. Any CONTROL_NOT_COMPARABLE finding blocks a false causal comparison; it does not authorize a second control run without a new decision.

CPU/runtime engineering wrappers may adapt ROOT/path/arm/metadata/counter allowlists and install the registered new dataset. They must not change science, silently eat old loop sizes, modify sealed old code or turn small finite losses into a reason to retune. Nonfinite optimization or a concrete input/gradient/precision contradiction is recorded and stops only the affected continuation under the existing recovery budget.

## 5. A3 — frozen-policy local map measurements, before and after calibration

Evaluate BROAD5120 and EQUIL5120 on the fixed1248 proposals at TWO exact noise tensors per cell. For j=0,1 derive a noise seed from SHA256 of `PI0274:FIELD:27403:{family}:{task}:{command}:{j}`, first8 bytes big-endian modulo2**32; use NumPy PCG64/Generator.standard_normal of shape(1,40,132), dtype float32, and persist actual bytes/hash before model calls. Use the SAME tensor across all13 poses within that cell and across both policies for a draw. The two noise banks are registered, not selected for outcomes. Preserve actual BF16 initial-noise bytes, S4 traces, complete decodedH40 and original gate results.

Use the original T0 language for M1 and unchanged M3 command text for ALL13 poses of a cell, so finite differences do not secretly change the instruction. These are TRAIN-family constructed diagnostic poses; actual consumed condition/label/language hashes are marked individually. Translation-only members are never training examples. No claim of independent blind TEST,192/2496 independent scenes, or online deployment robustness follows.

Maximum new field queries=1248*2 policies*2 draws=4992 (2496 each). The BROAD field may be measured before EQUIL training after all cards are frozen. Do not adapt the seven training poses, amplitude, ordering or sample weights based on that measurement. A poor/rejected model output is a valid quality result, not reason for a new draw. Missing/invalid data proposals are excluded with original denominators; incomplete banks remain incomplete rather than imputed.

For each accepted finite output, analyze the nominal E0 .4s endpoint and all original prefix/full errors. With D=diag(.005,.005,.005,.05,.05,.05), define scaled local input z=D^-1[p-p_star, Log(R R_star^T)] and output f(z) analogously at the decoded reference index8. The rotational perturbation convention and the output log convention must both be left/world. Keep positions and rotations separately in physical units as well.

For each draw compute b=f(0), finite-difference columns A_i=(f(+e_i)-f(-e_i))/2, symmetry/nonlinearity residuals f(+e_i)+f(-e_i)-2f(0), and position/rotation cross-blocks. Compare to the actual same-input teacher map, with exp(-.8) cited ONLY where the teacher terminal assumption holds. Keep per-draw values; a two-draw mean is descriptive. Report spectral radius/singular values only for complete valid matrices, and explicitly label them finite-scale estimates, NOT analytic Jacobians, Lyapunov proofs, robust stochastic stability certificates or eigenvalues of WBC. An empirical singular value>1 indicates sampled directional expansion, not a theorem about all trajectories. E1 also has memory/lag, so no E1 certificate is inferred from E0's map.

Primary local readouts: center .4s translational/rotational bias; signed rotational correction and perpendicular/cross-coupled movement at each pure angular perturbation; input-to-next goal-error ratio (undefined at a zero denominator, not set to zero); first/prefix/full fit and original gate. Some maps can be locally contractive but biased, some restoring only on one axis, and some non-restoring. Report all, including evidence against the motivating hypothesis. Do not subtract scalar minima or marginal medians to manufacture tracking/error propagation.

These field calculations use predictions already counted above; no finite-difference autograd calls, differentiable renderer, extra model head calls, output averaging policy or executed perturbed-start episode is authorized.

## 6. A4 — mandatory original-start feedback, retention and safety evaluation

For EQUIL5120 execute exactly the closed PI0273 memberships:
- offline FIT224+FACTOR304+LEGACY440:968 logical rows/up to968 new queries;
- early NEW_FIT192:192 queries;
- late/component296 at original j0/j1:592 queries;
- original eight developed EVAL families, M1/M3/two commands/E0/E1:64 main episodes,40s/max100 queries;
- original component32 proposals:8s/max20 queries.

Reuse BROAD's valid matching rows/episodes/CPU results and original AGG/ROT comparator data. No fresh BROAD main/control run and no TARGET repeat. Maximum new non-field queries=8792; combined with field4992 gives13784. All memberships and existing non-field noise are frozen BEFORE new training. Main success/evaluation labels do not enter the new data.

Inference remains RGB+language+actual EEF feedback -> native S4 -> fullH40 gate -> execute only0..0.4s -> next ACTUAL pose/frames. H40 contains40 points0..1.95s at.05s; E0/E1(.15s) and20ms saved steps are unchanged. Do not freeze actual pose after success, impose goal orientation, replace generated reference with the teacher, clamp rotation, stop on an oracle, or change the replanning rate. Full40s except original safety stops. New task-goal supervision is training-only; the runtime observation/worker contract must prove no goal coordinates, phase, success flag or target IDs leak into the actor. A component starts at a registered placed state for diagnosis only, not full qualification.

Primary intervention question: does EQUIL increase M1 reliable cases relative to BROAD5/32 and reduce post-arrival SO3 departures, while retaining first completion and not sacrificing other domains? Report matched case gains/losses/retention, not only net counts. Compare M1 final-window XYZ-only/SO3-only/both failures and temporal onset. Keep M3 entry, path, after-success reversals, terminal behavior and safety separate; M3 completion cannot be inferred from endpoint proximity or the new local field.

Complete candidate qualification is unchanged: per task/backend original and reliable at least13/16 and per command6/8, full-point safety, original completion by35s/fullsafe40s/all251 final-window ticks in35mm/.10rad, plus original-ROT LEGACY. ROT exact legacy XYZ=.015121491176463085m, SO3=.046877545849429575rad, REJECT0; do not use rounded values or substitute another baseline. Keep first-completion V3 semantics and report later reversals separately, without changing the rule after viewing failures. Each performance/retention/safety criterion remains separately visible.

Improved local response alone is NOT success of the repair. If local maps improve but original-start M1 holding does not, explicitly reject a sufficient goal-local-coverage explanation. If M1 improves but M3 fails, call it a targeted M1 advance, not a solved shared task policy. If both reliability and legacy/safety qualify, record a reliable VIRTUAL research candidate only. Formal selected/deployment remain NONE.

## 7. Budgets, resources and reuse

All predecessor budgets/FAILs are retained. PI0274 adds:

| Consumption | Planned maximum | Hard ceiling |
|---|---:|---:|
| New formal training arms/updates |1 /5120|1 /5120 healthy updates|
| New canary updates |16|16 planned|
| Total completed updates incl technical recovery |5136|5264 (128 technical reserve)|
| Training heads including recomputation/parity |92448 nominal plus bounded parity|98304, with up to256 parity heads INSIDE cap|
| Goal-local physical pose proposals/teacher calls |1248|1280 teacher calls incl technical repeats, no new poses|
| Training-permitted local physical poses |672|the fixed center+six rotation poses only|
| Diagnosis-only translations |576|the fixed six translations only|
| New field inference |4992|inside total quality cap|
| New total quality attempts |13784|14336|
| New quality S4 heads |55136|57344|
| New main/component starts |64+32=96|104 incl technical recovery|
| New observation frame pairs |1248+7040=8288|9216 incl technical recovery|
| New WBC/dynamic physics/hardware |0|0|

This is bounded synthetic task-space supervision, not target-robot teleoperation or data conditional on WBC success. Teacher calls and renders used for new labels are explicitly counted, NOT hidden as CPU checks. No additional neural calls for analysis/video, and no reserve converted into new axes/radii, ratios, seeds, arms, endpoints or experiments.

Use the existing compatible single-GPU qualified setup. Check actual quota, live jobs and write/staging/checkpoint overlap+10GiB; use scheduler facts, not inferred project credit. Retain only authorized discarded-canary cleanup, no broad removal of recoverable main checkpoints. Persist literal attempt/head/update/frame/teacher/start journals before work, including incomplete technical attempts. Reuse first valid scientific outputs after interruption; never retry a collision or bad valid prediction to improve a score. A queued job is not permission to submit a competing job.

## 8. Preservation, automatic continuation and terminal

Deliver A0 onset/visibility/source diagnosis; fixed pose/label/card/noise/exposure/eligibility/overlap manifests; new dataset archive and needed local stage checks; exact BROAD-control comparability proof; new actual canary and5120 training audit; two complete2560/5120 Trainer fixed-HF whole-file readbacks; full field and endpoint/feedback tables with actual-seen flags; independent arithmetic and saved-array/trajectory reconstruction with shared scorer/gate/renderer/executor disclosed; all new failures; one first-family EQUIL ALL8 saved-RGB video with actual timestamps and VIRTUAL/NOT_WBC/NON_REALTIME labels; new output/data/summary archives with fixed-version full-file/member readbacks; literal budgets; one terminal Issue #11 canonical and exact readback.

Old BROAD/AGG videos, raw predictions and whole-file readbacks are reused by identity, not remade. Renderer is not called again to make a video or a graph. Administrative filename/arm/root/serialization/kernel/library issues may be fixed narrowly with receipts; scientific data algorithm, input/loss/teacher/precision/scorer/controller changes require a new decision.

If the next approved operation is feasible, continue through dataset, actual learning, field/main/component evaluation, CPU/video/archive and publication without waiting for another routine GO or helper signature. Do not stop at DATA_READY, low fitting loss, completed training or upload-only. A genuine unavoidable data/source/resource/semantic block has a precise partial return and preserved counters; no fake PASS.

Terminal: **PI0274_POSE_EQUILIBRIUM_RECOVERY_COMPLETE — STOP_FOR_STRUCTURAL_FEEDBACK_OR_INDEPENDENT_TRANSFER_DECISION**.
If this goal fails despite improved local maps, do not automatically add more terminal replay. Use the onset/visibility/field evidence to choose ONE later structural feedback/representation/observation-history question. If M3 entry is still the limit, address it separately rather than pretending hold data encodes completed path history. No later intervention or real robot/executor transfer is automatically authorized. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE` throughout.

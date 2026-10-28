# PI0277 — Single-task StackCube repair with measured H200 acceleration

Date: 2026-10-06 SGT.
Authority: **PI0277_SINGLE_TASK_GENERATED_GRIPPER_AND_H200_FASTPATH_GO_V1**.
Research charter: `GROOT_OBJECT_MANIPULATION_MAINLINE_CHARTER_V1`, release `ed398fee17151cfec47864fed9d48ce149b28d61`, Issue #11 #5994405991.
Reviewed implementation/result SHA: `0110cbde2d1c0c1c4c28087d52a419fda4d7d597`; publication preflight SHA: `2ee7178f4f8d8b39749d1ceeaadd69fe1862cb98` (only archive-status additions since initial review). The new Issue #11 canonical binds the release containing this protocol and `reports/PI0277/design/REVIEW_AND_BUDGET.json`.

## 0. Decision and goal

**GO for actual H200 runtime acceleration qualification, a matched two-arm task-repair experiment, and teacher-free physical StackCube evaluation.** This is not another preparation-only goal. The first priority is to make the existing small object task work; no new skill, visual-domain expansion, moving robot, WBC or hardware experiment is authorized.

Keep the task, actual TRAIN episodes, task processor, source Panda/EEF executor, images, action ABI, S4, precision and pose losses fixed. Both arms start from the same sealed TASK8192 weights with fresh optimizers and matched 4096-update schedules. CONTROL4096 continues the old loss; GRIP4096 adds one differentiable loss on the **fully generated S4 gripper command**. This distinguishes the proposed repair from merely doing more training. No claim that gripper supervision is the unique cause or that this intervention must succeed.

Use a measured, equivalence-qualified H200 input/metric fastpath for both arms. A worker count in a config is not speed evidence. CPU preparation already exists but has NOT established actual processor parity, multiprocess throughput or GPU equivalence. This goal now explicitly authorizes the bounded missing measurements and real-model canaries.

| Charter field | Binding |
|---|---|
| object_task | Existing StackCube: grasp red cube, move it onto green cube, release, and preserve the native physical stack predicate. |
| high_level_contract | body/wrist RGB + measured world TCP and measured opening + the same fixed instruction -> H40 XYZ/row-rot6d/commanded opening. No joints, object truth, teacher phase or success in policy inputs. |
| frozen_wbc | Senior Go2+ARX5 WBC remains unchanged and NOT CALLED. Panda is the source-domain executor, not a second WBC or hardware result. |
| data provenance | Reuse PI0276 scripted source / commanded-FK / EEF-replay data; no task-data expansion or target-robot teleoperation. |
| primary intervention | Additional generated-command gripper loss, with matched extra-training CONTROL. Acceleration is common to both and must preserve the reference numerical path. |
| success | Actual grasp/transport/release and persistent object success; trajectory fit is diagnostic only. |
| non-goals | Multi-task, new appearance/camera domains, DAgger, new teacher labels for TRAIN, geometry-only curriculum, deployment. |

## 1. What the predecessor actually established

PI0276 fixed result: native joint and EEF experts each31/32 first and persistent TEST successes. BASE0 and TASK8192 each0/32; TASK8192 TRAIN diagnostic0/16. BASE0 TEST32 all stopped at raw gate. TASK8192 TEST32 completed the entire physical horizon without task success; TRAIN has15 completed physical rollouts and1 valid gate refusal. This is not simply first-success-then-drift: no model episode established even first stack success.

TASK8192 has sustained native grasp >=0.3s in8/32 TEST and5/16 TRAIN cases, so fine-tuning did change behavior. Its offline192-condition panel has0 REJECT, median XYZ0.0123378420m and SO3 0.00523420478rad, but mean gripper-command MAE about0.306469. TRAIN prefix8 gripper MAEs by source group0..3 are0.4604363441,0.1125450134,0.2936730981,0.5000934601. TRAIN prefix8 XYZ means are0.01577521,0.00724547,0.03059775,0.00205766m. An aggregate low pose median is not precise grasp timing or transport accuracy.

Code facts: PI0276's `gripper_error` detaches generated and target channel9, records MAE, and returns the pre-existing pose loss unchanged. Native gripper loss/gradients exist; the gripper was NOT entirely unsupervised. The old auxiliary objective explicitly supervises generated position/increments/rotation/speed, not generated gripper. Intermediate gripper coordinates can affect later S4 position through shared attention; do not assert all gripper gradients in S4 must be zero. The new experiment tests missing explicit generated-command supervision, not an already-proven disconnected native label.

TEST TASK has3126 IK-hold ticks and mean command-to-actual XYZ RMS0.012001m. These are source-Panda execution diagnostics, not senior-WBC errors. Visible case30000 has no IK holds and still fails. IK therefore cannot be asserted as the sole explanation. Gripper saturation occurs only9 TEST control ticks and cannot automatically explain all32 failures. Local grasp acquisition, carrying, alignment, release and post-release stability must be separated.

PI0276 model training, final policy evaluation, CPU scoring and videos are complete; full source/prediction archival closure was still pending at review. Do not relabel the goal fully archived. Adopt actual owner/process state and complete those independent old receipts. Preserve all old failures, exact results, denominators, model identities and budgets; no predecessor rerun or optimizer reset.

## 2. Single owner and source binding

Use one existing main owner, separate `scripts/pi0277_execution/` and `reports/PI0277/` roots, and the existing branch. Do not cancel healthy archival work, create a competing automatic goal, require helper signatures, add a monitoring service, use PR77 as a pointer, or migrate to another platform. CPU inspection can proceed alongside old archive readback only without IO contention. H200 work uses legal PBS allocation; source physics uses the existing isolated Corelab source environment when available. No login-node stress test or GPU compute outside PBS on Hopper.

Read source-bound implementations of PI0276 train/dataset/admit/infer/evaluate/report/replay/source_worker, inherited PI0273/PI0270 Trainer and actual imported S4/temporal/rotation helpers. Record `__file__`, file SHA256, controller/asset/package identities and effective args. Do not import a source worker in CPU preparation: it installs physics hooks. Do not use an upstream moving GR00T main or a same-named historical helper as the actual runtime.

Retain ManiSkill `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`, StackCube-v1/panda_wristcam, CPU PhysX100Hz, native control20Hz, original drives/materials/cameras and existing128->96 RGB resize. Keep absolute pd_ee_pose semantics (base-frame XYZ and intrinsic XYZ Euler as implemented), world/root/TCP transforms, row-rot6d and quaternion conventions, separate actual opening and command fraction. No new rotation convention, controller gains, IK fallback, shader or physics setting.

## 3. A0 — bounded read-only failure audit before training

Reconstruct from already saved PI0276 source, model and physical arrays. No new model query, physics episode or teacher is needed for this stage. Produce `FAILURE_FUNNEL.csv`, `TRAIN_EXPOSURE.json`, `GRIPPER_OBJECTIVE_AUDIT.json` and source-bound receipts.

For every original TASK TEST32 and TRAIN16, report raw gate outcome; first/longest native grasp interval; lift height relative to initial red position; carrying loss; first on-top occurrence; release before/after on-top; first/persistent task success; IK holds; commanded and measured opening; native contacts; and command->installed->achieved pose errors. Use lifted>=0.02m for>=0.3s only as a named DIAGNOSTIC, not a replacement native success predicate. Preserve aborted runs and missing channels; an observed grasp-to-release transition is not automatically an intentional successful release. Timestamp each component, rather than interpreting a scalar MAE as the whole episode's cause.

Join saved offline rows and consumption logs by immutable UID: distinguish actually consumed TRAIN windows, same-episode unconsumed windows and VAL. Compute generated prefix/full gripper OPEN/CLOSE error, event-transition timing and pose error without discarding gate failures. Source teacher timeline groups are offline labels, not an oracle phase assignment to a diverged learner. Report contradictory/near-identical inputs with different labels where supported; do not invent state aliasing from a failed task alone.

Audit all93 admitted TRAIN command sequences for actual open/close transitions, genuine H40 labels, masks, affine gripper mapping and event timestamps. Verify original native-loss channel weighting/denominator in the resolved source; do not assume a1/132 factor from output dimensionality. Confirm generated gripper logging was detached and that the final decoded gripper is not overwritten at execution. Reuse actual codec/positive source proofs; no full repeated dataset generation or HF donor downloads.

If an actual semantic contradiction is found (wrong labels, channel, timestamps, decoder or simulator transport), do NOT silently repair that scientific contract and then test the planned loss under the same identity. Stop `SEMANTIC_CONTRADICTION_REQUIRES_AMENDMENT` with the isolated mismatch and available results. Normal residual pose errors, source IK holds or native-loss nonzero gradients are not such contradictions and do not block the already-authorized matched experiment. Do not require this audit to prove the unique cause before proceeding.

## 4. Fixed data, donor and matched schedule

Initializer for BOTH arms: PI0276 TASK8192 weights only.
HF `leoyuer/pi0246-gate-pass-campaign` revision `7687ad4c04341b30c6a2a0959aeadf299fa3ba0e`, prefix `arms/PI0276/TASK/checkpoint-8192`, full Trainer tree `b78e300af57772c22ab8327918463afb1ea4b176689dbeba2ab243d4a27ae175`. Reuse the completed26-file proof, verify local weight files; no optimizer inheritance or donor tournament.

Reuse task statistics SHA256 `b3da3d500d514ce3d1ce83001de30846d9ab76d699dc48831db8e4367ce98572`, original processor, relative-SE(3) transform,132D embedding with10 active channels and all masks. Data archive: revision `0f64f053ad69b6b4803e712000a0b136cc4623d3`, tree `d30e98ac1979e1f6a8ad98c3131d50f98eb97492fb087a89895921a2f25c9fd1`. Existing positive TRAIN93 and VAL15 are unchanged; do not add source/learner episodes to TRAIN.

Each arm consumes the FIRST8192 microbatch entries of the original16384-entry TRAIN_PLAN, in exact order, with4096 optimizer updates (batch1, accumulation2). These are registered repeated task-training visits, not new independent demonstrations. The four source groups have2048 visits each. Reuse immutable UIDs/pixels/labels; record exact event exposure rather than claim every one of55,893 windows is consumed. No new sampling weights, hard-state selection or target-WBC filtering.

Freeze one new training-noise card using the original per-update RNG construction with explicit PI0277 namespace and seed27703. Match actual native epsilon, tau and S4 starting-noise tensor bytes/dtypes across CONTROL/GRIP for each visit; same integer seed alone is insufficient. Loader/cache generators are separate. Maintain fresh AdamW, lr1e-5, cosine4096-step context, warmup10, weight_decay1e-5, max_grad_norm1, and actual inherited beta/epsilon values. Exactly the donor471 trainable/559 frozen name/shape/storage-dtype inventory; no VLM unfreeze, new head, joint labels, history or phase input.

Precision remains the actual R1 contract: original mixed stored weights and native BF16 epsilon/autocast; generated training features/query/Euler states FP32; physical losses FP32; model-only evaluation BF16 query and existing FP32 physical decode. Do not substitute training BF16 query, TF32, FP8, quantization, fused optimizer, compile, DDP or fewer S4 steps. The new policy identity includes the unchanged processor plus the new weights and training-loss provenance.

## 5. One primary scientific intervention: generated-command gripper supervision

CONTROL objective is the exact old objective, including original operation order and temporal reductions:
`L_old = L_native + .4 L_position + .1 L_increment + .1 L_rotation + .1 L_speed`.
GRIP objective: **`L_old + .1 L_gripper_generated`**.

Let `u[k]` be the existing final generated normalized action channel9 and `u_star[k]` its label. Use the source-verified differentiable affine inverse to physical actuator fraction; under the fixed task action contract this is `g=(u+1)/2`, `g_star=(u_star+1)/2`. Confirm this against the actual codec before use. Do not use measured finger opening as the command target.

`L_gripper_generated = sum(m[k]*w[k]*((g[k]-g_star[k])/0.10)^2) / sum(m[k]*w[k])`.
Here `m` is the unchanged action-channel9 validity mask; `w=0.8/8` for indices0..7 and `w=0.2/32` for8..39. The new gripper prefix is the EIGHT actually executed commands; existing pose terms retain their old NINE-point/0..0.4s convention unchanged. This distinction is intentional, not permission to rewrite the pose loss. Empty/invalid gripper masks are data errors, not zero loss.

The0.10 command-fraction reference and coefficient0.1 are fixed prospective choices for a meaningful opening error; no coefficient sweep. Compute in FP32 without detach, sigmoid, rounding, thresholded open/close, clamping or extra generated pass. Add the new scalar once to the returned total objective; do not hide it in position loss or corrupt existing component receipts. Reuse the SAME S4 rollout graph. Existing native gripper supervision stays intact.

CPU analytical/finite-difference tests must verify zero error, nonzero error, mask invariance, prefix weighting and affine inverse. On real canary data, record isolated final generated-channel gradient and gradient reach through all four S4 steps. Distinguish native-path and generated-path observations. A missing explicit generated penalty is not a claim the old shared transformer had no indirect gripper gradients. No physics differentiation or privileged online teacher.

Retain old pose/gripper/error logs and add L_gripper_generated, its weighted contribution and prefix/full actuator-fraction errors. Report task and pose trade-offs: learning to close accurately can still harm approach/transport if shared weights change. Loss下降 is not task PASS.

## 6. H200 acceleration is implemented and measured in this goal

Use1 H200,12 allocated CPUs and225GB host RAM as in the qualified source; one sequential PBS owner. Benchmark in the same allocation; prioritize old archival IO and do not run a competing model server on the same GPU. The preparation return is PARTIAL, not an acceleration receipt. Read saved actual arguments: worker0/batch1/accum2/pin_memory=true.

Implement three CPU input routes before CUDA construction:
- REF_W0: unchanged logical PI0276 input construction/hash checks.
- FAST_CACHE0: deduplicated immutable, already-processed CPU inputs/labels for only the registered TRAIN visits, with compact per-visit metadata and ordered consumption.
- FAST_CACHE2: the same verified cache with2 CPU spawn workers, prefetch_factor2, persistent_workers and ordered delivery. No4/8-worker sweep in this goal.

Processed-input caching here means deterministic CPU processor outputs, NOT neural/backbone features. Key by raw-data/processor/statistics/resize/source hashes. Verify every cached condition/label/mask against old proofs and an original CPU construction; store read-only shards/memmap with member hashes, no duplicated H40 RGB banks. Do not cache scientific random tensors or advance sampler from prefetched samples. Cache budget<=48GiB, total process-tree host RAM<=180GiB and within225GB allocation; bound worker metadata replication and count attempted/discarded input work.

Workers must not import Trainer/model, initialize CUDA, install simulator hooks, mutate global model RNG or write scientific-consumption journals. Do not apply the CPU-only preparation monkeypatch that forbids torch Module calls to the model's parent process. Main process commits next-unconsumed micro-index at optimizer boundaries and reconstructs queues on resume. Test indices0,8,256,4096,8190 and stream end, including interruption after prefetch. Use installed ordered DataLoader API; do not upgrade libraries for an `in_order` argument. Preserve CPU BLAS identity required by old hashes.

Keep pin_memory=true and verify actual pinned CPU tensors in the H200 path. Same-stream non_blocking H2D may be used after value/order checks; no new CUDA stream scheduling. Persistent workers alone are not a speedup claim. Consolidation of already-detached diagnostic scalar transfers is allowed only if all original values, event order, nonfinite checks and reconstructible journals remain identical; no removed scientific checks or graph-retaining async queue. Keep durable per-operation reservations and fsync at completed update/checkpoint/error boundaries. Optimizer/model/physics operation order remains unchanged.

Run bounded real CPU input measurements on the exact first1024 TRAIN visits: up to2 fresh-process passes per route,64 warmup plus1024 measured calls per pass. Count prefetch, cache construction, parity/cursor work and training-input fetches under one PI0277 CPU-item hard cap65,536. No VAL/old TEST/fresh confirmation input is used to pick performance settings. Fresh process does not imply cold OS disk cache; do not evict shared caches.

Select the faster correct cached route for GPU qualification using CPU timing and memory, defaulting to FAST_CACHE2 on a measurement tie. The GPU workload is the old CONTROL objective. Two paired trials per route: REF_W0 and selected FAST each run96 updates from identical TASK8192 weights/fresh optimizer/RNG, full4096 scheduler context, no quality evaluation. Run order REF-FAST then FAST-REF. Exclude first16 updates as warmup and save per-update completed timings for the remaining80, using CUDA events/profiler or synchronized interval endpoints. Exclude model-load/save/network time from steady throughput and report end-to-end preparation/cache cost separately. No unsynchronized host enqueue time presented as GPU compute.

These384 updates are discarded qualification work, not part of either4096 arm. Compare same-visit inputs, actual noise, loss components, outputs/gradients and final model/optimizer/RNG/scheduler at8 and96 using original exact standards. Save candidate8 for the resume comparison below. Profile at most one already-counted interval; no uncounted repeated model pass. Capture loader wait, H2D, GPU/CPU timeline, peak VRAM/RSS and metric/checkpoint costs; metrics without measurements remain NOT_OBSERVED.

H200_SPEEDUP_PASS requires>=10% median steady-update throughput gain, improvement in both paired trials, equivalence PASS and resource safety. If the selected cached route fails speed but not correctness, the other cached route may get two96-update trials (192 extra updates maximum), reusing the same baseline recordings in the same allocation. Choose the fastest equivalent route and freeze it identically for CONTROL/GRIP. If neither cache route beats REF, explicitly record SPEEDUP_NOT_ESTABLISHED; use the fastest equivalent route to complete the task-repair goal rather than pretend a speedup or add an unbounded tuning loop. This satisfies actual acceleration work, not a promised acceleration result. If no candidate passes numerical equivalence, STOP_RUNTIME_EQUIVALENCE_FAIL rather than alter precision or train with corrupted inputs.

## 7. Real resume and formal training

Selected FAST CONTROL:4 updates, save, fresh process, resume4; compare the8-step result with the FAST continuous trajectory captured in the benchmark. This adds8 updates. GRIP: continuous8 versus4+freshresume4, adds16 updates, using the same selected FAST path. Both exact comparisons must pass before formal training. Full losses, gripper logs, input/noise order, optimizer, scheduler/RNG and permitted frozen/trainable weights are checked. Canary weights are discarded.

Then run CONTROL4096 and GRIP4096 sequentially, each independently from TASK8192/fresh optimizer. No reuse of CONTROL learned weights to initialize GRIP. The first128 updates are a health observation of each continuing run, not a reset or a score-triggered stopping point. Final physical evaluation is mandatory even when offline gripper error is disappointing, provided execution integrity is valid. No extra healthy updates, LR search or intermediate checkpoint tournament.

Save complete2048 and4096 Trainers for each arm. Evaluate only4096. Each Trainer includes weights, optimizer/scheduler/RNG, processor/statistics, parameter inventory, loss and fastpath contracts, actual source/consumption/noise identities. Fixed HF full-file readback for all four; reuse existing donor proofs and archive completed immutable artifacts without blocking healthy compute on optional signatures. Retire only explicitly discarded benchmark/canary payloads after comparison and durable evidence; never delete the sole recovery copy or unique task data.

New driver must use a typed run binding for roots, labels, inclusive/exclusive limits and metadata; do not blindly replace numbers or ROT/TASK literals throughout old code. Qualified math is reused, not a wholesale new Trainer. Preserve source copies and actual adapted AST/source hashes. This is a new runtime seal, not a patch to PI0276's already-executed files.

## 8. Fixed evaluation: prove this one small task before expansion

The old PI0276 TEST30000..30031 has now informed failure analysis. Name it **REPAIR_DEV32** in new reports, preserving its historical TEST label and old scores. It is not a new blind test. Reuse its old expert and TASK8192 results without re-running valid old policies. Reuse the same old16 lowest admitted TRAIN diagnostic initial states, not a success-selected subset.

For BOTH final new arms, unconditionally run:
- original192-condition offline TRAIN/VAL panel, same saved inputs/noise (384 new queries total);
-16 original TRAIN physical diagnostic starts each;
-32 REPAIR_DEV32 physical starts each.
This is96 new learned physical episodes,<=7680 model requests. Old TASK8192 physical results are reused as reference, not invented from new runs.

Additional fixed fit panel: in each of the old16 TRAIN diagnostic source episodes, choose one eligible, actually-consumed new-schedule window from each of four source groups closest in time to the first group transition (tie earlier). These64 conditions are selected from teacher records/visits without model scores. Freeze them and two noise packets before any new arm quality result. Query donor TASK8192, CONTROL4096 and GRIP4096,128 requests each,384 total. If a condition exactly matches an already stored input/noise/policy identity, reuse it and debit only actual new calls. Report same-condition gripper prefix/full errors, OPEN/CLOSE threshold mistakes (g<=.2 closed, g>=.8 open as diagnostics only), event timing, XYZ/SO3, gates and active masks. Do not use diagnostic thresholds to snap executed outputs.

Rollout remains the exact existing physical client:32s,20Hz actions,8 executed points then fresh same-tick body/wrist RGB and actual TCP/opening; native100Hz physics. Preserve fullH40 non-corrective pose gate, original actuator envelope and declared limited gripper saturation, native IK hold and all controller transforms. Never install a refused reference. No planner/teacher/object pose/phase/success on the policy side; no temporal action smoothing, binary gripper hack, target attachment or oracle stop. Continue policy after first task success. NON_REALTIME remains explicit.

Main task measure is the unchanged native on-top/static/not-grasped predicate with first success in0..30s and persistence at every20Hz sample in28..30s (41 samples), and full-horizon interface validity as in PI0276. Also retain100Hz contact/state reconstruction, physics failures, drops, penetration, grasp duration, lift, transport, release and post-release losses. Do not rename the result official ManiSkill leaderboard performance or robot safety qualification.

Basic repair milestone: an arm reaches>=12/16 TRAIN persistent successes AND>=24/32 REPAIR_DEV persistent successes. These fixed counts are prospective operational targets, not universal sample-size or significance guarantees. A single successful video is progress, not this gate. Both arms remain valid candidates: if CONTROL improves as much or more, report additional training as the sufficient explanation for this comparison, not a special gripper-loss win.

## 9. Conditional fresh confirmation, same task only

If either arm reaches the basic repair milestone, automatically run this already-authorized confirmation. Otherwise record confirmation DEFERRED_BY_REGISTERED_BASIC_TASK_FAIL, archive all results and stop; do not collect new domains to conceal a failure on the original small task.

Reserve32 fresh source seeds30032..30063 under `PI0277_STACKCUBE_CONFIRM`. Check against source-task split identities before use; no new randomization domain, object, skill or camera. Until BOTH4096 checkpoints, loss/runtime/data cards and milestone evaluation are frozen, do not instantiate/render/label/query these conditions. Never tune on them or replace failed expert seeds. A namespace conflict discovered before any use is a concrete blocking fact, not permission to select alternative seeds by results.

On each confirmation initial condition, run the unchanged native joint expert, converted EEF expert, CONTROL4096 and GRIP4096:128 full episodes, including64 new learned rollouts and64 experts. New expert data are evaluation-only and never enter training/statistics. Match recorded initial physical states and paired actual model noise; expert failures remain in both learner denominators. No rerun of prior valid conditions.

Register confirm task success>=24/32 for an arm; report paired CONTROL versus GRIP gains/losses, first/persistent success and Wilson intervals. Additional GRIP effect target is net>=8/32 over CONTROL, separately from absolute task qualification. No statistical significance or improvement is assumed. If neither reaches24/32, no task-qualified candidate. If both pass and the difference is small, task feasibility is positive but unique gripper-supervision advantage is unestablished. Same objects/layout distribution means held-out instances of this task, not multi-task, language or cross-body generalization.

## 10. Budget and resource accounting

| New work | Planned maximum (including conditional confirmation) | Hard ceiling |
|---|---:|---:|
| CONTROL + GRIP formal updates |4096+4096=8192|8192 healthy updates|
| GPU timing/equivalence discarded updates |384|576 (one alternate cached route)|
| FAST CONTROL resume + GRIP canary updates |8+16=24|included in total|
| Total completed updates |8600|9048, includes alternate192 and technical256|
| Training action-head calls |count all actual forward/recompute|172032|
| Offline original panel new queries |384|within quality cap|
| New64-condition/two-noise fit panel across3 policies |384|within quality cap|
| Learned physical episodes |96+64=160|same registered conditions; technical starts below|
| Confirmation source joint/EEF episodes |32+32=64|no extra scientific seeds|
| Native compatibility fixtures |2x10s maximum|2 scientific fixture starts|
| Full32s physical episodes |224|technical allowance only|
| Total physical starts |226|234 (8 same-condition technical replacements)|
|20Hz control calls |143760|150000|
|100Hz physical steps incl.setup/reset |718800 nominal|750000|
|RGB frame pairs |143760 nominal|155000|
|New source-teacher executed controls |40960|48000|
|Quality/inference requests |13568|14336, includes all probes/technical calls|
|Quality S4 heads |54272 nominal|57344|
|CPU input-item calls incl.cache/parity/bench/formal fetch/prefetch |count actual|65536|
|Senior WBC/hardware calls |0|0|

Arithmetic: formal8192 +384 speed +8 fast resume +16 grip canary=8600; optional alternate192 and technical256 ->9048. Model calls:160*80 +384 original offline +384 fit panel=13568. Physics:(96 learned original +128 confirmation)*3200 +2000 fixture=718800; control224*640+400=143760. No independent GPU training or physics is hidden in setup, profiling, model materialization or a helper. Mark all in-flight/discarded work, not just successful counters.

No unused PI0276/preparation budgets are transferred. Existing preparation's zero-GPU restriction is not carried over to this NEW explicit authorization; this goal grants the enumerated GPU/physics work. Old budgets remain immutable. Technical repetitions can repair the same semantic runtime/recording condition, not valid task FAIL or an undesirable speed/quality score. If a hard resource limit is reached, save durable valid state and close explicitly; do not silently increase it.

## 11. Required execution and closeout

Implement a finite resume-capable `python scripts/pi0277_execution/run.py all --release <canonical-release> --resume` with stages: audit_existing, cache_inputs, cpu_bench, h200_qualify, canary, train_control, train_grip, evaluate_fixed, confirm_if_admitted, cpu_replay, archive, closeout. These entrypoints are deliverables to implement, not claims they already exist. Passing substantive gates automatically advances; no stop at CODE_READY, CPU_READY or checkpoint save asking for another routine training GO.

Required receipts: existing failure funnel and exposure audit; loss/mask/gradient source binding; actual CPU cache/input/label/visit proof; original and accelerated H200 timing traces; candidate selection and speed status; exact runtime and GRIP canaries; matched noise/consumption and loss audit; four complete Trainers with pinned HF full-file readbacks; all fixed physical/offline/fit rows and conditional confirmation state; per-event gripper/object/IK metrics; raw/install/achieved arrays; old-policy exact reuse map; actual all-stage counters.

Reconstruct object predicates from saved arrays and new gripper losses/metrics from saved predictions where possible. Disclose shared native primitives; a second invocation of the same scorer is not a fully independent implementation. Save fixed videos for REPAIR_DEV30000..30003 and first4 TRAIN diagnostic seeds under BOTH arms, including failures; if confirmation runs also first4 confirmation seeds under BOTH arms. Encode only saved frames, watermarked SOURCE_PANDA/NON_REALTIME/model identity; no rerender for better-looking outcomes.

Write `FINAL_RESEARCH_RETURN.md`, `CLOSEOUT.json`, `SOURCE_INDEX.json`, `PERFORMANCE_VERDICT.json`, paired tables and per-case failure classification. Archive new evidence/data derivatives with actual license/provenance and fixed-file/member readback. Never delete sole recovery or unique physical records before durable verification. Publish and read back Issue #11 terminal. Report old PI0276 archival state separately if it has not yet closed.

Terminal: **PI0277_SINGLE_TASK_REPAIR_COMPLETE — STOP_FOR_NEXT_DECISION**, with separately reported execution integrity, measured acceleration, task feasibility, gripper-intervention effect and archival completeness; or a concrete bounded semantic/runtime/resource failure with exact deferred stages. No automatic multi-scene/multi-task training even on success. `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`; an explicitly named SOURCE_TASK_RESEARCH_CANDIDATE may be recorded only for the achieved limited qualification, not WBC deployment.

## 12. Interpretation and next-step discipline

If gripper fit improves but no sustained grasp appears, inspect approach geometry and timing before attributing failure to domain coverage. If grasp appears but carrying fails, inspect pose/reference versus measured object and IK behavior. If carry/on-top occurs but release fails, inspect commanded/measured opening and post-release actions. If TRAIN succeeds but held-out confirmation fails, then consider coverage/recovery under a new decision. If both matched arms improve similarly, accept that more exposure may explain the advance. If generated gripper loss harms pose or does not help, retain that negative result; no hidden binary controller, coefficient search, history addition or teacher takeover.

This experiment fixes one observed supervision asymmetry while keeping the old continuous task/action architecture. It does not settle initialization quality, visual grounding, native-flow objective interactions, all state-distribution shifts or controller sensitivity. Existing expert31/32 does not guarantee every learned reference is physically feasible. Missing gripper auxiliary supervision is a defensible hypothesis, not proof of a unique cause.

Review evidence: PI0276 fixed result/PHYSICAL_VERDICT/GRIPPER_PHASE_SUMMARY/mechanism note, sampled PHYSICAL144 rows, train.py/infer.py/evaluate.py/report.py, inherited PI0273 train and PI0269 temporal code; post-PI0276 PREPARATION_RETURN and cpu_dataset.py. The reviewer did not run remote physics, download/replay all raw outputs, launch GPU profiling or watch all videos. Native/private imported helpers not fully materialized in the reviewer environment must be source-bound by the executing owner. Public PyTorch data-loader/tuning documentation informs performance mechanisms only; the installed sealed version is authoritative. Historical provided notes motivate TRAIN-versus-generalization diagnosis, full policy identity and object-level success; their old selections, authorities and WBC-dependent data suggestions do not supersede the current charter.

Startup: read the new canonical/release, adopt old archive processes safely, perform saved-array failure audit and actual H200 cache/prefetch qualification, then run matched CONTROL4096/GRIP4096 from TASK8192 and fixed single-task physical evaluation; automatically confirm only when its preregistered basic-task gate passes. No multi-task/domain expansion, new TRAIN data, WBC/hardware, quality retries or budget resets; selected/deployment NONE.

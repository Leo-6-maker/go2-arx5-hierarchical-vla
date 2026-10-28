# Post-PI0276: faster task training and multi-scene collection preparation

Date: 2026-10-06 SGT.
Decision: **POST_PI0276_ACCELERATION_AND_MULTISCENE_PREP_V1**.
Status: **GO_CPU_PREPARATION_ONLY — not a new model-training or physics GO**.
Reviewed project SHA: `be3b44f09ff107d37f731afe74f31f2c82a362a2` (initial source review at `73d75bc9d0c8c35488a30d04800da74ddc475b06`; the intervening commit adds only TRAINING_LOG_SUMMARY.json).
Current science remains PI0276 canonical #6000169298, release `57fd8a9c53ce241c05a8d5b351034bed7ba56b14`.
Research charter remains #5994405991 / `ed398fee17151cfec47864fed9d48ce149b28d61`.

## 0. Purpose and explicit execution boundary

The user asks the agent to prepare future training acceleration, multi-scene construction and data collection while the current experiment finishes. Deliver working preparation utilities, CPU tests, measured CPU pipeline evidence where resources permit, scene specifications and a bounded successor draft—not just a prose proposal.

This decision authorizes read-only audit of already permitted records, isolated CPU code, CPU preprocessing benchmarks using existing TRAIN records, static scene/config construction, collection/dataset code and mock/CPU tests. It authorizes **zero new neural forwards/backwards/optimizer updates, zero simulator constructions/resets/steps, zero rendering, zero teacher-planner execution, zero new physical episodes and zero WBC/hardware calls**. Calling gym.make merely to inspect an environment can initialize physics and is outside this preparation scope. Static scene-card generation is not data collection or a claim of physical feasibility.

Do not alter PI0276 source locks, task statistics, data membership, weights, training semantics, evaluation, budgets or continuing archival work. TASK8192 has now completed; do not restart it to benchmark workers. The final evaluation and archival closeout remain the main owner's priority. Do not wait for its quality PASS to prepare reusable CPU tools. New GPU qualification and new scene execution require a separate numbered science/engineering authorization with its own budget.

One existing Codex main owner adopts actual work. Optional CPU-only helpers may own disjoint preparation directories; they are not additional coordinators and their signatures are not mandatory gates. No competing auto-goal, monitoring service, new long-lived branch, PR77 dependency, broad cleanup or forced merge. Publish preparation results to Issue #11 with fixed commits and readback. The preparation canonical does NOT supersede the current PI0276 science canonical.

## 1. Actual evidence and why preparation is justified

At the reviewed branch, POSTTRAINING_MAIN_TRAINING_AUDIT reports8192 formal updates,16384 actual microbatches,8208 total completed updates including16 canary updates,147768 training head calls and zero technical completed updates. Training integrity is PASS. The source job exited1 after training/save because the inherited read-only auditor did not map the exclusive range bound5121 to8193; a separate read-only repair closed the audit without retraining. Final policy quality is not established by that repair or by the successful checkpoint.

The source/data path already has DEV8/8, TRAIN93/96 and VAL15/16 successful native/EEF results,55,893 TRAIN windows and9,015 VAL windows. Use these as existing evidence, not permission to open PI0276 TEST for preparation. Current TRAIN inputs are93 independent episodes, not55,893 independent demonstrations.

Actual train.py hardcodes num_workers=0, has an episode LRU cache(maxsize96), incremental log reads and actual-consumption journaling. It re-runs the CPU processor/hash checks per item. Pin-memory is inherited from actual Trainer args; the final resolved value must be read, not guessed from defaults. The code also converts GPU-derived scalars to Python floats in generated-gripper, native-gripper and gradient hooks. These are candidate synchronization sites, NOT a measured dominant bottleneck.

TRAINING_LOG_SUMMARY is descriptive: first/last512-microbatch mean total loss1.76484/0.161348, generated gripper command-fraction MAE0.426311/0.298909. Inputs differ across those windows. This is neither a fixed-condition quality comparison nor proof of grasp timing. Prepare gripper/phase diagnostics; do not infer that more scenes or faster training will fix the policy.

The existing source_worker is tightly bound to StackCube, including cubeA/cubeB evidence fields and global physics-step instrumentation. Do not import it in a CPU preparation process and assume there are no side effects. Reuse its pure checked conversions in an isolated module; instantiate source simulators only in a subsequently authorized runtime.

## 2. Ownership, files and order

Use `scripts/post_pi0276_preparation/` and `reports/preparation/post_pi0276/`, separate from live PI0276 roots. The sole owner records current process/resource occupancy once at adoption and before any heavy CPU benchmark; no extra periodic monitor.

| Work package | Main deliverables | What can execute now |
|---|---|---|
| A: acceleration | saved-log performance inventory, pure CPU dataset adapter, loader benchmark, consumption/resume tests, future GPU qualification draft | existing-file reads and bounded CPU item processing only |
| B: scenes | pinned task registry, factorized scene builder/configs, split/group manifest, static visibility/geometry checks | schema/array/math/AST tests, no renderer or simulator |
| C: collection readiness | command/observation/evidence schema, finite runner dry-run, failure accounting, storage estimates, next-experiment draft | mock executor and saved-data validation, no new teacher or physics |

Start A0/B0 from existing files without waiting for the full model evaluation. A CPU benchmark must run on idle, legitimately allocated CPUs, not on a busy evaluation host or an HPC login node. Helpers must not compete with current graphics, archive I/O or physical-evaluation CPU cores. If no suitable allocation exists, complete code/config/mock evidence and mark timing NOT_MEASURED_RESOURCE, without obstructing PI0276.

## 3. A0: establish the bottleneck before changing worker count

Read actual Trainer args and environment lock, checkpoint/log timestamps, consumed-order journals, native/S4 execution accounting and already saved resource samples. Split known durations into setup/canary, data staging, steady training, checkpoint I/O, audit and network archive. Do not divide a full PBS walltime by updates and call it pure GPU compute. Do not infer GPU utilization from qstat CPUpercent. Missing loader-wait/H2D/GPU timelines must remain NOT_OBSERVED.

Create a future profiler insertion map for data dequeue, CPU processor, hash verification, H2D, native forward, S4 generation/backward, scalar extraction/log writes, optimizer and checkpoint. No new model profiler run is authorized here. Future CUDA timings must use appropriate device events/profiler or synchronized interval endpoints—not unsynchronized Python wall time attributed to kernels, nor a synchronization inserted into every operator.

Prioritize: (1) existing deterministic CPU input caching/prefetch; (2) avoid duplicated NPZ decoding and metadata copies; (3) typed run configuration instead of scattered numeric/path rewriting; (4) design-only reduction of unnecessary metric synchronization. Keep essential numerical/integrity checks and reconstructible scientific journals. No removal of audit checks to make a benchmark look faster.

`float(cuda_tensor)`/item/cpu transfers may synchronize; workers do not eliminate main-process device synchronization. Consolidating detached metric transfers can be proposed, but modifying their event order or persistence must have separately registered equivalence tests. Do not retain computation graphs in an asynchronous logging queue. No asynchronous checkpoint writer in this preparation.

## 4. A1: prepare ordered CPU-only loading

Extract a top-level, picklable map-style dataset/collator whose import path does not construct the model, initialize CUDA, import a running simulator or install the old Trainer's global hooks. Processor objects must be lazily constructed per worker from fixed local assets. Do not put a lambda, captured running Trainer or open NPZ handle into worker initialization.

Candidate W0: original worker0 behavior under a pure CPU wrapper. Candidate W2:2 workers, prefetch_factor2, persistent_workers=true, ordered fixed sampler and spawn context. Candidate W4:4 workers, otherwise identical, is a conditional CPU-only probe, not a default production setting. Freeze actual Python/PyTorch/numpy/processor/BLAS versions. If installed DataLoader lacks in_order, use its ordered default and test order; do not upgrade PyTorch to gain an argument. Never set in_order=false.

Keep batch1, accumulation2, sample sequence, labels, masks, normalization and all scientific randomness unchanged. Worker seeds come from a separate loader generator; workers may not draw model epsilon/tau/noise or touch CUDA. If a future transform is randomized, its randomness must be indexed by immutable sample/visit identity, not task completion timing or worker assignment. Current deterministic input hashes remain the reference.

The main process alone commits consumption. Store next-unconsumed micro-index only at a completed optimizer boundary; for the current layout it is2*global_step. Prefetched but unconsumed samples do not advance it. On resume discard worker queues, reconstruct the fixed sampler at that index and reproduce all inputs. Test source-end boundaries and restarts at updates0,4,128,4096 and8191; never claim a CPU cursor test proves optimizer/CUDA resume.

`persistent_workers` avoids recreation between iterators/epochs; the present training uses a largely single explicit stream, so it is not by itself an expected major speedup. pin_memory/non_blocking H2D are FUTURE GPU options to verify under real resolved args. CPU benchmarks keep pin_memory=false and CUDA hidden; do not claim they measured DMA or pinned-memory benefit.

Each worker may replicate dataset metadata and a private LRU cache. The existing93*640*96*96*3*2 uint8 RGB bank is3,291,217,920 bytes (~3.065 GiB) before states, metadata, processor buffers or model memory. This is not a prediction of peak RSS. Track aggregate process-tree RSS and PSS where supported, shared memory and page faults. Do not multiply the whole parent GPU-model allocation by workers and call that actual CPU memory.

Prepare read-only contiguous NPY/memmap storage for existing decoded arrays and compact metadata as an optional candidate. NPZ itself is not automatically memory-mappable. Preserve per-episode pixels/arrays exactly, cache key = data hash + processor/statistics/resize/config hash, and verify against existing input/label proofs. Do not cache frozen-backbone neural features or change graph/mode/RNG behavior. Do not materialize every H40 window as forty duplicate RGB arrays.

## 5. A2: bounded CPU benchmark and admission for a FUTURE runtime

Use ONLY the first2048 entries of PI0276's sealed TRAIN microbatch schedule; no VAL or TEST. Preserve repeated visits exactly. For each enabled config perform at most3 fresh-process passes, each64 warmup item calls followed by2048 measured item calls. Maximum base work for W0/W2/W4 is19,008 item calls. Additional queue-drain, extraction-parity and cursor checks share a hard cap32,768 actual item calls; count prefetched/discarded work. Do not reset the cap after an implementation error.

Record startup/first-batch time, per-item/dequeue median andp95, total samples/s, processor/hash/IO costs where measurable, raw bytes, process/thread count and peak memory. Fresh process is not cold disk: do not drop shared OS caches. Alternate test order W0-W2 / W2-W0 / W0-W2; optional W4 trials use the same source positions and host conditions. Keep comparable CPU allocations and no oversubscribed BLAS/tokenizer threads. Any thread policy change must still pass all old input hashes.

W4 is allowed only if W2 is correct, improves median CPU pipeline throughput by>=15% versus W0 and fits below half of the preparation memory ceiling, and four worker CPUs plus owner are legitimately free. This is an engineering screening threshold, not a paper metric. If W2 loses or yields only noise-level gains, recommend W0 or profile main-thread overhead rather than forcing more workers. Stop and preserve evidence for input/order/RNG mismatch, memory overrun, deadlock or resource contention.

A recommendation requires exact condition/label/mask/visit identity, no parent scientific-RNG consumption, correct interrupted-consumption restart, bounded memory and repeated CPU timing evidence. It is named CPU_LOADER_CANDIDATE, not TRAIN_RUNTIME_QUALIFIED or H200_SPEEDUP_PASS.

Draft, but DO NOT RUN, the next GPU qualification: matched worker0-continuous8, candidate-continuous8, candidate4-save-freshresume4 (24 completed optimizer updates in that draft), all with identical initial model/optimizer/scheduler, input schedule and actual noise; compare trajectory/gradient/model/optimizer/RNG against original requirements. Its real throughput timing and all model/head budgets must be authorized separately. Same effective batch is not enough to authorize batch2/accum1. No TF32, compile, fused optimizer, fewer S4 steps, DDP, loss/statistics changes or backbone unfreezing here.

## 6. B0: one platform; distinguish scene variation from new skills

Keep ManiSkill pinned to `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` and reuse the recorded source dependencies/assets from PI0276. Do not install LIBERO/RoboTwin in parallel. Existing single-skill physical pipeline is the preparation anchor; it is not proof of multi-task or cross-body success.

First wave builds three StackCube scene domains with the same native cube geometry, mass/friction/gravity, source Panda/EEF controllers, finger mapping, TCP,100/20Hz frequencies and success predicate. Preserve the existing extended-horizon policy as a draft default, explicitly distinct from an official leaderboard result. Do not modify PI0276's evaluator.

| Domain | Planned change | Must remain fixed |
|---|---|---|
| D0 LAYOUT | new base layout seeds in the native placement/yaw range | original appearance and both cameras |
| D1 APPEARANCE | same base layouts; table visible material/roughness and predeclared illumination gain | red/green task identity, geometry, cameras and physics |
| D2 CAMERA | same base layouts; external camera pose/FOV only | native visible materials, wrist mount/intrinsics, physics and task |

Draft D1 proposals: neutral table RGB presets(.35,.35,.35),(.55,.55,.55),(.70,.65,.55); roughness .5 or.8; illumination gain .8,1.0 or1.2. Task-object colors are not permuted; no target-dependent masks, goal overlays or marker colors. Under the existing minimal shader, a changed environment map may be invisible. Prepare a pixel-difference/visibility QA for later rendered DEV; do not silently switch shader or claim an unrendered material field causes visible diversity.

Draft D2 proposals: camera-eye world offsets within +/- .03m per axis relative to the original external camera, yaw/pitch offsets within +/-5deg, FOV80/90/100deg. Freeze a stable local camera convention in the card; no actual hidden target is used to aim the camera. Keep wrist geometry native in this first wave. Build FOV at reconfiguration, not by an unsupported runtime field mutation. These are engineering candidate ranges, NOT measured coverage or proven sim-to-real bounds. Static frustum/transform checks do not replace rendered visibility checks.

Do not apply D1 and D2 together in the first collected matrix. Do not simultaneously add different robot masses, grippers, moving bases, distractor physics, new object geometry and new tasks. Moving-camera, object-size/shape, shelf and door recipes belong to a separate backlog with explicit physical/source/sensor prerequisites, not extra hidden factors in D0-D2.

## 7. B1: concrete scene cards and leakage-safe splits

Prepare static base groups: DEV50000..50007(8), TRAIN51000..51095(96), VAL52000..52015(16), TEST53000..53031(32). Each base group has D0,D1,D2:152 groups /456 candidate scene cards, with96 independent TRAIN base groups /288 TRAIN variants. These are proposal records, not collected episodes. Confirm no overlap with any existing protected project split; on namespace collision report it and revise the draft before future authorization, never after model results.

All correlated variants of a base layout remain in the SAME split. Separate immutable RNG namespaces for placement, appearance, cameras, source planning and later policy noise; changing a visual seed must not accidentally change object geometry. Record desired parameters, actual resolved values, source versions, generator hash, parent group and eventual initial-state hashes. Same numeric seed alone is not an initial-state proof.

TEST is metadata/ID reservation only in this preparation: do not instantiate, render, label, predict or inspect TEST scenes. A future unseen-appearance or combined-factor split needs its own frozen specification; the present D0-D2 design measures new layouts under declared domains, not novel object categories or new skills.

Generate JSON/YAML cards and a pure-Python dry-run table, detect split leaks/duplicate group identities and compute static transform/dimension constraints. Initial overlap or frustum invalidity is a predeclared scene-generation issue, not permission to select by expert/model success later. Preserve all proposal rejections. During future native execution verify physical state/camera/RGB consistency and dataset licenses.

Second task audit candidate: PlaceSphere-v1 at the same pinned source. It genuinely checks object-on-bin/static/not-grasped, but its supported list is panda/fetch, NOT panda_wristcam; its observations include bin_pos and is_grasped even beyond full state mode. Prepare an explicit camera/robot adaptation plan, planner-source inventory and policy-input allowlist. No raw obs-dict passthrough, no pretending wristcam or a working teacher is already qualified. Shelf insertion/door opening stay design-only. A new task is not merely a new seed, and a paraphrase is not a new skill.

## 8. C0: prepare a finite collection runner, not an unbounded data campaign

Implement future commands in a new namespace with default --dry-run; --execute under THIS preparation release must reject before any native import/gym.make. Build a fake-executor test that proves all native, teacher, render and neural launch paths are forbidden at this authority level. The API names are deliverables to implement, not a claim they already exist.

Source data flow remains: fixed scripted joint commands -> independent FK -> saved commanded EEF -> matched physical EEF replay -> fresh actual RGB/TCP/opening plus future commanded EEF/gripper labels. No WBC call/filter and no target-robot demonstration. A source Panda success filter can be explicitly disclosed for positive BC data; failures keep their fixed denominator and evidence. Do not collect until a target number of successes is reached.

Reuse verified pure transformations: world/root/TCP, source absolute XYZ Euler, quaternion order, project row-rot6d, native command fraction versus measured pad opening. Source command and achieved state remain distinct. Arbitrary default env extras such as object positions, teacher phase, reward, contact-derived grasp and success cannot enter the learner.

For visual variants, joint plans may only be reused in a later authorized run if starting physics, controls and task are unchanged and proved; fresh observations must still come from the appropriate variant and actual EEF executor. Do not reuse stale images or run a hidden teacher to fix model trajectories. No attachment/pose-setter shortcut after initialization.

Prepare schemas for episode, parent scene, frame timestamp, command timestamp, raw/reference/installed/achieved TCP, opening command/measurement, instruction, stage-for-analysis, contact and native object predicates. Keep phases evidence-side. Default H40 remains40 genuine future commands at .05s, no fabricated suffixes. First split complete groups/episodes, then cut windows. Retain full-H40 gate-before/after-codec records and all exclusions.

Statistics options must be decided prospectively: for a matched nominal-versus-diverse study with identical geometry/labels, share one fixed processor and a matched update schedule; only visual data varies. Across new tasks or geometric ranges, a new TRAIN-only statistics identity may be needed. Do not recompute PI0276 stats in place or attach its old scores to new normalization. Multi-task schedule proposals should balance task then phase, not blindly concatenate windows so long terminal episodes dominate.

Prepare future collection health controls: registered native and EEF expert pairs, always-open-gripper controls, original predicates plus fixed persistence, same starting-state proof, invalid/numerical/plan/bridge/source-quality categories, timeouts and bounded technical retries. No quality-driven seed replacement. Reuse native scoring semantics; a new skill's metrics cannot inherit StackCube cubeA/cubeB names or thresholds blindly.

A minimal FUTURE pilot draft is8 DEV groups x3 domains x2(native+EEF)=48 full32s episodes, before any production expansion. This corresponds to30,720 control calls and153,600 physics ticks, before setup/reset/negative controls. These counts are planning estimates and authorize ZERO starts now. The next decision must add exact setup/negative/retry/renderer budgets rather than pretending48 is the complete cost. TRAIN/VAL production and TEST evaluation stay locked pending that separate decision.

Storage estimator uses existing actual files/frames and compression statistics, reports command/observations/contact/media overhead separately and sizes projected TRAIN/VAL/domain banks. Keep one array bank plus window indices, not repeated RGB perH40. No full redownload of the existing28.7GB archive or historical weights to perform this estimate.

## 9. Required artifacts and acceptance

Produce these (or an explicitly mapped equivalent tree):
- `perf_audit.py`, `cpu_dataset.py`, `bench_loader.py`, `test_consumption_resume.py`;
- `scene_specs.py`, `validate_scene_specs.py`, `collection_dryrun.py`, `validate_episode_schema.py`;
- `PERFORMANCE_BASELINE.json`, `CPU_BENCHMARK.csv/json`, `CPU_ORDER_HASH_REPORT.json`, `ACCELERATION_PROPOSAL.md`;
- `SCENE_CARDS.jsonl`, `SPLIT_GROUP_MANIFEST.json`, `TASK_REGISTRY.json`, `DATA_SCHEMA.json`, `STORAGE_ESTIMATE.json`;
- `FUTURE_COLLECTION_AND_LEARNING_DRAFT.md`, `PREPARATION_RETURN.md`, `PREPARATION_CLOSEOUT.json` and actual zero-native/model counters.

Each planned runtime file must have AST/import-side-effect tests, typed input checks, clear paths and error receipts. Replace future run-specific constants/paths with a small typed configuration binding, including inclusive/exclusive update limits, task statistics and source IDs. Reuse qualified math via source-bound imports; do not blindly replace every occurrence of5120/5121 orROT strings, and do not use this as an excuse to rewrite the whole historical trainer.

Testing statuses are explicit: PLAN_ONLY, CODE_WRITTEN, CPU_STATIC_PASS, CPU_PIPELINE_MEASURED, NATIVE_NOT_RUN, GPU_EQUIVALENCE_NOT_RUN. CPU_LOADER_CANDIDATE and SCENE_SPEC_READY are not DATA_READY, quality PASS or a deployable policy. Missing assets/timing can be PARTIAL with precise blocker rather than fabricated success. CPU correctness is mandatory for a recommendation; actual native visibility/source success and model qualification are future gates.

Main owner reviews separate helper outputs, integrates only the coherent preparation files, writes a fixed-commit Issue11 return and reads it back. No helper approval wait if the owner can perform the checks. No change to selected_checkpoint=NONE or deployment_selected_checkpoint=NONE.

## 10. Bounded preparation resources

| Work | This decision's hard bound |
|---|---:|
| New model/gradient/optimizer calls |0|
| New teacher-planner/simulator/reset/physics/render calls |0|
| New target-WBC/hardware calls |0|
| CPU dataset item calls incl. prefetch/discard/tests |32,768|
| CPU worker candidates |0,2; conditional4 only|
| Benchmark repetitions |3/config maximum|
| Scene specification cards |456, including reserved metadata-only TEST|
| Heavy CPU preparation |<=8 allocated CPUs, aggregate<=16 CPU-core-hours|
| Peak preparation process-tree RAM |<=64GiB AND within actual allocation; stop earlier on host pressure|
| New local derived artifacts/cache |<=20GiB; no unique source deletion|
| New network data |source/config/docs only <=256MiB; reuse task assets/data, no new weights/large datasets|

Use serial heavy benchmarks; never occupy cores/IO needed by the current evaluation. Do not request an extra H200 job for a CPU benchmark. Legal CPU compute can use existing allowed allocations; no ad hoc stress tests on login nodes. Preserve partial counts; budget is not replenished by reconnecting or spawning another helper.

## 11. How PI0276 results choose the next scientific action

Do not choose a future initializer or automatically start a multi-scene training run before fixed PI0276 results.
- If TRAIN and TEST both fail, keep scene tools ready but first diagnose the observed task-fitting/command/phase failure on existing data; do not assume coverage is the cause.
- If TRAIN is strong and TEST weak, prioritize a matched-budget nominal-repeat versus scene-diversity comparison. Same optimizer count, initializer, action semantics and processor when labels/geometry permit.
- If first success improves but persistence/release fails, inspect gripper event timing and actual rollouts before adding unrelated new scenes.
- If the source task is reliably learned, choose controlled scene expansion or the next single object task, retaining old-task evaluation. Frozen WBC composition remains a separate explicitly scoped validation.

A source-scripted trajectory is not a closed-loop learner's recovery teacher; preparing hooks does not authorize DAgger or generate counterfactual labels after rollout failure. Shared-weight multi-task, new layouts, unseen assets, language control and cross-body transfer must be separate reported claims.

Preparation terminal: **POST_PI0276_CPU_PREPARATION_COMPLETE — STOP_BEFORE_GPU_QUALIFICATION_OR_NEW_COLLECTION**, or documented PARTIAL with completed subpackages. No new scientific GO is implied by readiness.

## 12. Sources and evidence boundaries

Actual project sources reviewed at the pinned project lineage: `scripts/pi0276_execution/train.py`, `source_worker.py`, `reports/PI0276/execution/POSTTRAINING_MAIN_TRAINING_AUDIT.json` (checkpoint inventory and status/accounting), `POSTTRAINING_REPAIR.json`, `TRAINING_LOG_SUMMARY.json`, and current Issue11 comments #6010138754/#6011159275. The reviewer read source/receipts and computed plan arithmetic; no SSH, live resource profiling, model, data-loader or native simulation execution was performed by the reviewer.

Official implementation/docs, for mechanism guidance only; installed pinned versions remain authoritative:
- https://docs.pytorch.org/docs/stable/data.html (workers/order/randomness/memory/persistence; inspect actual installed API).
- https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html (async input and device synchronization).
- https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html (camera construction/reset limits and minimal-shader caveat).
- `mani-skill/ManiSkill@62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/envs/tasks/tabletop/place_sphere.py` (supported robots, native predicate and privileged observation extras).

Provided historical notes support identity of weights+processor/statistics/mapping/decode, episode-level splits and object-level outcomes. Their older status/authority and any WBC-success prerequisite for collecting tasks do not override the current charter. In particular, do not silently reinstate target-WBC filtering when generalizing the dataset.

Startup: read this preparation canonical/release and PI0276 authority; adopt actual continuing jobs without interruption. Prepare CPU-only acceleration evidence, factorized scene specs and a finite data runner with dry-run guard. Preserve original TRAIN/TEST boundaries and all model identities. Publish code/tests/readiness and bounded future qualification/collection proposals; zero new model/physics/WBC calls now, selected/deployment NONE.

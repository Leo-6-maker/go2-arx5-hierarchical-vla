# PI0285 R3 — parallel Codex work packages, one first-policy critical path

Authority: **PI0285_R3_PARALLEL_EXECUTION_AND_POLICY_DELIVERY_GO_V1**.
Parent: PI0285 R2 canonical #6065031954, release `97773a036ebaffe59285bd2ecbee0c155f12822b`, protocol `docs/protocols/PI0285_R2_1000_SCOPED_ADMISSION_AND_POLICY_DELIVERY_V1.md`; R1 task definitions at release `7e840324c71aef47b401bd861b22710774334c97`.
Review anchor: `ba57d6833ba494b6b14eaa9d9eea94a9405c50ee`, existing branch `pi0267/executor-decoupled-recovery`.
Read actual owner evidence #6065801898, 2026-10-09 01:50:33 SGT, and the newer committed fixed-DIAG CPU bias preparation. This is repository evidence, not reviewer live SSH/PBS/GPU access.
This file and `reports/PI0285/design/R3_WORKPACKAGES_AND_LIMITS.json` activate together only after fixed-release Issue11 publication and readback.

## 1. What changes, and what does not

The user requests more useful Codex tasks. R3 expands parallel engineering, saved-array diagnosis, conditional recovery readiness and narrow no-update model probes. It does NOT add a third robot task or competing optimizer arms. Main outcome remains independently confirmed T1 pick/place, protected before any shared T1/T2 optimization. All unmodified R2 gates, task metrics, supervised fidelity requirements, frozen statistics, model lineage, parameter scope, optimizer/precision recipe and finite resource caps remain binding. R3 takes precedence only for worker allocation, its specific additional descriptive inference permissions and the optional preparation allowance.

**Hard deadline remains 2026-10-09 10:00 Asia/Singapore (02:00 UTC).** No clock reset at adoption, job retry or stage transition. Stop starting optimizer updates by08:30, all neural/teacher/physics work by09:45, deliver by10:00; stop earlier to preserve the R2 confirmation reserve. If adopted after a cutoff, the corresponding work is not launched. No automatic date rollover.

At the reviewed snapshot the R2 scoped admission has actually been written; all960 admitted array members /10,391,162,880 bytes and input/processor identities were read back at Hopper. Original all-split codec FAIL and two DEV clipping cases remain. Job655259 completed zero optimizer updates and failed after one native forward/backward microbatch (four chunks): passive audit JSON serialization mishandled native BatchFeature. The narrow Mapping-aware audit fix has CPU evidence. One replacement655273 was Q at the last owner snapshot; its CURRENT status must be read by the owner. No completed carry or learned-policy quality PASS is inferred. Do not repeat admission/data transfer or resurrect the resolved scope question.

All WBC/Go2/target simulation/target-policy/hardware calls remain0. No target task teleoperation, target-WBC filtering of training data, hidden phase/object truth, scripted model finishing or success-based stopping. selected/deployment=NONE; a positive source qualification is reported separately.

## 2. Ownership and finite parallelism

Keep the existing single main owner. Raise optional subagent limit from3 to **5 workers**, not five additional owners. Suggested allocation: workerA TRAIN/RUNTIME; B EVAL; C DATA/REPRESENTATION; D RECOVERY; E SHARED/DELIVERY. The eight work packages below can share these workers. Fewer workers are acceptable when orchestration or CPU contention would cost more than it saves.

One accepted training lease; at most one active model-evaluation GPU service per actual GPU. Multiple agents do not imply multiple available GPUs. CPU-only optional analyses: at most two concurrent processes, each at most4 computational threads unless the owner allocates less. All compete within the existing256 CPU-core-hour cap. No unrelated project process, network interface/service, scheduler identity or credential is modified.

Each work package records its owner, state, dependency hashes, immutable input/output directory, intended calls, consumed counters and result. States include READY, RUNNING, COMPLETE, BLOCKED_EVIDENCE, DEFERRED_PRIORITY and NOT_TRIGGERED. A prepared module or CPU PASS is never a training/physics PASS. Proposed output paths below are requirements to implement, not existing evidence.

Use disjoint paths under `reports/PI0285/r3/WP*/` and isolated prospective helper modules. Only the owner integrates code or scientific ledger changes. Never restage/edit a healthy imported runtime. A fix is adopted at a stopped-process or saved-update boundary with its identity recorded. Existing runtime hashes remain valid for the work they actually produced. No new permanent branches or PR77 pointer is needed. A subagent sign-off is not a prerequisite for the main owner to advance a satisfied gate.

## 3. Eight concrete work packages

### WP1 — training runtime reliability and resource adoption (P0, workerA)

First refresh actual process/job/lease state, completed and attempted work, and remaining budget. If655273 has started healthily, adopt it unchanged. If queued for20min without allocation, apply the existing R2 fallback ONLY when an existing legal Corelab GPU can fit the identical scope: withdraw the nonrunning queue request, verify no live duplicate, bind one fallback. No speculative second qsub. A runtime message is not proof of completed training.

Reuse R2_SCOPED_DATA_ADMISSION, destination readback and two-worker proof by exact hashes. CPU regression tests must exercise genuine native BatchFeature/Mapping, nested tensors/lists/scalars, tensor hashes, JSON output, temporary-file atomic rename, checkpoint metadata and the exact fixed instruction padding. Unknown container types fail explicitly; never silently stringify tensors and lose input identity. Add interruption fixtures for incomplete export manifests and mismatched processor hashes using disposable CPU files, not a kill of healthy training. Reuse already valid tests rather than rerunning every test at each stage.

Complete the single real continuous8 vs fresh4+restore4 canary required by R2; do not duplicate it per export or for unchanged carry semantics. Every attempted microbatch and completed canary/rollback update is charged;655259's attempted work is not erased. Do not continue scientific updates after an essential audit/accounting failure without a recoverable boundary.

Deliver `WP1/RUNTIME_ADOPTION.json`, `WP1/PASSIVE_AUDIT_REGRESSION.json` and the real job/carry/update receipts. Acceptance is actual running/complete native optimization with the declared input/scope/noise and recovery proof, not qsub acceptance alone. Then execute PLACE4096 as R2; no new LR/loss/batch search. Main training must not wait for WP4/WP7 descriptive research.

### WP2 — checkpoint-to-physics feedback pipeline (P0, workerB)

Prepare and CPU-test the exact model-only transfer/reconstruction, request schema, episode launcher and saved-array scorer BEFORE the first eligible snapshot. Use verified data/processor locally; optimizer remains on training host. Bounded in-goal request handling is allowed, not a new recurring automation/service.

A snapshot READY record contains full model/changed-tensor/base identities, processor/statistics/mapping/K4, completed update, schema and file sizes. Read only immutable complete snapshots; reject partial/mixed files. Test wrong SHA, stale request/result ID, task-language mismatch, missing arrays and wrong clock as CPU negatives. One source consumer owns the physics ledger. Reuse an existing identical query result by input/model/noise identity; do not use this as permission to suppress a quality failure or resample noise.

Keep R2's eligible PLACE screens1024/2048/4096 plus one authorized branch endpoint. Queue at most two unconsumed exports; apply backpressure to optional export/diagnostics, not overwrite a valid result. The optional1024 screen can be skipped prospectively for resource backlog as R2 allows, never because a score looks poor. No arbitrary new model selection checkpoints. DEV/CONFIRM priority exceeds supplementary analyses. Direct local transfer, not HF upload/download as feedback transport.

For every completed original episode, automatically emit `WP2/EPISODE_FUNNEL.json` and `WP2/CHECKPOINT_RETURN.json`: complete/aborted status, grasp/lift/transfer/target/release/withdraw/hold, first success and first loss, raw-to-installed and same-control achieved errors, gate/IK events, final individual predicates and actual video references. Report task metrics separately from codec fidelity, offline fit and correctness PASS. The owner can act on validated task arrays without waiting for polished videos or bulk archives; all raw records and candidate integrity checks remain required.

### WP3 — saved-array temporal and action-contract diagnosis (P1, workerC/B)

Reuse the now-prepared DIAG192 identities and raw-versus-codec CPU bias instead of rebuilding them. Trace RGB timestamp -> actual StateV1 -> commanded EEF/gripper -> installation -> achieved TCP/object. Specifically test first8 execution indexing, pre-command versus post-command capture, gripper measured/command polarity and phase_before/issued-command/phase_after at transition boundaries. Do not silently relabel the sealed data or re-time commands.

Use existing source physics/contact arrays for all teacher admission and each new model episode. Inspect enough fixed approach/close/carry/release/withdraw/hold cases to locate the FIRST divergence and whether a task loss precedes or follows IK/gate/contact events. Consecutive correlation is not proof of a unique cause. No new model/physics queries in this package and no unnecessary full76928-window roundtrip repeat. Two DEV285200 target clipping cases remain visible, raw labels remain evaluation truth, and original missing aggregate TRAIN maxima remain UNOBSERVED unless genuinely recomputed.

Deliver `WP3/TEMPORAL_CONTRACT.json` and `WP3/FAILURE_ATTRIBUTION.csv`; list confirmed faults, ruled-out hypotheses, unknowns and targeted next checks. A proven runtime input bug can invoke already authorized same-semantics engineering repair, not a new data recipe or scoring exception.

### WP4 — bounded language/goal response and pretraining probes (P2, workerE/C; NEW inference scope)

This is descriptive only, subordinate to T1 delivery, not an extra quality-selection panel. Use the first8 TRAIN layouts285100..285107. For each use the T1 teacher's existing real observation at the first phase2 (carry/lower) anchor; freeze each UID, timestamp, raw image/state hash and two complete native BF16 noise packets before any new probe result. Feed the SAME genuine observation with each of the two valid existing task instructions. Only text changes; no object truth, task ID, fabricated state/image/history or teacher label is supplied to the model. Reuse raw arrays; no rendering/physics.

Each model point uses8 observations x2 instructions x2 noises=32 calls, K4. At most3 DISTINCT points: (a) the earliest already produced eligible PLACE checkpoint; (b) the actually frozen T1 candidate, or if none the final available T1 endpoint; (c) the final available SHARED endpoint if that stage is admitted. If a point coincides, reuse results, do not fill the slot with another checkpoint. Maximum96 new calls/384 heads. Missing teacher-phase data produce missing rows, not replacement layouts. Running out of time produces NOT_EXECUTED, not zero sensitivity. After a failed confirmation, no further neural probes run; saved-array reporting only.

Report same-noise action differences, first8/fullH40 displacement/rotation/gripper differences, noise spread and declared goal geometry descriptively. Geometry is evidence-only; do not pretend to have counterfactual expert action labels from the other instruction at that exact state. Shared initial grasp motion can correctly be identical for both tasks, and a T1-only trained policy is not expected to master T2. Therefore no numerical sensitivity threshold triggers a training branch, candidate change or claim that language is ignored/solved. These results prepare interpretation of the existing real paired-task qualification.

Additionally allow at most32 existing original-DROID inputs at the final delivery-model point, original processor/tag/noise/K4, ONLY if not already covered by R2 retained queries and if delivery is not delayed. Decoder/finite checks and output drift are descriptive, not task retention success. No fresh model download or broader benchmark. Total R3 added descriptive-call envelope **128 requests/512 K4 heads**, all inside unchanged global caps. Reused calls are not charged twice, actual load/parity/failed calls still count in the global ledger. Deliver `WP4/RESPONSE_PROBES.json` and optional `WP4/DROID32_RETENTION.json`.

### WP5 — fixed representation and coverage counterfactuals (P2, workerC; CPU only)

Finish the R2 bounded source/state/action coverage analysis, behavior occupancy, actual unique observation and event-first8/tail-only exposure. Separately quantify observed roundtrip clipping and proposed mathematical[-1,1] support for VALID rotation-matrix components, preserving XYZ/gripper math. Compare raw targets, empirical-range decoded targets and this alternative using CPU only. Include scaling effects; absence of clipping is not proof of a better learning geometry. No production statistics/processor refit or switch.

Audit near-observation/temporal ambiguity using true stored histories: exact-duplicate nonconflict does not prove full observability; similar single images do not prove contradictory labels. Produce one concrete HISTORY/ROTATION design with tests and explicit changed fields, if evidence warrants, marked NOT_AUTHORIZED_FOR_PRODUCTION. No new history model input, visual unfreezing, state dropout, sampler or third training arm this sprint. Deliver `WP5/COVERAGE_AND_ALTERNATIVES.md`; it cannot block valid current training.

### WP6 — recovery path implementation BEFORE the gate (P1, workerD)

Implement the original R2 branch-selector, source-prefix restoration checks, correction-suffix writer, language-aware index and recovery sampling as isolated future modules now. Unit-test with synthetic CPU fixtures clearly labeled not physical data. Design exact TRAIN8 x times2/6/10/16s branch IDs, missing-state handling, duplicate detection, timestamp offsets, no windows crossing takeover/reset, original32-slot denominator and codec contract. Do not fabricate actual model states or count prepared labels as collected examples.

No new teacher/model/physics collection is allowed until the R2 RECOVERY gate on completed primary4096 is actually met. Then the existing single collection of32 actual states proceeds without another permission round; at least24 distinct, restored, physically corrected and codec-compatible suffixes are required. Supervisor proof from successful teacher episodes alone does not admit learner-state labels. Inference never receives teacher phase/object truth or a teacher finishing script. Keep R2's latest collection start06:00 and45-minute preparation/run limit; time already spent in prospective coding should reduce, not consume, that critical-path setup.

Prepare DOSE plan generation and exact comparison inputs as the mutually exclusive alternative. R2 branch rules and LR schedules are unchanged: RECOVERY priority; otherwise dynamic-fit inadequacy plus registered improvement/stability gates may admit DOSE. A failed recovery bank does not permit switching to DOSE or another experiment. Deliver `WP6/RECOVERY_READY.json`, then if triggered a real `WP6/BRANCH_CARD.json` with evidence. Readiness is not a new qualification gate or a second subagent sign-off.

### WP7 — two-task ingestion/retention readiness (P1 preparation, workerE)

Reuse already successful T2 teacher banks; no third task, object or scene collection. Verify one shared instruction-aware tokenizer/collator/cache, same-layout input identity, common TRAIN-only statistics and exact50/50 task stream. At most CPU schema/mask/stream tests occur before T1 confirmation; no T2 gradient updates or expanded source physics. Run existing R2 acquisition baseline only when its gate opens.

Generate a frozen paired-result aggregator: per-task TRAIN/DEV, wrong-goal outcomes, both-goals-success per identical reset layout, and old T1 retention. One model/row/decoder, not task-specific weight switching. T1 confirmation IDs must never enter StageB data or selection. Unknown unused confirmation layouts may be checked by identifiers/history only; do not render/query them in readiness work.

Once T1 passes CONFIRM_A and remains protected, R2's existing SHARED schedule can start before06:30 if sufficient measured time remains. Shared development/confirmation gates and single permitted recovery remain unchanged. If T2 work would threaten delivery of T1, mark it DEFERRED. Deliver `WP7/SHARED_READY.json` and the eventual actual paired report; code readiness does not prove task capability.

### WP8 — runnable policy bundle and evidence handoff (P0 preparation, workerE/owner)

Build the load/evaluate CLI schema and dependency manifest now, not after09:45. It must bind weights, row mapping, task-language inputs, processor/statistics, K4, EEF/TCP/frame/time/gripper semantics, code/import hashes and source environment. Tests reuse actual canary/DIAG input hashes and existing evaluation outputs; no extra independent quality screen is opened. A fresh process reload that consumes a neural call must use the inherited allowed load/parity budget, not invisible calls.

Produce `WP8/DELIVERY_MANIFEST.json`, `WP8/LOAD_RECIPE.md`, `WP8/QUALIFICATION.json`, exact source video pointers and a concise factual handoff. Qualification.json cannot be set true by file presence or teacher success; it cites actual frozen-candidate results. Protect pristine, original/raw evidence, last full recovery and an independent copy of qualified PLACE_V1 before SHARED. Never overwrite the first qualified artifact with a failing successor. Credentials/private addresses are excluded from public files. Owned-storage cleanup removes only verified redundant exports/caches, never the unique evidence or only resume point.

Update repository navigation to the actual active authority without rewriting history; only the owner writes canonical status. Do not reopen PR77 or create long-lived coordination branches. Local delivery is mandatory, HF/public archival remains deferred and cannot trigger scientific reruns.

## 4. Work admission and priority rules

On adoption write a small `R3_ADOPTION.json` with the parent/progress receipts, new fixed release and actual current work state. Do not restart a healthy run to add R3 text. WP1 can finish operational adoption while WP2/WP6/WP8 prepare independently. WP3 consumes each new immutable record. WP4/WP5 are optional descriptive work, not dependencies of admission, optimization, candidate freeze or confirmation.

Device priority: (1) actual approved native training and mandatory recovery/checkpoint integrity; (2) the next admitted TRAIN/DEV/CONFIRM gate; (3) only then bounded WP4 probes; (4) videos/bulk archive. If an optional GPU probe cannot finish before the next mandatory request, do not start it. Optional analysis should produce a partial/inconclusive result rather than hold the mainline. There is no requirement to exhaust every listed task.

Increase the total OPTIONAL preparation allowance R2+R3 from90 to **180 aggregate worker wall-minutes**, not180 per worker and not a reset of already consumed R2 time. It covers WP4 CPU setup, WP5, and optional prospective analysis; mandatory training/evaluation integrity and necessary delivery remain within original global resource caps. End optional research by05:30. New correction collection latest06:00, optional optimization/SHARED starts latest06:30, optimizer stop08:30, model/physics stop09:45, handoff10:00. R2's45/75min confirmation reserve floors and larger measured latency reserves still bind.

The owner posts milestones on actual admission/carry/first formal update/checkpoint/branch/qualification/failure transitions. Do not spend three turns re-confirming the same resolved blocker. If progress is blocked on a real external dependency, identify it and prepare independent finite work; do not fake a healthy process or successful job. This is execution inside the user's active Codex goal, not a new ChatGPT automation or perpetual monitor.

## 5. Scientific learning and qualification are unchanged

T1 primary PLACE4096 from genuine pristine; interface1e-4/DiT1e-5, native loss, declared scope, current actual dual images/state/language, fixed data sampler/statistics. One optional T1 phase of at most2048: RECOVERY or DOSE according to R2, never both. No additional trainable layer, optimizer reset outside declared phase rules, precision mode, learning-rate sweep, normalization/history experiment or data relabeling.

T1 first qualifying checkpoint TRAIN>=7/8 and DEV>=13/16 is frozen; one CONFIRM_A285300..285331 must achieve>=26/32 full R1 task successes. If unsuccessful, no new model selection/training/neural probes afterwards. T1 qualified artifact is permanently preserved. Only then StageB can train the shared T1/T2 policy (4096 plus at most one original2048 recovery phase). Each task requires TRAIN7/8,DEV13/16 and paired DEV12/16; one CONFIRM_B285400..285431 requires each task26/32 and paired24/32. Full32s task, all41 samples28..30, genuine grasp/lift/transfer/place/release/withdraw predicates are unchanged. Source qualification is not a population confidence guarantee, real-time operation, Go2 success or cross-embodiment generalization.

Exploratory response/coverage probes do not tune on DEV/confirmation or authorize extra candidates. Engineering repairs preserve observed failures; they do not label a quality rejection invalid merely to retry. No new physical starts are introduced by R3 beyond the existing R2 screens/corrections/confirmation. No new teacher bank is required.

## 6. One cumulative budget, not another grant of all resources

R1+R2+R3 totals remain: formal updates A6144+B6144=12288; all completed<=12512 including64 canary+32 memory+128 completed rollback; technical training recoveries<=4. Actual attempted microbatches/chunks and failed allocations are also recorded. No reset after the Mapping audit bug. Quality requests<=32768, K4 heads<=131072; physical starts<=840 including at most32 technical replacements and zero quality retries; controls580000, integrations2900000, RGB580000, dispatch1160000, teacher controls400000, planner4096; GPU32h, CPU256coreh, owned new outputs240GiB. Data and prior176 teacher episodes already count.

The R2 nominal upper recipe has29952 quality requests. At most128 extra R3 descriptive requests makes30080, or120320 K4 heads, leaving2688 request/10752-head headroom below the unchanged hard caps. Do not interpret headroom as permission for other tests; retries/load/parity still require their named authorization. The conservative R2 physical recipe remains656, or688 with all32 technical replacements; R3 adds0. Optional work yields before any hard cap or deadline threatens the next admitted confirmation.

Required terminal facts: exact work-package completion/deferment, real optimizer/episode counts, valid quality failures, codec caveats, selected candidate identity and explicit qualification state. No complete/qualified claim from prepared code, aggregate model loss, archived weights or expert100%. selected/deployment=NONE unless a later separate target authorization changes them.

## 7. Publication and short owner instruction

This upgrade changes task allocation and limited descriptive permissions, not live execution code. Reviewer reads: R2 protocol and actual latest owner return #6065801898; branch ba57d683 and its fixed-DIAG raw/codec CPU preparation. No reviewer SSH, qsub, model calls, physical execution or remote agent launches occurred. New work packages are prospective. Parent scientific claims and unknowns are preserved.

Owner startup: adopt R3 canonical/fixed release without restarting R2; use one owner and up to5 workers for the eight bounded packages. First close actual carry and enter PLACE; pipeline immutable physics feedback, prepare one authorized recovery route and delivery; descriptive language/retention<=128 queries and optional CPU<=180 cumulative worker-minutes do not block the mainline. Keep data/weights/precision/WBC and gates as declared, one training lease, no HF dependency, no new task or optimizer search. Confirm/protect T1 before shared learning; stop by Oct9 10:00 SGT and report actual results.

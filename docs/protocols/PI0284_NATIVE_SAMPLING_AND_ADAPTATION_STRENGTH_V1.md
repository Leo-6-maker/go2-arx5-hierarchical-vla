# PI0284 — Native sampling gap and matched adaptation strength

Authority: **PI0284_NATIVE_SAMPLING_AND_ADAPTATION_STRENGTH_GO_V1**.
Date: 2026-10-08, Asia/Singapore. Scientific ledger: Issue #11.
Final review SHA: `e873939fa6a78698211b18a8995cfce3f664f831`; initial review SHA: `e193847f3feeb2a4bf1e5028f0fd3204f65eb0a0`.
Predecessor terminal: **6053296607**, published **2026-10-08 13:49:46 SGT**, under PI0283 canonical6048886543/release508e3db7b63d3c47758794f3ffbad87b577e6e9d.
The new canonical binds this protocol and `reports/PI0284/design/REVIEW_AND_LIMITS.json` to an immutable release.

## 1. Decision and mainline

The user requests actual review and the next experiment along the shared task-space VLA mainline. PI0283 completed during this review. Preserve its complete, valid negative result: do not repeat EVENT, reset its budget, or treat the earlier104-update receipt as current progress. This new goal first asks whether the current native sampler adequately realizes the learned action distribution; only then tests adaptation strength with two matched learning-rate arms. It is not another unconditional lower-LR continuation or a new task-specific auxiliary loss.

The deliverable is a physically tested, locally reloadable source-Panda EEF+gripper policy, or a finite diagnosis separating sampling, dynamic action fitting, conditioning and closed-loop behavior. The manipulation charter remains: task-space behavior above a fixed EEF interface, no target joint-action labels, data independent of target-WBC success. Simple pose matching is diagnostic, not the research endpoint. A useful source policy is a prerequisite for a separately authorized shared multi-task and later target-composition study, not evidence that either is already achieved.

**WBC, target physics, target VLA and hardware calls remain ZERO.** No new robot, environment, camera distribution, task, teacher, recovery data production or full-VLM tuning. Original task weights outside the pristine-derived PI0282/83 lineage remain excluded. All previous drivers stay sealed or paused. `selected/deployment=NONE`; source qualification is a separate field.

The user's explicit wall-clock removal is recorded in PI0283 `CLOCK_RESOLUTION.json` (user instruction: 时间限制解除，按计划推进). Do not reinstate09:00 or silently roll it to tomorrow. This goal has new finite work/resource caps, not unlimited execution or a promised completion time. Refresh actual job/process ownership at adoption; if an old healthy process is still closing out, do not compete with or modify it. No second-owner signature is needed after a specified gate.

## 2. Reviewed evidence and hypotheses

PI0283 MORE2048 and EVENT1024 are complete. Their cumulative continuation is NOT a matched-start sampler ablation. Compared with MORE2048, EVENT reduces TRAIN prefix XYZ median15.244 to12.193mm and fit2/24 to6/24; all six fits are late-hold probes. Approach/grasp/transfer each remain0/4, all four first-release probes fail. EVENT approach errors are42.066/19.205/24.199/28.453mm; first-release gripper MAEs .864258/.617432/.745850/.375488, median .681641. A small aggregate gripper median .002686 conceals the transition tail.

EVENT sustained grasp3/4, ever on_top2/4, first complete success1/4, persistent0/4. Condition10003 has44 consecutive20Hz successful samples6.65..8.80s, first loss8.85s. Loss coincides with static=false; causality is not established. Finally static/not-grasped are true but on_top0/41. Do not call this continuous100Hz proof, permanent placement, or the old COVER terminal-only failure. Other conditions have earlier interaction deficits. Reference installation and actual tracking are separate; two EVENT conditions have zero IK holds.

Training was real: shared DiT and row10 changed, VLM and other31 rows stayed fixed, native masked reduction and accumulation were checked. EVENT Adam global step5120; native loss first/last16 means .0364563/.0231171 are different sampled batches, not a fixed validation curve. Dynamic free-running fit still being poor does not prove low model capacity, missing task breadth, or one precision threshold.

Two concrete untested levers remain. The pinned official sampler uses Euler updates `actions += (1/N)*pred_velocity` and exposes the number of inference steps; N=4 is the default, not a proved accuracy guarantee on our adapted distribution. Official finetune config defaults LR1e-4, whereas actual DiT peaks have declined1e-5 ->5e-6 ->2.5e-6. This suggests a sampling and/or optimization-strength hypothesis, NOT a demonstrated root cause or proof that the official LR is optimal. Four versus16 sampling steps concerns integration of the learned flow, not BF16 versusFP32 numerical precision, H40 length, or control frequency.

The prior discussion also warned that repeated tuning on four development states can overfit the research process even while dynamic TRAIN windows remain poorly fit. This protocol therefore limits both hypotheses and retains one never-opened confirmation panel. It does not claim multi-task learning from a fixed StackCube instruction.

## 3. Immutable parent and invariants

Parent P is **PI0283 EVENT1024**, full mixed-master SHA256 `ac8fad1a39aad656fac3f0bdafaa856b984d67cd55a0df111103ae79a80f8e2a`, Adam global5120. Complete source: `/scratch/e1192548/Projects/go2-arx5/experiments/PI0283/learning/EVENT/main/checkpoint-1024`, actual14-member hashes in `hop/runtime/EVENT_TRAINING_FINAL_CPU_AUDIT.json`. HF recovery reference: `leoyuer/pi0246-gate-pass-campaign`, `arms/PI0283/EVENT/checkpoint-1024` at `2a288c49d2d2aa7517da2103bf92911bade01947`; local complete recovery is preferred. A model-only view is not an optimizer restore. No historical-score parent search.

Pristine ancestry stays `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`, official code51d4c89f72fda44cbf77285c6a8114b52676b8a1, Cosmos9ce19a195e423419c349abfc86fd07178b230561. Do not reload a fresh/random interface or copy a source embodiment row in this goal.

Keep exact PI0282 EEF10 processor/statistics/mapping: original TRAIN-only min/max, native constant handling and clipped inverse; actual two raw128 current RGB views; actual TCP pose and measured gripper; fixed language; relative EEF conversion from absolute task-frame commands and actual anchor; absolute gripper command; first10 active dimensions in132 native dimensions. H40,20Hz,.05s points, execute8 points then reobserve,32s task,100Hz physics remain fixed. The original source controller, SafetyGate and object scorer are unchanged. No joint/oracle/object truth/time/phase/success enters policy observations. All132 native noise channels remain, including masked padding. No smoothing, thresholded gripper, altered crop, normalization change, prefix extension, synthetic history, RTC/previous-action inpainting or success-triggered finish.

Reuse existing data, original expert codec4/4 and exact cache proofs; do not re-run their18-minute pipeline or old valid failure panels. Verify current file identities and actual loaded `__file__`/runtime manifests. The PI0283 builder generates runtime code from PI0282; a similarly named historical helper is not runtime evidence. Use an explicit PI0284 stage/sampler/arm config and regression checks rather than repeatedly replacing all occurrences of2048. Never edit files loaded by an active process.

## 4. G0 — Adoption and low-cost decomposition

Refresh jobs, resources, actual quota or owner-approved volume, model/data/observation availability and confirmation exposure. Link old PI0282/83 counters without transferring unused allowances. Preserve unique complete recoveries and failures. The audited final science can be adopted independently of optional remote archive completion.

Before new learning, independently summarize existing ACTION/MORE/EVENT results by approach, grasp, transport, first release and late hold. Reuse raw arrays to align requested/installed/actual TCP, gripper command/opening/contact, object outcome and success-loss times. Report per-channel/per-time pre-inverse out-of-range; aggregate OOB does not prove a bad normalizer. If a proven contract/label/scorer contradiction exists, stop the affected new science for a precise decision; this authority permits location-only/serialization/runtime repairs, not new labels or retroactive score changes.

On the unchanged DIAG48, compare training construction with inference-only condition construction at the final prepared state/video-token/attention-mask level, allowing only the declared dtype cast and removal of supervised action fields. Record actual module modes, not inferred frozen flags. Native scalar loss must still be the native masked mean with correct accumulation. Existing complete proofs are reused; no new full-weight audit per microbatch.

Allow at most120 no-gradient, native forward diagnostic examples:24 fixed TRAIN anchors x5 deterministic diagnostic seeds, preserving the native noise/time sampler. Record actual time, velocity loss by active channel and phase, and sampled-time bins. This is target-conditioned validation, never a policy episode or a reason to inject targets into `get_action`. Keep these counters separate from free-running queries and training; total native diagnostic/extra parity examples cap256. Missing coverage of a time bin is UNKNOWN, not permission to tune the sampling distribution.

## 5. G1 — Same-weight, same-noise native S4 versus S16

Use P and the unchanged DIAG48. For each observation compare native N=4 and N=16 on two fixed full132-dimensional BF16 noise packets. Packet0 reuses the actual predecessor packet/output whenever model, conditioning and sampler identities exactly match. Packet1 is drawn once from namespace `PI0284_DIAG:<UID>:1`; save actual bytes and replay them for the paired N. No best-of-two, averaging actions, seed sweeps or discarding bad samples. Upper bound192 free-running calls; exact reused results debit0 new calls.

Only the supported inference-step parameter may change. No alternate integrator, precision change, time-grid patch or padding projection. Verify CPU config and one native-wrapper parity within the shared reload pool. The previous `==4` execution/head-count guards must become explicit allowed-N guards, not be deleted. Hook and count actual DiT forwards: S16 is16, not4. Save sampler N as part of policy identity. Backbone work occurs once per query; do not claim end-to-end latency necessarily becomes four times larger or stays unchanged. Measure it.

Report first8 and fullH40 XYZ/SO3/gripper, original gate result, event timing and channel OOB separately. Selection uses TRAIN only; VAL is descriptive development evidence, never a new blind test. For each TRAIN anchor average over the two fixed noise packets:
`E_N = mean_noise[(XYZ_RMS/0.010)^2 + (SO3_RMS/0.10)^2 + (gripper_MAE/0.10)^2]`.
These scales define a diagnostic comparison, not new task thresholds.

Set the single prospective sampler S*=16 only if ALL hold: median paired `E_16/(E_4+1e-12)` <=0.80; at least16/24 TRAIN anchors have strictly lower E; each component median is no more than1.10 times S4 (absolute numerical floor1e-8); no added nonfinite/invalid decode; no increase in original gate REJECT counts among the48 TRAIN draws. Otherwise S*=4. Publish the complete table and the selection before any new physical score. This small test can show sensitivity to sampling resolution, not prove S16 convergence or a unique numerical cause.

If S*=16, execute exactly one new P/S16 TRAIN4 panel using the original condition/query-index noise packets. This is a declared different sampler, not a retry of P/S4. If TRAIN>=3/4, run one DEV8. If DEV>=6/8, immediately freeze P/S16 for the sole confirmation in section8; do not train to spend the budget. Failure of confirmation terminates this goal. If development fails, retain those results and continue only through section7's fitting gate. If S*=4, reuse P/S4's existing physical panel; never repeat it.

## 6. Conditioning probe — natural examples, not fake packets

From already recorded TRAIN initial observations, freeze at most12 disjoint pairs by seed/UID order before new scores. Require actual TCP differences<=1mm and<=0.01rad, actual opening difference<=0.01, same camera configuration and instruction, and expert future XYZ difference>=10mm at one executed-prefix point. Each member retains its own real state, images and labels. Do not swap images into an inconsistent state or feed simulator object truth.

Query the24 natural packets with two fixed paired noise draws using S*, upper bound48 calls. Compare predicted target displacement changes with expert displacement changes, and between-noise spread. This reports whether outputs respond appropriately across these naturally different scenes; weak response is not proof that the VLM is unused or the only cause of failure. If fewer eligible pairs exist, report the exact reduced denominator and do not render new scenes. No model training decision depends on cherry-picked pairs or language probes with no task labels. This diagnostic and CPU alias checks should be light; do not build a new perception benchmark.

## 7. G2 — One matched adaptation-strength comparison

Enter only if P is not development-qualified and expert-state fit under S* is still inadequate. Define fit per anchor as both fixed noises meeting XYZ<=10mm, SO3<=0.10rad and gripper<=0.10; adequate requires>=18/24 and at least half the probes in each represented phase. If expert-state fit is adequate but physical failure persists, stop further optimization in this goal and return `FIT_ADEQUATE_CLOSED_LOOP_GAP`: prepare a source-state recovery/observability decision, but do not produce teacher data or start an unregistered repair. Likewise, a confirmed semantic bug blocks learning rather than authorizing a different architecture.

When fitting is inadequate, the gate grants TWO independent2048-update arms automatically:

| Arm | Row10 interface peak LR | Shared DiT peak LR |
|---|---:|---:|
| LOW2048 | 1e-4 | 1e-5 |
| STANDARD2048 | 1e-4 | 1e-4 |

Both start from the exact P weights AND identical carried Adam states/global5120; neither is a continuation of the other. The between-arm intended difference is ONLY the shared-DiT LR. STANDARD's value is informed by the pinned official default, not a claim that the rest of our recipe is fully stock or that a higher LR must work. Both resetting the phase schedule and raising the interface LR relative to EVENT are COMMON changes; comparison to P alone cannot attribute gains solely to DiT LR.

Keep the same470 trainable packed parameter tensors:14 row10 state/action encoder/decoder tensors plus456 shared-DiT tensors. VLM, other31 projector rows, position embeddings, vlln and VL attention remain frozen. Projector decay0, DiT decay1e-5, AdamW(.9,.999),epsilon1e-8,global gradclip1. Carry moments, reset only the declared cosine2048 phase scheduler with103 warmup updates and the table's peaks. No adaptive retuning, learning-rate tournament, additional loss, LoRA, full-VLM unfreezing or new row initialization. Monitor loss/gradient/update-to-weight norms; a real nonfinite fault stops that arm without substituting another LR.

Both arms share one pre-score PLAN16384 generated by continuing the sealed EVENT sampling rule: all93 original TRAIN episodes, four phases25%, within each group50% critical-event stream plus50% whole-phase first/last/eight-strata stream; fixed hashed cycling and continued offsets. Freeze UIDs, labels and all weights once. Existing15 VAL are never training data. No new recovery suffixes, scenes, labels, image augmentation or statistics. Report unique UIDs, first8 versus tail-only close/open events, and late static repetition. More draws are not more independent demonstrations.

Effective batch8, identical layout per arm, preferred H2004x2. Only prospective memory-driven2x4 or1x8 before both arms is allowed; freeze the common layout. Same FP32 trainable masters/BF16 autocast, deterministic runtime, TF32/fp16 off and original native flow-matching. Same actual training-noise/time packets per update/microbatch across arms, independent loader RNG. Verify LR groups after optimizer restore; no accidental zero-LR continuation. Native target noise remains full132-dimensional.

Run a real8 continuous versus4+fresh-process4 proof for each arm (32 discarded completed updates maximum). Reuse generic mechanisms and already validated data; record exact frozen tensors, data/noise/cursor and numerical/bitwise learned-state equality. Retain prior numerical tolerance atol1e-6/rtol1e-5 for updatedFP32 and require no prediction/gate discrepancy; do not relax on failure. Frozen DiT or VLM handling must not disconnect permitted encoder gradients. Emit actual module-mode checks in the proof, first8,health128 and checkpoints.

One training lease only. Prefer STANDARD first for the practical native-strength path. Complete the second arm for an interpretable comparison unless a fully confirmed source candidate already exists, resources expire, or a real engineering/numerical failure stops it. Any omitted/shortened arm is explicitly NOT_EXECUTED/PARTIAL, never a completed ablation.

## 8. Fixed evaluation, selection and no holdout recycling

Save full local Trainers at1024/2048. Each arm at both points receives DIAG48 with the same two fixed diagnostic noises, maximum384 new query slots; no mid-point physical panels or subjective early selection. Only2048 endpoints run TRAIN4, using S* fixed before training. Endpoints reaching TRAIN>=3/4 may each run DEV8. At most three DEV8 panels total, including P/S16. All compared policies get their own actual new observations and the same noise bytes per condition/query index; divergent physical visits are part of behavior.

The task is unchanged source-Panda native PD, non-realtime32s, first8/40 action execution at20Hz, fresh actual RGB/TCP/opening every replan. Persistent qualification requires a complete valid32s AND on_top/static/not_grasped at ALL41 control samples28..30s. Grasp, geometric on_top, first success, low RMSE and faithful state reconstruction are not replacements. Preserve original safety, contact/IK and all failures. Model predicts every episode to the end; no teacher, time/phase/success-triggered handoff or stopping.

If the two training endpoints qualify, choose by greater DEV persistent count, then TRAIN count, then LOW (smaller shared update). Do not use videos or the retention probe to change this rule. If STANDARD is confirmed before LOW runs, stop and disclose the incomplete causal comparison. Only one model+processor+sampler identity may enter confirmation for this goal.

Refresh exposure of CONFIRM30032..30047. If any were used in PI0283 or elsewhere, do not recycle or silently substitute seeds; return HOLDOUT_ALREADY_USED and leave independent source qualification pending a separately fixed panel. With the currently recorded unqueried state, freeze one candidate, fresh-process reload/hash-check it and run exactly16 conditions once. TRAIN>=3/4, DEV>=6/8, CONFIRM>=12/16 are all required for SOURCE_PROTOTYPE_QUALIFIED. On confirmation failure, stop the goal: no further training, second candidate or seed search. Incomplete confirmation reports measured/missing counts, not invented failures or a PASS.

For P and both completed endpoints, at most32 original DROID saved observations each, original processor/tag/stats/noise and S4, form a96-call descriptive retention check. Reuse any exact saved outputs. Parent retention may run before training; endpoint retention follows. Output drift is not itself measured catastrophic forgetting, and output similarity is not a multi-task success benchmark. Frozen parameter integrity remains mandatory even if a retention probe is resource-deferred.

## 9. Implementation and acceleration

Implement isolated `scripts/pi0284_execution/` and `reports/PI0284/` with an explicit stage/sampler/arm card, source `__file__`/hash manifests and CPU cap tests before GPU work. Reuse actual validated native model/codec/source code, not a same-named old helper. All admissible gates automatically continue within this authority; no extra DATA_READY or subagent signoff.

Use one owner and up to three disjoint workers (DATA/DIAGNOSTICS, TRAINING, SOURCE/AUDIT). Actual GPU leases, atomic global attempts and artifact ownership govern parallelism. H200 is preferred for training; Corelab for resident inference/source physics. Never duplicate queued/running jobs, touch other projects, or fabricate PBS IDs. If a legal common runtime does not fit, report the resource gap without silently changing scope/precision.

Keep two-worker verified CPU caches, pinned memory and useful samples/s reporting. No new worker/compile/DDP/fused optimizer search. Cache keys bind the unchanged codec and original sample identities; no stale RGB or trainable-network activations. A frozen model may share its own deterministic conditioning features within a paired diagnostic only with native parity; this is optional, not a mandatory new system.

Before remote training, make model-only direct transport usable and validate a parent reload. PI0283 revoked its ephemeral key; do not assume it remains active. Reuse an already approved scoped transport or owner-approved local route, never commit credentials or request broad trust. No public-HF upload/download in the feedback critical path. Original models/data are already local; no new weight/platform downloads. HF archival is deferred, local complete recovery and truthful evidence are not. Transfer whole changed tensors with exact full-state reconstruction, not numeric residuals or quantization; never transfer optimizer merely for inference. Snapshot copying must be atomic at a completed optimizer step.

Protect the parent and last complete checkpoint, verify quota/headroom and no unique-artifact deletion. Diagnostic and video processing read saved arrays/frames; no extra render to beautify results. If the finite experiment ends without qualification, deliver the best explicitly unconfirmed model and a mechanism table, not another implicit continuation.

## 10. New finite counters

Old PI0282 completed2080/formal2048,874queries,3492heads,16starts and PI0283 completed3088/formal3072,1108queries,4424heads,13starts stay sealed. New budgets do not reset or consume their unused allowances.

New formal updates<=4096 (2x2048), new formal draws<=32768. Canary<=32 completed, prospective memory<=16, completed technical replay<=64; total new completed updates<=4208. At most2 technical training recoveries; zero quality restarts. Count every attempted native training forward separately, cap35000 forward batches to cover a declared1x8 fallback; no additional gradient experiments outside these updates.

Native no-gradient diagnostic/parity examples<=256, counted separately from free-running policy requests. Nominal maximum free-running requests:52 model episodes x80=4160; paired sampler DIAG<=192; natural conditioning<=48; four training checkpoint DIAG panels x96=384; DROID retention<=96; serving/canary/reload<=64; total4944. Hard cap5120 free-running requests and81920 actual quality-DiT evaluations, with per-query N recorded. Native diagnostic forwards and training forwards are separate counters, never counted as successful sampler queries.

Nominal52 physical starts: P/S16 TRAIN4 + two learning endpoint TRAIN4 (8) + at most3 DEV8 (24) + one CONFIRM16. Hard56 starts including at most4 same-condition technical replacements, zero valid-quality retries. Nominal33280 controls/166400 integrations; hard40000 control steps,200000 per-environment integrations,40000 RGB pairs,80000 control/gripper dispatches. Startup/neutral/prefix/partial work counts; no hidden vectorized extra environments.

Existing legal resources only:8 new cumulative GPU-hours across machines,96 CPU-core-hours,120GiB retained new outputs/cache, no new model downloads/purchases. These are ceilings, not time estimates or success promises. WBC/target simulation/target queries/hardware/new physical TRAIN collection/new teacher/multi-task campaign counters all0. Resource depletion terminates with honest partials; it does not open an additional arm.

## 11. Interpretation and downstream mainline

A successful P/S16 would show that sampler configuration matters for this adapted policy, not prove that no further learning is needed generally or that S4 is universally wrong. A STANDARD advantage over LOW would support stronger action-side adaptation under this matched setup, not prove pretraining was useless or that all official defaults are optimal. Both-arm failure with poor dynamic fit directs the next decision toward conditioning/observability, label consistency or representation—not another automatic halving of LR. Adequate fixed fit with weak physical behavior motivates a separately bounded learner-state/closed-loop study. Native-loss scalar improvement alone cannot choose the branch.

Only after a physically confirmed source policy exists should the next scientific protocol jointly train two or more related manipulation tasks under one EEF ABI, natural same-scene/different-instruction supervision, task/event/episode-balanced data, old-task retention and independent layouts. No per-task model merging, WBC-specific dataset filter or claim that diffusion policies cannot be multi-task. This goal may deliver a small CPU-only next-study specification, but authorizes no such collection or training. Source prototype qualification does not remove the frozen WBC's separate execution gap.

Required terminal artifacts: parent binding and reused evidence; paired sampler table; native-loss/conditioning diagnostics; chosen S* and actual latency; fixed training plans/noise, common-start/optimizer/LR/frozen proofs; local load recipe and complete recoveries; all per-condition task funnels/retention/results; actual budgets; immutable GitHub terminal and readback. Possible states: SOURCE_PROTOTYPE_QUALIFIED, VALID_QUALITY_FAIL, FIT_ADEQUATE_CLOSED_LOOP_GAP, BLOCKED_SEMANTICS_OR_RESOURCE, PARTIAL. Keep selected/deployment=NONE.

## 12. Audit basis and limits

Actually read: latest branch/comments through PI0283 terminal6053296607; final report/terminal receipt and MORE/EVENT full Trainer CPU audits at e873939fa6a78698211b18a8995cfce3f664f831; prior clock resolution, EVENT card, actual runtime builder and phase/restore hooks, EEF codec; pinned official native input/flow sampler and finetune config. Official sources are NVIDIA/Isaac-GR00T at51d4c89f72fda44cbf77285c6a8114b52676b8a1 (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`, `gr00t/configs/finetune_config.py`, documented denoising-step parameter). The supplied prior review is framing, not live evidence. Source-derived facts, hypotheses and prospective choices are separated above.

Reviewer independently checked headline/error/budget arithmetic, not all remote raw tensors, videos or loaded process state. No SSH, model query, training, simulation or remote stop was performed by publication. An initial guess at `scripts/pi0283_execution/train.py` returned404; the subsequently read build.py showed the generated runtime path, so no guessed file was represented as actual source. Publication authorizes the existing owner; it does not prove a new job started.

Startup: adopt sealed EVENT1024 and actual terminal; freeze WBC. First run same-input/full-noise native S4/S16 and conditioning checks, choose S* by the fixed TRAIN gate. A qualified sampler-only policy goes directly to one held-out confirmation. Otherwise, if dynamic fit is inadequate, run independent LOW2048 and STANDARD2048 from the same full parent/Adam state, same event data/native loss, only shared-DiT LR differing. Corelab source task gates and exact local delivery remain mandatory; no new tasks, teacher, HF-critical transport, old-driver restart or unbounded continuation.

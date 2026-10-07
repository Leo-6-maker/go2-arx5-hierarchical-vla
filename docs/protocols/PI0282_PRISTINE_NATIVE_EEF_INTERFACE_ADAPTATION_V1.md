# PI0282 — Pristine GR00T native-loss EEF interface adaptation

Authority: **PI0282_PRISTINE_NATIVE_EEF_ADAPTATION_GO_V1**. Date: 2026-10-08, Asia/Singapore.
Review/evidence base: `6217dd890689a61bcc53f74a41c047017e049ede`, branch `pi0267/executor-decoupled-recovery`.
Predecessor terminal: Issue #11 **6042626674**, published **2026-10-08 00:50:36 SGT**, under PI0281 release `360b412f55be8aec1568054d7d039ce7de2bc597`.
The new Issue #11 canonical binds this protocol and `reports/PI0282/design/REVIEW_AND_LIMITS.json` at one immutable release.

## 1. Decision and boundaries

The requested native baseline has now produced enough evidence for a minimum adaptation decision: genuine original model calls work; original joint and direct-EEF source task policies fail the developed StackCube panel; the expert source bridge succeeds; no raw-model route accepts honest Go2/ARX5 EEF-only feedback; the delivered WBC still fails orientation qualification. The target GR00T+Go2 closed loop has NOT been completed.

**This new authority explicitly permits one bounded, fresh-pretrained EEF-only adaptation comparison, including training and source physical evaluation.** It is the sole new exception to the prior no-training priority, not an automatic restart of PI0279 or any GRIP/FOCUS/COVER/CLEAR driver. Keep those pause markers and all old weights, data, failures and ledgers. PI0280/PI0281 remain sealed. No target WBC training, target physics, new teacher collection, multi-task/scene campaign or hardware is authorized here.

The immediate deliverable is a locally runnable source-domain EEF+gripper task policy, or a finite, interpretable failure of native adaptation. In parallel, package the remaining WBC gap for the senior researcher; do not use high-level learning to conceal poor low-level orientation tracking. The task-space charter #5994405991 remains in force: task supervision and admission are independent of target-WBC success, with no target joint-action labels. Success on source Panda is not Go2 deployment or cross-embodiment validation. `selected/deployment=NONE` throughout; use a separate `source_candidate` field.

## 2. Reviewed evidence and what it does not establish

- Reused PI0280 direct EEF: persistent0/4. PI0281 fresh original-base JOINT:0/4; commanded-joint FK EEF replay:0/4; existing expert DROID/time-grid bridge:4/4, all41 final samples. Source controller fixtures passed. The joint10002 condition grasps and lifts about0.522m but does not place/release; other new joint conditions do not grasp. This is not the tuned COVER policy's terminal-only failure pattern.
- Same-input outputs differ: A3 per-condition H40 mean EEF-vs-FK(q) XYZ differences40.66/70.92/28.68/57.42mm, SO3 .308/.295/.112/.367rad. Neither route succeeds, so mismatch alone is not the full cause. No proven source input/decode/clock bug opened the conditional direct-EEF rerun.
- JOINT10000 stops before dispatch on q6=3.758334875>3.7525rad; FK10000 encounters the original angular-speed gate. Keep valid truncated attempts in denominators. Installed IK targets and actual URDF penetration are distinct; do not use one as the other.
- The actual WBC branch repair removes the immediate return-path jump/abort, but final progress .043294m differs from s_ref .100000m. Final2s SO3 RMS hold/monotonic/return=1.105218/.629325/.543153rad, versus the unchanged .15rad criterion. The checked actor/preview/base/same-TCP FK chain is numerically consistent. This supports a capability/configuration-distribution gap for this delivered version and these fixtures, not a universal proof about all WBCs or the senior's unreceived upgrade.
- All8 original pretrain routes remain structurally unsuitable for honest target EEF/gripper-only observations. Removing required joints by silently padding a DROID request is not zero-shot transfer.
- Important source-control caveat: `pi0281_execution/source_math.py::expert_bridge` resamples pose through15Hz but explicitly preserves original20Hz gripper events. Its4/4 is not proof that resampling the entire pose+gripper program to15Hz is lossless. This goal retains the actual existing task-data20Hz grid instead of relying on that stronger claim.

These results justify trying adaptation, but do not show that projector-only training is sufficient, that a particular data count guarantees success, or that model capacity is inadequate. They also do not establish that the old long fine-tuning lineage destroyed pretraining; a matched initialization comparison would be a different experiment.

## 3. Fixed model identity and a genuinely declared new interface

Use the already verified pristine `nvidia/GR00T-N1.7-3B`, revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`; official code `51d4c89f72fda44cbf77285c6a8114b52676b8a1`; Cosmos `9ce19a195e423419c349abfc86fd07178b230561`. Reuse original local caches and the PI0280 loading report. Before training, match all effective tensors and missing/unexpected keys against that source. Never initialize from project-adapted checkpoints or task-posttrained DROID/LIBERO weights. A location-only rebind of identical local dependencies is allowed.

Register a new `NEW_EMBODIMENT` EEF10 contract at the original mapping row10. Both arms start from **the same serialized original row10 bytes**, with no fresh random initialization, source-row copying, LoRA, new head, distillation, or learned source-joint imputation. Row10 is an original placeholder of unverified pretraining coverage, not a pretrained Go2 interface. Name the initial artifact `UNADAPTED_INTERFACE0`, not raw zero-shot GR00T. Both the data-contract adaptation and subsequent learned changes are disclosed.

The invariant shared weights are genuine pretrained weights; this initialization choice does not prove that the placeholder is optimal. Do not search row initializations in this goal. Archive row10 slice hashes so a later initialization study can be cleanly distinguished.

Model input: actual body/external RGB, actual wrist RGB, actual achieved TCP pose, actual measured gripper opening fraction, and the original fixed instruction `Stack the red cube on the green cube and release it.` No joint state, body configuration, teacher phase, elapsed task time, success flag, object/target truth or source planner result enters the policy. Simulator truth is allowed only for source safety and scoring.

For this first single-task comparison use current-frame video indices[0] and state[0]. Do not fabricate history, add an oracle phase, or spend the goal recollecting history. This is a declared new interface, not preservation of the original DROID [-15,0] contract. History-dependent learning remains untested. RGB uses the existing original raw128x128 source arrays where recorded, never downsample96 then enlarge; unavailable raw members are a concrete data blocker, not permission to invent pixels. Future target camera differences remain a separate domain gap.

Action: H40 at the existing20Hz task-data period .05s, first8 actions executed before replanning. External commands are absolute task-frame TCP XYZ+row-rot6d plus absolute gripper command fraction. The official processor converts the EEF part relative to the actual anchor TCP (`RELATIVE/EEF/XYZ_ROT6D`); gripper remains `ABSOLUTE/NON_EEF`. Actual measured opening is not the action target. Keep the original source actuator conversion and physical width/polarity documentation; no fictitious joints are inserted. Output/state beyond first10 dimensions are structural padding in132 dimensions, not missing DROID joints.

H40/132 and native S4 are unchanged architectural dimensions. The20Hz cadence and new fields are learned interface semantics, not a claim of15Hz DROID equivalence. No pose or gripper retiming is needed for the source data or evaluation.

## 4. Data reuse, native processor, and evidence admission

Reuse the93 already admitted PI0276 TRAIN EEF-expert episodes and15 admitted VAL episodes, identified by their original card and file hashes. All original96/16 source attempts and exclusions remain retained; do not claim that55,893 overlapping windows are independent demonstrations. No new task, source teacher, learner-state collection, PI0279 recovery suffix, or target-WBC filtering is introduced. Reuse existing commanded references and observations from the same actual source EEF executor; never label future achieved poses as commanded actions.

Native state/action processing is a new artifact. Fit NEW_EMBODIMENT state, action and relative-action statistics once on the complete eligible TRAIN window population, not the visitation sample; preserve every existing pretrain route's original statistics. Use the official supported `use_percentiles=False` (full TRAIN min/max), `clip_outliers=True`, original constant-dimension handling and original clipped inverse. This explicit baseline choice avoids quantile clipping of valid TRAIN extremes without importing the old unbounded affine monkeypatch. It is not the unchanged PI0276 statistics and not the exact official default recipe. No handcrafted rotation ranges, fitted target offsets or test-derived normalization.

Record constant/near-constant dimensions, native normalized magnitudes, mask shapes, and roundtrip XYZ/SO3/gripper errors. Do not silently replace degenerate statistics, drop difficult labels or add an epsilon/clamp outside the native processor. Finite but degenerate rotations must be reported and never repaired into successful commands. The admission goal is sound data, not identical scalar representation error on constants: report their physical reconstruction effect. Unexplained physical roundtrip error above1e-5m/1e-4rad/1e-6gripper fraction on TRAIN requires a concrete codec/data correction decision rather than training through it.

Retain original model-side image/token processing. For this bounded deterministic comparison configure supported full-field resizing (shortest edge256, crop_fraction1.0), zero random rotation and zero colour jitter, no extra augmentation. Do not patch forward or normalization functions. Record resolved processor values rather than assuming that None disables a checkpoint default. Both arms share exactly the same encoded conditions and labels. Changing augmentation is a later experiment.

Perform CPU checks on actual encoded inputs: legal field allowlist, state shape1x132, action/mask40x132 with first10 valid; one moving-anchor/nonidentity rotation test; action/actual-state separation; same labels with inference-only input construction; clamp frequency and invalid-rotation checks. Full132-dimensional native training and inference noise remains. Masked loss channels are not assumed causally absent from the network. Do not zero unused generated channels or alter the sampler to hide coupling.

Before learning, use four existing expert command streams10000..10003 through this exact encode/decode20Hz task pipeline and original EEF physical executor. This is four reference-replay checks, not new training data. All four must retain full32s and original persistent task success. If not, keep failures and stop training on a proven supervision/codec blocker; no seed substitution. No additional data-generation gate or HF upload is required.

## 5. One common frozen sampling plan

Each arm consumes16,384 chunks across2,048 updates, effective batch8. Freeze one plan and use it in both arms. Sample each of the four existing teacher phases equally (4,096 chunks/phase); cycle through all93 TRAIN episodes equally within each phase. Phase is sampler metadata, not policy input.

Within each episode/phase, expose first and last eligible anchors, then cycle through up to8 nonempty rank/time strata of that phase; choose deterministic hashed order within each stratum. Do not consume only the first44 windows of a long phase. With the fixed allocation every episode/phase gets44 or45 visits and all its nonempty strata. Verify actual membership and missing phase groups before running. This covers a phase's time span, not every unique window or recovery state. No favourable-condition filtering, oversampling after results, or automatic focus switch.

Freeze a48-observation diagnostic card:24 TRAIN conditions from the existing four TRAIN initial episodes (six fixed phase/late anchors each), and24 VAL conditions from the first six admitted VAL episode IDs (one fixed valid anchor per phase). Resolve anchor ties by time then UID, before new scores. Never use VAL or confirmation to fit statistics or select data.

## 6. Matched adaptation arms and exact parameter scope

| Arm | Trainable learned content | Frozen content |
|---|---|---|
| INTERFACE2048 | Row10 only of state_encoder, action_encoder, action_decoder | Entire VLM, shared DiT, vlln, VL self-attention, position embeddings and all other category rows |
| ACTION2048 | The identical row10 set plus `action_head.model` shared DiT | Entire VLM, vlln, VL self-attention, position embeddings and all other category rows |

Both start independently from the identical pristine+untrained-interface snapshot, with fresh optimizer, the same samples and actual training-noise packets. ACTION is not a continuation of INTERFACE. Prefer ACTION first for the practical policy path; then run INTERFACE as the bounded parameter-scope comparison. Healthy arms are not restarted because an early screen fails.

The only intended between-arm adaptation change is whether shared DiT is trained under native train/frozen-mode semantics. Record dropout/mode differences if the native code introduces any; this is a comparison of adaptation scopes, not proof of an abstract capacity theorem. No copied DROID row or changed normalization is introduced in one arm only.

The inspected row sizes are1,710,592 state +7,285,248 action encoder +1,184,900 decoder =10,180,740 allowed-to-change scalars, in14 W/b slices. Verify these against actual headers. `tune_projector=True` alone also trains other rows and shared position embeddings, and is NOT a row-only implementation.

Use official packed projectors with explicit row10 gradient masks, fresh optimizer state and ZERO weight decay for the packed projector group. Enforce zero gradients/moments in other rows and prove byte-level immutability of all frozen rows. Whole packed tensors contain325,783,680 scalars: report actual optimizer storage, not just10M effective changes. No new row-leaf architecture is necessary. Never rely on gradient masking with nonzero AdamW decay to preserve other rows. All shared modules excluded from an arm must remain frozen and use the native frozen-module evaluation behavior.

Autograd must still pass through the frozen DiT to the trainable state/action encoders in INTERFACE. Do not wrap that DiT forward in no_grad. Frozen VLM feature computation may avoid parameter gradients. Canary must verify finite nonzero module gradients in all three interface modules and actual row10 changes; ACTION must also change the intended DiT parameters.

## 7. Native objective, learning recipe and short recovery proof

Only the official masked flow-matching objective:
`x_t=(1-t)*noise+t*target; velocity_target=target-noise; loss=sum(mask*(prediction-velocity_target)^2)/(sum(mask)+epsilon)`.
Keep the actual official time sampler, forward and action decoder. No differentiable free-running S4 auxiliary loss, physical XYZ/SO3/speed/increment loss, generated-gripper loss, box loss, stop token, phase token, output smoothing, RL or WBC gradient. Free-running S4 is for evaluation only. Do not import old pi0264/68/77 training overlays or `pi0260_r2_codec.install()` into this process.

Optimizer: torch AdamW, projector LR1e-4, projector decay0; ACTION DiT LR1e-5 and decay1e-5. Both use betas(.9,.999), epsilon1e-8, global grad clipping1.0, cosine schedule over2048 formal updates, warmup103 updates. State dropout0 for this declared deterministic baseline. Model trainable master weights FP32, native BF16 autocast, fp16=False, TF32=False, no custom R1 free-running training precision path. Source quality uses original BF16 S4 inference/FP32 decode. These settings are prospective choices, not claimed optima or exact official defaults.

Effective batch8. Prefer one legally allocated H200 with microbatch4 x accumulation2. If a prospective memory-only check fails, before either arm starts choose microbatch2 x4, then1 x8; at most32 discarded completed updates across all memory checks, no quality-based choice. Freeze the same layout/runtime for both arms. Do not change layout mid-arm, compare equal optimizer counts at unequal sample exposure, or reduce precision to fit. Corelab remains the evaluation location; using Corelab for training is allowed only if an identical common layout fits measured resources, not by silently freezing more layers. No new purchases or duplicate PBS requests.

Use seed namespace28203 with independent loader RNG and per-update/microbatch CPU+CUDA training RNG scopes, preserving actual native noise/time sampling. Same seeds alone are not evidence: save/check actual noise and time identities for both arms, without replacing the native target distribution. Capture native outputs passively; no teacher/success signal affects noise or routing. Resume restores optimizer/scheduler/RNG/data-consumption cursor; prefetch is not consumed data.

Each arm has one8-step continuous versus4+save+fresh-process+4 canary (16 completed discarded updates each). Inputs, actual noise, scheduler/cursor and frozen tensors must be exact. Record bitwise model/optimizer equality; numerical acceptance additionally requires all updated FP32 tensors allclose(atol1e-6,rtol1e-5), all nonfinite flags false, and no prediction/gate discrepancy on the fixed canary inputs. No tolerance relaxation after seeing results. Passing canaries automatically grants formal training within the same authority; do not stop at DATA_READY. Health128 is a checkpoint within formal training, not another restart or GO.

## 8. Evaluation, timing and fixed selection

Use the unchanged source Panda physics, original20Hz control/100Hz integration, first8/40=.4s replan interval and32s task length. Read new actual source images/TCP/opening each query. No invented source joints, time/phase/success input, teacher takeover, timed finish or stop-on-success. Original pose SafetyGate, physical controller and object scoring stay fixed. Stock inverse clipping is now explicitly part of this new processor; log pre-inverse normalized out-of-range fraction and postdecode values so it is not misrepresented as learned validity. No additional output patch is allowed.

The old PI0280/81 direct/joint results use a different contract and are contextual baselines, not a causal fresh-interface control. Evaluate UNADAPTED_INTERFACE0 once on TRAIN4 and DIAG48, labelled untrained-interface diagnostic. Then evaluate each arm at1024 and2048 on TRAIN4 and DIAG48. Only the2048 endpoints are eligible for DEV/selection. Evaluate BOTH endpoints on the fixed DEV8(30000..30007), even if an endpoint fails TRAIN, to separate fitting from same-task variation. No repeated same-checkpoint quality trials, inference-noise tournament or hidden early-checkpoint selection.

Same physical conditions use the same stored full native noise packet per query index across compared policies; each policy receives its own actual observations. Differences in visited states are part of closed-loop behavior. Preserve early gate/physics termination and count it in the original denominator.

Task metric is unchanged: complete valid32s plus original on_top AND static AND not_grasped at ALL41 control samples28..30s. Also report first success, sustained grasp/lift, placement, release, losses of success, raw/installed/achieved EEF, contact/IK and native clamp statistics. Do not call load/parity, low RMSE, grasp or transient placement a qualified policy.

Development qualification: the SAME endpoint has TRAIN>=3/4 and DEV>=6/8. If either qualifies, choose exactly one endpoint by DEV persistent count, then TRAIN persistent count, then prefer INTERFACE (less shared adaptation); no subjective video selection. Freeze model+processor+statistics+mapping+decode before opening CONFIRM16(30032..30047). Verify those reserved conditions have never been queried; if they have, preserve that fact and do not invent replacement seeds under this authority. Evaluate one frozen candidate once on16 conditions. Source prototype target>=12/16 persistent success. Four/eight developed cases and16 confirmation cases do not establish broad generalization; report binomial uncertainty and exact denominators.

If no endpoint qualifies, confirmation remains locked and both arms stop. If confirmation fails, stop without going back to train or testing the second candidate. No budgeted fallback4096 extension or automatic multi-task expansion.

To assess narrow pretraining retention, run each2048 endpoint on the32 existing official DROID source observations using ORIGINAL DROID processor/statistics/tag and the saved original noise, not the new EEF10 processor. Compare to the already saved pristine output:64 new calls total. INTERFACE should retain that route exactly if all shared/other-row values are frozen. ACTION differences are descriptive; 32 recorded observations are not a complete multi-task retention benchmark. These observations are evaluation only, not replay training data or a model-selection criterion.

## 9. Acceleration and local-first execution

Use one owner and at most three workers with disjoint responsibilities: DATA/CPU-cache and contract; LEARNING/source evaluation; independent AUDIT/WBC handoff. One training owner and reservation ledger. Same-GPU jobs serialize; parallel source evaluation uses an actually available separate allocation. Do not confuse three agents with three available GPUs, create competing goals, or operate unrelated jobs.

Reuse the proven two-worker/pinned-memory/ordered-loader pattern, not the old processed tensors. New processor/statistics/RGB/masks require new content-addressed cache keys and actual original-vs-cached parity. Cache deterministic CPU inputs, never stale observations, future RNG, or learned activations for a trainable network. Do not rebuild a large complete cache as a mandatory first-step barrier; lazy verified entries can feed training. Stop worker-count/compile/precision benchmarking. Report measured samples/s, optimizer steps/s, loader wait, forward/backward time and GPU memory after warmup. Native loss eliminates the previously added free-running S4 training unroll by design; actual speedup is not promised.

Reuse original base files locally. Save full local Trainer at1024/2048 and safe stops; keep latest/previous recoveries and source parent. Exact changed-tensor model-only transfer to Corelab is allowed with reconstructed full-state hash parity, not numerical residual addition or quantization. No optimizer transfer for inference. HF bulk upload/readback is DEFERRED and cannot gate training, physical evaluation or the decision. Native buffered logging may avoid per-scalar/per-record fsync, with ordered records, bounded flush, end fsync/count/hash and explicit crash-loss boundary.

Check actual user quota and atomic-save headroom; PI0281 quota was UNKNOWN. Shared df or a GPU-allocation upper bound is not free space or utilization. Reuse/link byte-identical base/cache objects. Do not delete unique paused recoveries, failures, datasets or broad historical directories. If resource limits prevent execution, preserve a concrete resource blocker rather than alter the scientific plan.

## 10. WBC disposition: handoff, not another upper-level compensation experiment

No new WBC actor call, target simulation, live target model query or motor command is authorized. Retain both native and once-repaired PI0281 WBC identities/results; do not turn the orientation gate off or add another progress patch. Package saved requested/installed/actual same-TCP poses, exact actor/normalizer/config/asset identities, native proxy versus true SO3, history checks, progress residual and representative traces for the senior researcher. Inventory any genuinely delivered upgraded version and its ABI differences read-only; new delivery is not silently substituted or automatically executed.

A source-policy PASS cannot remove the WBC qualification block. Conversely, the old WBC failure does not block source task adaptation or filter its training data. An EEF10 API removes the need to fabricate Panda joints structurally, but does not establish semantic or visual transfer to Go2. Only after separate target-controller qualification can a future frozen-policy composition protocol be issued. Old raw-model target route incompatibility remains a true baseline result.

## 11. New finite budget

Formal updates2x2048=4096; common effective samples2x16384=32768. Planned canary completed updates32; prospective memory checks cap32; same-run technical recovery cap64. **All newly completed updates hard cap4224**, formal healthy updates hard cap4096. Interrupted/rolled-back completed work still counts. Native training forwards are counted separately from free-running S4 quality queries; at the preferred layout formal native training forward batches=8192. No free-running training unroll is allowed.

Nominal maximum model-quality calls:5 policies/checkpoints x48 diagnostic calls=240;52 learner episodes x80=4160;64 source-retention calls; up to32 serving/parity calls: **4496**. Hard cap **5120 requests /20480 S4 head evaluations** including warmup, unsuccessful/rejected and technical replacements. Saved-output CPU comparisons are not fresh calls.

Nominal physical starts: UNADAPTED TRAIN4 +2arms x2milestones xTRAIN4 +2endpoint DEV8 +one CONFIRM16=52 learner episodes;4 expert codec replays; total **56**. Hard cap **60 starts**, at most4 same-condition technical replacements, zero valid-quality retries. Nominal100Hz integrations179200 and20Hz controls35840. Hard caps220000 total per-environment integrations,45000 control steps,45000 RGB pairs and90000 control/gripper dispatches; all reset/partial/pre-roll and any hidden vectorized environment work counts. One environment per source episode unless a separate identical-semantics batch implementation is proven within these same caps; no extra trials.

New compute cap12 GPU-hours total across authorized training and evaluation,96 CPU-core-hours. These are ceilings, not timing promises. New outputs/cache cap120GiB, with local quota headroom verified before saves. No new model/platform download; missing non-model dependencies cap5GiB. New teacher tasks/learner recovery production, target-WBC execution/training, multi-task collection, target VLA calls and hardware all0. Existing source record conversion for this new TRAIN contract and its statistics fitting are explicitly allowed; they are not new physical data acquisition.

PI0281 actual215policy/860heads/23starts/50301integrations and PI0280 actual304/1216/7 are preserved. New budgets do not reset or transfer unused old allowances. Old paused completion permissions are not reopened.

## 12. Delivery and stop decisions

Implement isolated `scripts/pi0282_execution/`; publish actual code/config/import manifests before GPU execution; write `reports/PI0282/`. Required steps: adopt -> actual data/native-contract proof -> common plan and interface initialization -> expert codec replay -> two native-loss canaries -> ACTION/INTERFACE matched training and fixed source panels -> endpoint retention/DEV -> one conditional CONFIRM -> local policy and evidence handoff.

Deliver actual loading/row provenance, legal input example, native statistics/clip audit, parameter scope and frozen-row proofs, shared data/noise consumption, full local recovery receipts, untrained/adapted task tables, original-DROID retention, failure events/videos, measured runtime, WBC handoff and final budget. No requirement for a second owner signature after an already specified gate. A model saved but never run physically is not the deliverable.

If INTERFACE succeeds, that supports reuse of a frozen shared trunk on this adapted single-task interface, not universal few-shot transfer. If ACTION succeeds while INTERFACE fails, this budget supports adapting shared action generation as well; it does not prove interfaces could never suffice. If both fail, separate native training fit, source execution, terminal behavior and observed-state coverage before the next intervention; do not automatically resume customized losses or multi-task training. If old WBC remains unqualified, report target composition still blocked regardless of source result. Final states separate EXECUTED, SOURCE_PROTOTYPE_QUALIFIED/NOT_USABLE, PRETRAIN_RETENTION_DESCRIPTIVE, and TARGET_BLOCKED. selected/deployment remainNONE.

## 13. Audit basis and limitations

Actually read at reviewSHA: PI0281 terminal comment, final REPORT/GAP_MATRIX; audit_wbc.py's same-TCP/actor reconstruction and gate; source_math.py's preserved20Hz gripper events; existing interface-preparation draft; current README. Official fixed source CategorySpecificLinear/action encoder, finetune config and native normalize/unnormalize were also read. The draft's proposed multi-task collection,15Hz resampling, two-frame history and row reinitialization are NOT inherited. This protocol deliberately chooses one task, existing20Hz current-frame data, unchanged original placeholder bytes and two matched scopes.

Reviewer independently checked scalar/visit/budget arithmetic, not private weight payloads, all actual cached tensors, raw physical arrays or every video. No SSH, training, new model inference, remote job action or simulation was performed by publication. Runtime/proof unknowns are explicit admission checks rather than claims of completed work. The fresh-pretrained comparison is authorized by this new decision; it is not an experiment already run.

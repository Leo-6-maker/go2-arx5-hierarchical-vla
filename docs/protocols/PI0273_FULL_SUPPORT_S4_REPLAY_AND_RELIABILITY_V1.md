# PI0273 — Full-support S4 replay, corrective fitting and reliable task feedback

Authority: **PI0273_FULL_SUPPORT_S4_REPLAY_AND_RELIABILITY_GO_V1**.
Date: 2026-10-05 SGT.
Review base: `8fed51eae2af34ec0002bc1635e26e12c7c215d7`, branch `pi0267/executor-decoupled-recovery`.
Predecessor: PI0272 V1 release `e62e6a469dce1f4d3df5d49b38bad4a3c627e969`, phase-specific R1 release `8a2ffbefe0c57536c0a7e340da0616c7987c74f1`, terminal Issue #11 comment `5981967270`.
This document and `reports/PI0273/design/REVIEW_AND_BUDGET.json` are bound to an immutable release by the new Issue #11 canonical. The canonical supplies that release SHA; publication is not a claim of runtime launch.

## 0. Decision and objective

GO for ONE matched S4 learning-and-feedback experiment: **BROAD5120 versus TARGET5120**, both initialized from the same original PI0270 AGG4096 WEIGHTS with fresh optimizers. Do not continue developing DIRECT in this goal. Do not deploy the small-set PI0272 S4FIT checkpoint as if its local fitting panel were full task qualification.

The question is whether interleaving the complete original training stream with the previously demonstrated amount of corrective-support replay can retain early behavior while improving original-start task completion and reliable holding. A same-update, same-context broad-replay arm controls for extra optimization and repeated scene/language exposure. The sole primary intervention between arms is the state/label source chosen for the registered supplemental replay slots. Architecture, teacher, input, loss, decoding, precision and executor are identical.

This is not another local-fitting-only goal: after both fixed endpoints, the owner MUST run the registered original-start M1/M3 feedback panels and component diagnostics, preserve safety failures, and report original/reliable/legacy qualifications separately. It is not authorized to stop at a lower offline loss or at training completion. Passing local directions or fitting cannot replace mainline evidence.

One Codex main owner executes serially. Helpers are optional, never mandatory approvers. Adopt already completed valid work and check current actual locks/jobs before launch. Do not cancel healthy work, submit duplicates, install a competing automatic goal, reset old ledgers, force-push, merge, add a permanent branch, create generic issues or restore a PR77 pointer. New source/artifacts remain under isolated `scripts/pi0273_execution/`, `reports/PI0273/`, `docs/protocols/PI0273_*` and PI0273 runtime roots. Never sync moving code into a sealed live runtime.

## 1. Evidence reviewed and limits

The reviewer refreshed the branch and Issue #11, read the complete PI0272 research return, final closeout and canonical, selected Trainer inventory entries and actual per-condition CSV rows, and examined the qualified PI0270 data loader and feedback/reporting code. The head commit is dated 2026-10-05 00:14:48 SGT; the terminal canonical is 00:13:18 and the closed receipt 00:14:00. These are repository evidence times, not reviewer SSH/PBS observations.

PI0272 is COMPLETE: both1024 endpoints,32 canary updates,2080 total completed updates,24987 training heads,2148 quality calls/6228 heads,1360 exact old outputs reused,3504 non-repeat metric rows, no technical recovery training or quality retries, four complete25-file Trainers and new-result readbacks. Missing scientific requirements are empty. The detailed report SHA256 is `b00f052f9fe05cc822de1654fe21f4bf91b40e0a84b80764301d40c3a6c3375f`.

M3 terminal wrong directions are AGG35/48, S4FIT0/48 and DIRECT0/24 final outputs. There are24 conditions from TWO TRAIN families, not48 scenes. Direct-minus-matched-S4 per-condition prefix MSE has median +4.59272529e-7 m2; only10/24 conditions and one of four family-command summaries favor DIRECT. A slightly lower marginal DIRECT XYZ median is not the predeclared paired criterion. Both readouts can learn these local corrections under extra fitting; this counters a uniquely DIRECT-required explanation but does not identify data quantity as the sole cause or prove all S4 tasks learnable.

The repair has an early-support cost. Original NEW_FIT192 prefix XYZ medians AGG/S4FIT/DIRECT are M1 2.701/4.264/5.026mm and M3 2.120/3.525/3.264mm. S4FIT is worse than AGG on76/96 M1 and70/96 M3 paired XYZ conditions. Those192 conditions had no exact input/label overlap with the PI0272 fitting set, although they were previously consumed by AGG. They are developed TRAIN-support diagnostics, not blind TEST or LEGACY. Some terminal teacher displacements are almost zero; continuous signed displacement and magnitudes must accompany direction signs.

PI0270's latest full task evidence remains M1 original32/32, M3 original2/32, reliable6/64, all64 safe, original-ROT-relative LEGACY PASS. PI0272 created no new episodes. Therefore neither 0 wrong directions nor a few-mm support error establishes reliable task behavior. The separate two-development-family M1 deficit, including t=0 observations, was not repaired or reevaluated by PI0272; do not attribute it solely to later action-induced visitation.

Interpretation motivating this goal: the observed narrow-support tradeoff makes joint full-stream/recovery replay a justified next test. The 4:1 mixing below is a fixed design choice, NOT an empirically optimal ratio. No ratio search, curriculum selection from EVAL results, extra data collection or automatic iterative aggregation is authorized.

## 2. A0 — source identity and reuse

Accept PI0272's closed `FINAL_CLOSEOUT_RECEIPT.json` and exact terminal5981967270; bind its report/archives and preserve the valid negative DIRECT-superiority result. Reuse closed PI0270/71/72 outputs and archive receipts. Do not regenerate them merely to enter this goal. Read current actual resource/job state; a specific new source/semantic contradiction blocks only the affected path. An unresolved unique neural root cause is not a launch blocker.

Common initializer for BOTH arms:
- HF repository `leoyuer/pi0246-gate-pass-campaign`.
- Revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`.
- Prefix `arms/PI0270/AGG/checkpoint-4096`.
- Complete tree SHA256 `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`.

This is deliberately the broad-support AGG initializer, not the early-regressed S4FIT1024 endpoint and not DIRECT. Load WEIGHTS only; new optimizer and5120-step scheduler per arm. Prior initializer optimization is disclosed, not recounted as this experiment.

Bind original PI0270:
- `PAIRING_PLAN.json` SHA256 `fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef`.
- `RECOVERY_MAPPING.json` SHA256 `53fe049e406f079e079d03af71e25c0ffd339b587fab7d8e5dc80bebe35fd633`.
- Actual AGG `sample_audit.jsonl` SHA256 `938d1514a372b48a506ab538d7aa206689bc73c7f9258af3c66b5274af9e5cbc` (8192 consumed microbatches).
- Effective ROT statistics SHA256 `0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9`.
- NEW_FIT192 SHA256 `553d254cea5ad5c32d0d8acf615f2abfe222459ac10e9d0bb0e09b786884d226`.
- Training-data archive revision `d16394d8e32d535465828ef5d572b491a373e759`, tree `fa95b40b358254e5a0196548c6c5e805edc500973670aed6389e88e56ba7fbde`; this contains recovery materials, not a claim that all older nominal sources reside in that tar.

The supplemental296 conditions are exactly PI0271 LATE_SUPPORT_CARD SHA256 `2899f730ef8f3fd1a76350662ec284237439ab7104192d9d6dfe2c30c503528e`, also bound by PI0272 FITTING_CARD SHA256 `02a77e4f30a7ddb6c2d4219bdc30418ff603b52b047568d1e0e48ff8161b6880`. They are a subset of actual AGG inputs/labels. PI0271 support/noise revision `5b5dc9bc65dec326c9f4d758a8920a4268d00a29` and PI0272 summary revision `4ae9b50af558fb438b1794e58e8bab592a11cf18` supply closed diagnostic identities. Reuse local sources; restore genuinely missing required artifacts only from exact immutable sources. No historical weight sweep or full optimizer copies to Corelab.

## 3. A1 — full-base stream plus matched supplemental replay

### Base stream B: identical and complete in both arms

Reconstruct the EXACT8192 processed-input/label occurrences used in PI0270 AGG main, ordered by original(unit,micro):4096 COMMON_NOMINAL plus4096 RECOVERY_INTERVENTION. Use the existing qualified nominal source records and frozen learner-state samples, language/template, masks, processor and labels. All original4096 optimizer-unit pairs and their within-unit order occur ONCE in each new arm. Preserve duplicate logical occurrences and disclose unique identities;8192 occurrences are not8192 independent states.

Do not substitute the older PREFIX recovery data for the AGG mapping. Do not use an old provenance anchor as the actual sampled state. There is no new teacher label, observation, state perturbation, statistics refit or augmentation. Old source hashes/codec checks remain binding. 'Full support' in this goal means this complete empirical AGG stream, not every conceivable recovery state or the entire historical project dataset.

### Target supplemental stream T: repeat the registered296 conditions

Sort the296 conditions by numeric original(unit,micro), as PI0272 did. For supplemental zero-based micro index j=0..2047, choose T[j]=row[j mod296]. This retains the successful local fitting EXPOSURE COUNT from PI0272:272 conditions are sampled7 times and24 sampled6 times. No selection by current loss, failure sign or success; all296 memberships remain. M1 has128 conditions and M3 168; components cover only two families. Publish exact task/family/command/language/bucket counts and distinct-state counts.

### Broad supplemental stream C: control for extra optimization and context

For each T[j], define key=(family,task,command,exact language string). From B construct an ordered pool of ALL occurrences with that same key, sorted by original(unit,micro). Do not filter this pool by error, phase, nominal/recovery status, inclusion in the296 set, or NEW_FIT membership. Preserve original occurrence multiplicities, rather than silently turning it into a uniform-unique-state pool.

For each key, maintain a separate zero-based occurrence counter n_key. Set C[j]=pool_key[n_key mod len(pool_key)] and then increment n_key. Copy the entire chosen observation/label/language packet unchanged; do not splice the target label onto a different state. Each pool is nonempty because T is source-bound to B; an actual contrary finding is a source-binding defect, not permission for an ad hoc fallback. Exact T/C coincidences are allowed and explicitly reported, not replaced to exaggerate a treatment contrast.

This is a CONTEXT-MATCHED broad empirical replay control, not a claim of globally uniform sampling. T and C have identical supplemental family/task/command/language order. Their state/label distribution, including source-phase and component-placement exposure, is intentionally different. That distributional intervention is not a single-axis perturbation or proof of one failure mechanism.

### Global training schedule: 4 base units, then1 supplemental unit

For block b=0..1023:
1. Execute base units4b+1..4b+4 in original order (8 microbatches).
2. Execute one supplemental unit from micro indices2b and2b+1: C for BROAD, T for TARGET.

Each arm therefore receives **4096 full-base updates +1024 supplemental updates =5120 updates**,10240 microbatches. Base10240-fraction is80%; supplemental20%. No original base input is dropped to make room. This DATA mixing fraction is separate from the unchanged80/20 TEMPORAL loss weighting.

The endpoint5120 is derived from one complete base pass plus the same supplemental dose as PI0272, not performance tuning. No additional pass, early stopping for favorable quality, staged endpoint selection or ratio sweep. Both arms use seed27303 and per-global-update seed=int.from_bytes(SHA256('PI0273:TRAIN:27303:{update_1based}')[:8],'big') modulo2**31. Preserve the qualified per-unit two-micro RNG lifecycle and zero-worker exact restore. Record and require paired actual native epsilon/tau and generated initial-noise identities; same seed alone is insufficient. Do not require old AGG optimization packets at the different global schedule to match; the historical match is DATA identity, and the new random-packet match is BETWEEN the two new arms.

Freeze B/T/C, both full schedules, all source and processed hashes, overlap and context-match receipts before neural training. CPU admission checks source encoding and every10240-micro logical schedule per arm without learned calls. Previously verified identical source packets may be encoded once/cached with explicit immutable dependency reuse; do not repeatedly encode the same296 samples only to inflate audit work. Actual training later must prove all10240 consumed entries, including8192 shared entries per arm. NEW_FIT192 is in the broad original bank and consequently again training-supported; never call it independent holdout. A3 development labels, protected EVAL labels, FACTOR and LEGACY evaluation rows are prohibited as newly imported training sources.

## 4. A2 — unchanged S4 learner, explicit phase precision and actual training

Both arms use the existing S4 readout, same AGG parameter names/shapes/trainable flags (471 trainable/559 frozen tensors), same condition-only RGB/body+wrist, actual StateV1 and language. No DIRECT branch, new head, history/phase token, source-ID feature, target oracle, backbone unfreezing, camera/encoder replacement, parameter averaging or online teacher.

Objective is the ACTUALLY bound qualified PI0270/PI0272-S4 objective:
`J = L_native + .4 L_position + .1 L_increment + .1 L_rotation + .1 L_peak_speed`.
Keep original native full-H40 supervision, chordal physical rotation loss, physical scales/masks, PREFIX pose weights(.8/9 on0..8,.2/31 on9..39), transition weights(.8/8 on0..7,.2/31 on8..38), and full39-transition peak-speed term. No loss-weight/normalizer/clamp/gripper change. Different empirical sampling is the authorized intervention, not a hidden new loss. Training uses four differentiable native S4 nodes with inherited mode restoration and checkpoint recomputation.

Carry PI0272 R1's VERIFIED phase boundaries, not its superseded V1 shorthand:
- Training/canary: original mixed parameter storage, trainable FP32, original frozen BF16/FP32; native BF16 epsilon/autocast; processed VL/query and Euler states FP32; physical losses FP32.
- Quality: original model-only BF16 loading and BF16 actual S4 query, fixed FP32 source-noise files; raw-to-FP32 physical decoding unchanged.
- Same-phase parity compares equivalent paths on the same packet/parameter-dtype/autocast state. Do not demand cross-phase bit equality or loosen same-phase tolerances. Record actual parameter, feature, epsilon, query, readout and Euler dtypes separately. Preserve Prescott source-encoder binding; capture actual loaded-library identity where available and disclose missing evidence literally.

Reuse the qualified PI0272 S4 training/resume path and PI0270 full input reconstruction through narrow PI0273 wrappers. Update only necessary new source-plan/arm/path/budget5120 guards; do not accidentally consume a1024/4096 hardcoded loop or inherit DIRECT hooks. Follow actual __file__ and source hashes; bind mature full native/S4/physical helpers, not merely same-name files at repository HEAD. Freeze source/config/mode filesets before actual use, allowing isolated preparation to continue without modifying a running seal.

Use fresh AdamW per arm: LR1e-5,warmup10,cosine5120,weight_decay1e-5,max_grad_norm1,microbatch1/accumulation2; no old optimizer/scheduler. Both real continuous8 versus4/full-save/fresh-process-resume4 canaries are required under the5120 schedule,16 updates per arm/32 total. Use already-required first backward to prove all4 generated nodes have finite nonzero gradients, native packet match, scopes and mode restoration. Verify model/optimizer/scheduler/cursor and Python/NumPy/CPU/CUDA RNG equality. Parity uses existing same-phase raw1e-4/decoded1e-4/XYZ1e-5m plus original-gate agreement, with no performance screening.

After both real canaries pass, run BROAD then TARGET serially, each5120. Preserve canary evidence; only the already authorized exact discarded-canary cleanup pattern may retire redundant payloads after durable proof, not unique main recoveries. Save complete2560/5120 Trainers per arm (four formal checkpoints), with optimizer/scheduler/RNG, exact model/processor/statistics/embodiment/decode/mode, both release/R1 lineage and PI0273 source/schedule provenance. Only5120 endpoints are quality evaluated; no intermediate checkpoint or lucky seed selection. Full fixed-revision file inventories/readbacks are mandatory for these new four artifacts. Model-only views go to Corelab.

Audit actual schedules, all shared source identities, supplements and actual packets, scopes, native/full physical losses, all4-node gradients/modes, updates and resources. A large finite loss or frequent clipping is not by itself a reason to change the recipe. A concrete corrupted source, unmatched actual packet, changed phase precision, nonfinite optimization or semantic/gradient failure is preserved and blocks only the affected continuation. Valid poor predictions are quality results, not technical retry candidates.

## 5. A3 — fixed endpoint evaluation, including original-start closed loop

Freeze the following memberships BEFORE new training. Exact reuse requires policy+processor/statistics+input+actual noise+decode identity, not just seed or point name. Use already frozen PI0270 scene/task/main/component/offline cards and PI0271 A1 j0/j1 inputs/noises. Code may extend isolated path/point/counter allowlists only. No new benchmark geometry, color/language sampling or factor rendering. All eight main families have been developed on and are NOT blind TEST.

| Panel | New per arm | Both new arms | Exact historical reuse |
|---|---:|---:|---|
| Original offline FIT224+FACTOR304+LEGACY440 |968 logical rows/up to968 queries|1936|AGG968; ROT legacy comparator and closed identities|
| Original NEW_FIT192 Q0/Q1 |192 queries|384|AGG192 and PI0272 S4FIT192 for descriptive context|
| Late/component support296 at j0/j1 |592 queries|1184|AGG592 and S4FIT592; do not add another K8 bank|
| Original-start main64,40s/max100 queries |64 episodes/up to6400 queries|128 episodes/up to12800 queries|AGG main64, not a newly rerun control|
| Original component32,8s/max20 queries |32 proposals/up to640 queries|64 proposals/up to1280 queries|AGG component32|

Total planned maximum NEW inference **17584**, new virtual episode starts192, new frame pairs14080. Prequery exclusions/early stops remain in logical denominators; they do not buy replacement cases. Derived combined tables should distinguish primary AGG+BROAD+TARGET panels from descriptive old-S4FIT context. Expected primary logical counts: offline2904,early576,main192,components96; the four-point late context table has2368 rows, with no claim that all four points have full main evaluation. Do not invent S4FIT main outcomes or compare its nonexistent reliable score.

Main execution uses fresh actual rendered RGB/StateV1 at each query, original stochastic native S4, full-H40 gate, then only0..0.4s of the accepted reference; save20ms installed/achieved states. H40 is40 poses0..1.95s at.05s. E0 and E1 advance by the unchanged implementations (E1 time constant.15s); they are mathematical EEF executors, not WBC or two robots. Use frozen main noise by family/task/backend/replan, including the old command-sharing semantics. Both policies encounter their OWN actual states; shared noise is not a claim of identical later observations. No teacher tracking, retiming, smoothing, target absorption, hold overwrite, successful-state freezing or oracle-success stop. Continue40s except original safety stop; failures retained.

Components use the original32 explicit starting placements per arm, with original eligibility checks. Preserve their TRAIN family/support provenance. They distinguish approach/entry/follow/hold but do not replace main qualification. Zero negative direction at consumed support is not a task PASS. The virtual gripper remains the nonarticulated fixed tool; retain predictions/gate without claiming grasp/contact capability.

For every new output save native S4 trace, actual noise/dtypes, full decoded output and all original gate outcomes. Do not drop finite WARN/REJECT, nonfinite predictions or invalid rotations. Journal attempts before running, partial-head counts literally; reuse completed valid output after interruption and resume only missing registered work.

## 6. A4 — qualification, causal interpretation and saved-array analysis

Primary endpoint is EACH NEW MODEL'S unchanged original AND reliable main-task qualification with full-point safety, not the296 fitting score. Preserve exact existing V3/scorer/reliability functions and source hashes. At least13/16 successes per task/backend and6/8 per command; reliability requires original completion by35s, full safe40s, and all251 ticks in35–40s inside35mm and.10rad. Do not add or remove post-completion reversal rules: preserve original first-completion semantics, report all later reversals/exits separately, and disclose what the current qualification does and does not guarantee. Full-H40 gates/speeds/collision constraints stay unchanged.

Legacy qualification remains against the ORIGINAL ROT2048, not a looser S4FIT or newly trained comparator: REJECT count no greater and XYZ/SO3 medians no greater than the source-bound original legacy values. Current reported ROT references are15.121mm/.046878rad and0REJECT; compare exact closed values, not rounded numbers. AGG's13.326mm/.024064rad positive retention is also reported but not substituted silently into the historical gate.

Report three separate dimensions: (a) original/reliable/safe/legacy candidate qualification, (b) supplemental-target intervention advantage over BROAD, (c) early/late support tradeoffs. A model can improve without being qualified; BROAD can qualify even if the targeted replay hypothesis fails. Do not insist TARGET win or alter the comparator afterwards.

Predeclared descriptive comparisons:
- Main: matched case successes/reliability and task/backend/command counts, TARGET-minus-BROAD and each minus AGG; summarize at the eight-family level. Single training seed and developed geometry limit inference; do not use correlated ticks/noise draws as independent trials or claim broad population causality.
- Early: original192 per-condition prefix/full XYZ/SO3 deltas, Q0/Q1 and command/family strata, direction and gate; compare each new arm to AGG and TARGET to BROAD. These are actual TRAIN-supported guards after the full-base pass, not independent holdouts. Report 0/positive/negative changes, not only a grand median.
- Late: all296 conditions, both exact j0/j1 and their per-condition mean MSE before aggregation; retain full/first/prefix/endpoint/suffix XYZ/SO3 and continuous correction vectors. M3 terminal24-condition/two-family results and /48 noisy-output denominators remain explicit. Compare to old AGG and S4FIT without pretending equal total training exposure across historical and new models.
- For the replay hypothesis, a favorable finding must include better terminal correction than BROAD/AGG WITHOUT merely trading away early precision or SO3/gate, and higher reliable task counts or qualification under the original rules. Report each criterion separately, not a newly fitted composite score. A fixed-ratio negative or mixed result is valid.
- Historical oldFIT224 must retain ACTUAL per-arm consumed/not-consumed identities; not all were in the old AGG stream. NEW_FIT inputs may not be relabelled 'unseen'. Diagnosis-only A3 labels remain excluded from all fitting.

CPU reconstruction covers every new S4/decode result and every new main/component trajectory, using bounded threads. Independently recompute physical/prefix arithmetic and entry geometry where supported; explicitly disclose shared V3/gate/executor functions. Reuse closed predecessor replay, never rerun old model calls for CPU comparison. Keep reference, installed and achieved errors/time indices separate. On saved main transitions, classify missing entry, path failure, late completion, first-success exit, final-window XYZ versus SO3 failure, reference-versus-tracking errors and post-completion reversals. Do not call offline reference accuracy formal closed-loop stability.

Produce saved-frame videos for the first fixed main family, ALL8 conditions for EACH new arm; include failures and late behavior, policy name, timestamps, VIRTUAL/NOT_WBC/NON_REALTIME. No new renderer/model calls for video. Early stops retain actual length/cause; do not select successful chapters or stretch them to imply40s. Existing predecessor videos need no remake.

## 7. Hard budget, resource and execution discipline

Old PI0270/71/72 ledgers and all previous FAIL/positive findings remain closed, not reset. New PI0273 limits:

| Consumption | Planned maximum | Hard ceiling |
|---|---:|---:|
| Official training updates |5120 per arm /10240 total|5120 per healthy arm|
| Real canary updates |16 per arm /32 total|32 planned|
| Total completed updates incl. technical recovery |10272|10528 (256 technical reserve)|
| Training head calls incl. recomputation/parity |nominal184896 plus bounded parity|196608; up to256 parity heads WITHIN this cap|
| New quality inference attempts |17584|18432|
| New quality S4 heads |70336|73728|
| New main/component episode starts |128+64=192|208 incl. technical recovery|
| New observation frame pairs |14080|15360 incl. technical recovery|
| New teacher labels/task-data collection |0|0|
| New WBC/dynamic physics/hardware |0|0|

The training budget is larger than PI0272 because it explicitly preserves the ENTIRE8192-input base plus the demonstrated2048-micro corrective replay dose in each matched arm. Do not secretly reduce full-base coverage to fit the old1024 budget or keep training until success. Technical reserves never buy new scientific conditions, ratio/seed sweeps, a third arm, changed precision or an extra endpoint.

Use one compatible serialized Hopper training allocation, then Corelab model-only inference serially, CPU bounded analysis. Check actual user quota, live competing jobs and peak checkpoint/staging/write overlap plus10GiB. Do not claim project credit from absent accounting; use actual scheduler acceptance. A queued healthy job is not permission for duplicate submissions. Genuine time/resource interruption permits source-bound exact recovery under the SAME counters; preserve first attempts and do not restart completed arms. No broad cleanup, repeated historical model downloads, email/monitor or unrelated-process action.

CPU implementation fixtures are allowed and labelled constructed, not real model results. Same-semantic import/path/serialization/kernel/lifecycle/counter/metadata fixes can proceed autonomously with narrow receipts. Changing sampling algorithm, weights/loss/input/teacher/statistics/precision/scorer/executor is a new scientific decision. Do not revive the already resolved PI0272 precision authorization blocker. Missing unique-root-cause proof or helper availability is not a reason to stop.

## 8. Preservation, finish and next decision

Deliver: source-bound A0; fullB/T/C cards and10240-micro exposure/overlap proofs; runtime/code/phase/source identities; both actual canaries; two5120 training audits with matched real packets; four complete2560/5120 Trainer inventories and fixed-HF whole-file readbacks; all fixed endpoint panels, reused outputs and literal failures; full numerical/trajectory replay; videos; new-result archive and fixed-revision all-file/member readbacks; actual cumulative budgets; one terminal research report and Issue #11 canonical with exact readback. Effective policy identity includes weights, processor/statistics, embodiment mapping, decode and native phase/readout contract. File counts follow actual required provenance, not a guessed old18/25.

Preserve old complete model/result readbacks by reference. Archive only NEW outputs and necessary source/proof-of-reuse manifests; restore missing fixed dependencies only as needed. Training completion alone is not experiment completion. If the next authorized operation is feasible, continue through CPU/video/archive/terminal without waiting for another routine GO. A concrete unavoidable block has an explicit partial return with exact completed/remaining work.

Terminal: **PI0273_FULL_SUPPORT_REPLAY_RELIABILITY_COMPLETE — STOP_FOR_INDEPENDENT_TRANSFER_OR_SINGLE_TARGETED_REPAIR_DECISION**.

If either arm truly meets original/reliable/safety/legacy gates, list its immutable identity as a reliable VIRTUAL research candidate; independent new scenes and executor/robot transfer still require separate authorization. Formal `selected_checkpoint=NONE` and `deployment_selected_checkpoint=NONE` remain throughout this execution goal.

If both improve support but not original-start reliability, do not automatically add more replay: use saved transition/entry/pose evidence to select one later coverage/conditioning/dynamics hypothesis. If TARGET loses early support, the frozen4:1 mixture did not solve the measured tradeoff; no in-run retuning. If BROAD is as good or better, retain that counterevidence. This goal does not guarantee a working system, equate mathematical executors with robots, or convert previous quality FAIL/WBC PARTIAL into PASS.

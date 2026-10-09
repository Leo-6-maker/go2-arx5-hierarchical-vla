# PI0286 — bounded corrective fine-tuning, then reporting at 13:00 SGT

Authority: **PI0286_1300_T1_ADAPTATION_THEN_REPORT_GO_V1**.
Review base: `adcb9fa561fe6744936f36cc8d711d658a1964e2`, existing branch `pi0267/executor-decoupled-recovery`.
Predecessor: PI0285 R3 scientific terminal #6070372729 and final delivery #6070882425; report at `c2ed223b9f677f1ec4a3a3753a420625490144f6`.
Activation requires this protocol plus `reports/PI0286/design/PLAN_AND_LIMITS.json`, a fixed-release Issue11 canonical, and readback. Publication is not a remote job start.

## 1. New permission, not a rewrite of the failed confirmation

The user explicitly authorizes further fine-tuning until **2026-10-09 13:00 Asia/Singapore**, followed by summary/reporting. PI0285 remains COMPLETE / VALID_CONFIRMATION_FAIL: PLACE4096+DOSE2048, TRAIN7/8, DEV13/16, original CONFIRM_A20/32 against26/32. Its6144 formal updates,32 canary updates,6901 quality requests and257 physical starts remain sealed. There is no4/32 final confirmation result. Old unused B/shared allowances are not inherited.

This is a NEW bounded T1 follow-up with separately accounted work. The old10:00 deadline is not silently extended inside PI0285. One owner may adopt the new authority and necessary stopped-process runtime readers; do not overwrite old STOP or qualification receipts. Check current job/PID state first and avoid competing work. No remote process was operated by the reviewer.

Goal: improve and independently reassess the existing real T1 manipulation capability, not train again from scratch or silently turn a development candidate into a qualified policy. WBC, target simulation/queries, Go2, hardware and joint-action supervision remain0. T2/shared learning is DEFERRED for this short window even if T1 passes; protect the first result and prepare the meeting report. No new scene, task, VLM unfreeze, history input, normalizer, auxiliary loss or sampling-step search.

After13:00, only saved-data CPU recomputation, plotting/video packaging from recorded frames, checkpoint/file integrity checks, source review, GitHub/report writing and optional archival are allowed. **No new neural forward/backward, optimizer update, teacher rollout, rendering or physics step after13:00.** No automatic extension; time and resource caps jointly bind.

## 2. Evidence and the limited hypothesis

The latest32 confirmation cases contain12 failures: six miss at least one grasp/lift/transfer requirement and six complete those histories but fail later task/hold; seven have IK holds and five do not. None of the12 is a SafetyGate rejection. These observations do not establish a sole precision, controller or data cause. The model has substantive ability, but20/32 is not the registered qualification.

The completed optional phase was DOSE: original expert replay only. Corrective learner-state collection was never executed. Existing WP6 is prospective infrastructure, not an admitted recovery bank. Reviewed `r3_recovery.py` is hard-bound to PLACE4096/Adam4096 and old32 slots; it cannot simply be invoked against the final DOSE model. Reviewed `teacher.py` always initializes approach with an open gripper; naive takeover can disrupt a held object. This protocol explicitly permits the narrow new offline takeover initialization below, rather than pretending it is already physically validated or a same-semantics scientific fix.

Working hypothesis: a small amount of correct supervision from actual learner states may improve robustness while original expert replay retains the learned skill. This borrows the dataset-aggregation motivation of Ross et al.2011; this finite stochastic flow-policy experiment does NOT inherit DAgger's theoretical guarantees and has no matched-update causal recovery control. A readiness-based replay fallback is explicitly registered, not an unbounded alternative-model search.

## 3. Parent, immutable interfaces and adoption

Only parent: final PI0285 DOSE2048 / REPAIR_A2048-2048, **global Adam6144**, full14-file Trainer tree `fb17b65effc3ae8dcbefa784adc8f7e6fab916821d29e4b2539e5be685ed8469`, fixed HF revision `75432eb91a2c602930a9e1e979aaebbeaebff02b`, prefix `arms/PI0285/REPAIR_A2048/checkpoint-2048`, repo `leoyuer/pi0246-gate-pass-campaign`. Use the verified local full Trainer first; Corelab model-only is not an optimizer recovery. Do not download all historical weights or reset to pristine. The inherited VLM/action network remains a descendant of the genuine official pretrain; preserve both pristine reference and old DOSE artifact.

Production identities remain: statistics SHA256 `c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81`; processor `fae35db9d84083cc1faac16442c062c74a22691fa3358cfc58efb759e01977d3`; mapping `fb47934b4bd4b5e11ec1c61ade0c5e03ca9110ec5f16abce9633b10cfd9edfbb`. Same genuine current body/wrist RGB, measured EEF/gripper and T1 instruction; native relative EEF/absolute gripper, H40, active10/132, K4, execute first8 at20Hz, fresh feedback every0.4s,100Hz source physics. Same T1 scene/scorer/safety/native Panda PD. No task IDs, joints, object truth, teacher phase or success signals in policy input.

Retain PI0285 scoped admission and known DEV clipping cases. TRAIN and ALL new supervised correction windows require original finite/nondegenerate checks and roundtrip XYZ<=1e-5m,SO3<=1e-4rad,gripper<=1e-5. DEV evaluation always uses raw teacher targets, never silently clipped target replacements. No fitting on DEV or any confirmation. Do not rescan76928 windows solely to reproduce existing evidence when exact bindings remain valid.

Adoption card binds full parent, remaining resources, real source/host paths and new deadline. Prior scope/carry proofs may be reused when mathematically identical and actually identified; new-data/cache/consumer readers must be tested. At most one new continuous8 versus4+save+fresh4 carry proof (16 discarded updates) is allowed if required; no repeated canary per checkpoint. Verify actual parent moments, new phase LR schedule, input/noise/cursor and frozen-row integrity. No fabricated prior-proof reuse.

## 4. One time-boxed corrective-data attempt, TRAIN only

Immediately after adoption, prepare the dataset and immutable evaluation path in parallel. Data gate closes at **10:55 SGT**; do not delay useful training beyond it. No new parent rollout or teacher branch starts after10:45. If adopted too late or unavailable, mark preparation NOT_EXECUTED and use the replay fallback, subject to training/evaluation time feasibility.

Use exactly these eight EXISTING supervised TRAIN layouts, deliberately separate from the frequently screened first eight:
`285108,285116,285124,285132,285140,285148,285156,285163`.
They remain TRAIN, not a new test set. Run the frozen parent once per layout for the complete original32s, fixed original noise rule with prebound noise bytes and no retries by quality. Record real commands, physics and camera/state. If a byte-identical valid parent result already exists, reuse it. Total new parent rollouts<=8 /640 neural requests.

For each layout take actual states at **task1s and6s**,16 requested branches. Selection is fixed before seeing scores; do not substitute old confirmation failures or easy states. An unavailable/aborted state counts in16. Restore by actual installed-action prefix replay using recorded initial state and compare physical state to1e-6 and branch RGB/StateV1 byte-exactly as the inherited verified routine; no teleport-to-goal. Replay plus suffix may share one environment start; at most one start per requested branch here. No alternative restore tolerance or hidden simulation pass. All prefix steps count.

Create a new isolated **offline takeover adapter** to the existing teacher:
- initialize command from the actual compatible branch state/reference, record it and maintain existing0.15m/s,1rad/s proposal limits and native gate;
- if the object is currently genuinely grasped, keep grip closed; use actual TCP orientation/object-to-tool relation, populate the actual destination and enter lift or carry according to actual height. Never reopen solely because the constructor defaults to approach;
- if released, supported and already in the correct destination, enter collision-checked withdrawal/hold with a fixed terminal reference, not repeated resets to achieved TCP;
- otherwise use existing geometry-aware approach/regrasp from actual object state. Unsupported/fallen/unrecoverable states are failed branches, not object pose resets;
- privileged state and history are used only OFFLINE. This is a newly declared data-teacher intervention, never a runtime model assistant.

Freeze adapter code/rules before the16 outcomes. CPU fixtures test empty gripper, held low/high object, placed object, wrong goal and missing state; these are not physics evidence. Use the first fixed requested branches as the physical check, not extra pilot trials. Do not search multiple teacher variants after valid failures. Keep original dynamics, scene, contact restrictions, scorer and original total32s end. A valid combined parent-prefix/teacher-suffix must complete original T1 success; this is teacher-supervision admission, never policy success.

Store ONLY fresh corrected suffix observations and actually sent commands for gradients. Failed learner prefixes are not positive labels. H40 cannot cross the takeover/reset boundary or contain invented padding. Keep command-producing phase distinct from post-transition phase for new metadata. Full source history and every rejected branch remain available for audit.

Admission requires>=12/16 distinct valid corrected branches, >=5/8 at EACH1s/6s stratum, covering>=6 layouts, physical success plus strict frozen-codec fidelity. A successfully corrected trajectory with any out-of-contract supervised window is retained raw and makes the correction recipe inadmissible for this goal; do not remove hard windows, expand stats or normalize with DEV. Exact duplicates do not increase the count. Also require the proposed event/phase sampler can consume the admitted set without inventing missing behaviors.

At or before10:55 write **ONE TRAINING_RECIPE.json BEFORE the first formal update**:
- `CORRECTIVE2048` when the real bank and new input/consumer checks pass:75% original64 T1 expert episodes,25% correction suffixes;
- otherwise `REPLAY2048`:100% the original64 T1 expert episodes, with explicit READY_TIME/TEACHER/RESTORE/CODEC failure reason and preserved attempt ledger.
This fallback is specifically authorized here, superseding the old no-fallback rule only for this new goal. No model has been retrained yet, so this is readiness selection, not quality-driven candidate search. Once training starts, the dataset recipe is frozen: no switching later and no second recipe. A failed correction attempt is still useful diagnosis, not discarded evidence.

## 5. One bounded fine-tune, not another training ladder

Formal updates<=2048, effective batch8 (<=16384 draws). Initialize from the full parent and carry Adam moments at6144. Explicit new cosine2048/warmup103, peak interfaceLR5e-5 and DiTLR5e-6; AdamW/betas(.9,.999)/eps1e-8, interface decay0,DiT decay1e-5,gradclip1. Native masked flow-matching ONLY. FP32 trainable masters/BF16 autocast, BF16 K4/FP32 physical decode, TF32/FP16 off. Train row10 interface plus sharedDiT; VLM/other31rows/position/vlln/VL attention remain frozen. No new LR, K, precision, loss or state-dropout sweeps.

Original expert sampling retains25/25/25/20/5 behavior weights and declared event coverage. For corrections, first balance branch IDs, then available nonstatic behavior buckets; each branch reserves at most5% of its draws for static hold when present, redistributing unavailable buckets only inside that branch and logging achieved counts. This explicit source-specific sampling avoids making a late suffix pretend it contains a new approach phase. Native H40 loss remains unchanged. At full CORRECTIVE2048 the expert/correction counts are12288/4096. Freeze the complete draw stream, noise/time rules and cache identity, including instruction and source. No failed UID oversampling based on confirmation results.

Save immutable full recoveries at1024/2048 and safe deadline stop; retain optimizer on training host. Eligibility for quality screening is only1024 and2048. If deadline forces an intermediate stop, save it as unscreened/partial rather than automatically opening a third selection checkpoint. Progress, failed attempts and consumed unique states are logged. No requirement to use all updates once a candidate is frozen.

Readiness and legal allocation must support completion of a fixed endpoint plus required evaluation before13:00. No new training lease after11:15; no new optimizer update starts after **12:00** (earlier if measured evaluation reserve requires). H200 preferred; after15min queued, an actually available equivalent Corelab allocation may replace the unstarted queue request, with no duplicates/unrelated cancellation. Do not silently reduce scope or dtype to fit. Reuse validated4x2/two-worker layout; a known validated2x4/1x8 fallback may be fixed before first update only.

## 6. Development, fresh confirmation and leakage safeguards

Reuse parent results7/8 TRAIN and13/16 DEV as historical references without rerunning the old model. At eligible1024 and2048 endpoints run unchanged T1 fixed TRAIN8 (285100..285107) and, when the TRAIN gate passes, all original DEV16 (285200..285215). The final2048 DEV panel is reported even if TRAIN fails when time permits; omissions are explicit. No more than two screening rounds. Reuse old fixed DIAG96 inputs/noise but generate real new predictions, with raw-target versus codec bias separated. Dynamic and terminal metrics are separate.

The FIRST new checkpoint with TRAIN>=7/8 AND DEV>=13/16 is frozen as the one candidate; stop further training safely and retain any already completed in-flight work. Only then open **FRESH_CONFIRM32 layout IDs286300..286331** under the same frozen T1 layout generator/distribution and scorer. Before training, check identifier exposure in retained logs; if any were used, mark fresh confirmation blocked rather than picking another convenient range. No rendering, teacher query, input probe or model run on these layouts before candidate freeze. Original285300..285331 remains a FAILED, EXPOSED confirmation set: read-only diagnosis/reporting only, never training, new model selection or an independent re-test. Original285400..285431 shared confirmation remains untouched.

Fresh confirmation is exactly32 conditions once, pass>=26/32 with unchanged full32s and all41 hold samples28..30 plus grasp/lift/transfer/place/release/withdraw and green-object rules. Keep every valid failure. Any confirmation failure stops learning/model/physics immediately; no second candidate or second fresh set in this goal. Incomplete confirmation by12:55 is NOT_CONFIRMED, not a smaller denominator PASS. The source milestone may be positive while selected/deployment remainNONE. A new26/32 versus old20/32 on different layouts is NOT a paired causal improvement estimate or a population-success guarantee; disclose adaptive development.

No T2 acquisition, SHARED training or extra language/DROID probes in this window. Preserve any new qualified T1 independently before report preparation; otherwise protect the old DOSE and the newest complete research model without declaring either qualified.

## 7. Timing and reporting switch

All times are2026-10-09 Asia/Singapore. These are operational cutoffs, not time-to-success estimates.
- Adoption: verify stopped old execution and immutable parent; check disk/output paths and a real local transfer/evaluation path; begin finite data preparation only if feasible.
-10:45: last start of a new parent collection episode or corrective branch.
-10:55: freeze the sole CORRECTIVE or REPLAY recipe; unfinished optional preparation no longer blocks training.
-11:15: last new training-job launch. Do not begin work that cannot reach at least one eligible endpoint plus complete validation within the boundary.
-12:00: no new optimizer updates; finish the in-flight atomic update, save safely; eligible candidates can be frozen/confirmed earlier.
-12:55: all model/teacher/render/physics work stops. Finish saving partial records, not unrecorded new physics. Reserve at least45min for a candidate's32-case confirmation plus measured load/save/CPU time when not already completed; use longer measured1.25xp90 latency reserve if needed. Do not reserve unopened hypothetical stages so aggressively that no useful training runs.
-**13:00: enter REPORTING_ONLY**. No new experiments, not even called demo or validation. Fresh qualification pending CPU recomputation remains pending until completed from already saved arrays. Summaries/archiving may continue, but no new automatic training day/goal.

Read-only report deliverables under reports/PI0286/final/: (1) FINAL_REPORT.md answering current capability/gap and any real improvement; (2) MEETING_BRIEF_ZH.md with a one-page senior-researcher summary; (3) RESULTS_AND_FAILURES.csv with per-condition status and first divergence; (4) MODEL_LOAD_RECIPE.json and immutable model/processor/qualification identity; (5) BUDGET_AND_STOP.json and actual saved-video index. Separate facts, hypotheses and future work. Include at least fixed first actual success/failure videos if present, no cherry-picked replacement or synthetic footage. Explain pretraining scope, source-domain restriction, WBC freeze, old and new confirmation, and any data/optimization/engineering bottleneck. Human-readable report sections may be drafted during training from existing facts; blank result fields stay pending.

## 8. Resource budget and engineering protection

New PI0286 ledger, not reset/borrowing of PI0285. One owner, up to3 workers: TRAIN; SOURCE/EVAL; CPU/AUDIT/REPORT. One training lease; actual independentGPU resources only. No more five-way optional research while under the3-hour practical window.

New caps: formal2048; canary16; memory16; completed rollback32; totalcompleted2112; technical training replacements<=2. Maximum8 parent rollouts +16 prefix/teacher continuations +48 candidate screens +32 fresh confirmation =104 nominal physical starts; <=8 same-condition engineering replacements gives112 hard. No valid quality retries. Quality<=8192, K4 heads<=32768; controls<=80000,100Hz integrations<=400000,RGB pairs<=80000,dispatch<=160000; teacher decisions<=20000; GPU<=8 hours across devices; CPU<=64 core-hours; new owned outputs/cache<=80GiB. Latest old totals6144 formal/32canary/6901queries/257starts are cited separately. Any existing same-identity result reused is not charged as new work.

Observed prior SSD ENOSPC makes output reservation a real gate: estimate remaining raw episode bytes, model view, two full recoveries and safety margin on the ACTUAL destination; use approved writable HDD/local durable tier if necessary. Protect parent, unique evidence and all confirmation data. Only exact inventoried redundant files may be retired after verified recoverability/consumer checks; no broad directory deletion or unsupported quota claims. At first write-capacity failure, preserve incomplete records and do not let repeated jobs fill disks. All technical replacements require original scientific input/noise/state identity and budget.

Local immutable changed-tensor/model-only transfer and full reconstruction checks; no HF upload/download feedback roundtrip. No new full pretrained download, no migration to another benchmark, no huge canary archiving. Restore/data/mapping/scope receipts from the same actual semantics are reused; necessary new-data tests remain. Detailed audits first8/every128/checkpoints plus normal finite/global-grad checks, not full tensor copying every step. Input/noise/time and losses remain real-accounted. Actual TF/mode/carry differences are not silently tolerated.

## 9. Interpretation and sources

This plan changes new-state supervision and adds optimization, not a clean experiment proving one unique cause. Recovery is not guaranteed to help. If only REPLAY runs, report it as a final bounded dose check, not evidence that recovery worked. If both development and fresh confirmation fail, report the capability20/32 previously observed alongside the new failure, not no learning ever. If data readiness failed, distinguish teacher/representation/engineering failure from neural quality failure.

Reviewed: PI0285 final report; terminal/load identities; WP6 BRANCH_INTEGRATION_AND_CARRY.md, RECOVERY_READY.json; scripts/pi0285_execution/r3_recovery.py and teacher.py; latest branch and Issue11. Literature motivation only: Ross, Gordon and Bagnell2011, https://proceedings.mlr.press/v15/ross11a.html . No literature-based convergence/sample-count promise.

Startup: read this fixed release/canonical, preserve PI0285 failure, adopt full DOSE6144 parent; try the fixed16 TRAIN-only corrections before10:55 and freeze one2048 recipe, otherwise explicit replay fallback. Use native unchanged interface and one lease, screen only1024/2048, one fresh286300..331 confirmation. Stop updates12:00, neural/physics12:55, report-only13:00. No WBC/shared/old-confirmation retraining; deliver real result and loadable research/qualified artifact with honest status.

# PI0270 — Final research return

Authority: `PI0270_LEARNER_STATE_RECOVERY_WITH_REUSED_CONTROL_GO_V1`, release `7a6eea5ae38854bec7b80782213a92ca02fd0318`, Issue #11 canonical5969189182. Continuation/preservation is under PI0271 release `cbdcaf85ab6c25999fe054bf5da5285fadc394a3`, canonical5977478618.

**Scientific execution completed and budget/numeric audits passed. Quality: VALID_QUALITY_FAIL.** Original main qualification FAIL; reliable main qualification FAIL; donor-relative LEGACY retention PASS. There is no qualified reliable virtual research candidate. Final summary archive and publication identities are recorded separately in the final closeout receipt; this report does not substitute for their readbacks. Formal selected/deployment remain NONE.

## 1. What changed and what was actually executed

AGG starts from the same fixed ROT2048 weights as the closed PREFIX control, with fresh matched optimizer and the inherited4096-unit plan. All4096 COMMON microbatches remain bit-identical. Only4096 recovery microbatches change to coherently matched learner-state RGB, actual StateV1 and same-state correction labels from the fixed collection. Teacher, camera, language schedule, effective ROT processor/statistics, native BF16 S4, FP32 unclamped relative SE(3)/row-rot6d decoding, fullH40 SafetyGate, loss/temporal weights, precision and trainable scope remain fixed. The source intervention combines naturally visited and explicitly placed TRAIN component support; it does not isolate those two sources.

The fixed collection executed192 TRAIN episodes/19200 queries. All6784 teacher proposals are retained:6782 eligible and2 visibility exclusions. All96 family/task/command support cells passed, with at least62 distinct eligible states each. The immutable mapping selects4096 distinct recovery sources once, leaving2688 proposals unused. Source buckets Q0/Q1/Q2/Q3 contain1032/1028/996/936 records; M1_TERMINAL/M3_ENTRY/M3_MIDARC/M3_TERMINAL contain32/24/24/24. Quartiles are query-time buckets, not teacher phases. NEW_FIT192 selects the first distinct Q0/Q1 conditions only.

One actual8 versus4/save/fresh-process-restore4 canary passed exact model/optimizer/scheduler/cursor, modes, packets and Python/NumPy/CPU/CUDA RNG checks. AGG completed4096 official updates plus16 canary updates,74040 training heads, zero technical recovery updates. All8192 actual microbatches match the CPU schedule; all4096 COMMON inputs/labels and all8192 actual native epsilon/tau/stock-noise packets match the reused control. All192 NEW_FIT conditions were consumed. Full four-step gradients, loss application, clipping/mode and frozen scope have actual audit evidence.

The100-file small training evidence package contains raw journals, canary and training logs with verified tree `01d1299386a5d8c41eafb26b702e4fe4fc617c934f39e8eb35f56901ab885e15`. No optimizer/weight payload was copied to Corelab. Earlier CPU encoding differences were reconciled by the actual Prescott OpenBLAS kernel without changing statistics, labels, precision or tolerances.

## 2. Complete evaluation and exact reuse

| Panel | Logical scope | New work | Exact reuse |
|---|---:|---:|---:|
| Standard offline |2904 rows|968 AGG|1936 ROT/PREFIX|
| NEW_FIT192 |384 rows|192 PREFIX +192 AGG|0|
| Original-start main |192 episodes|64 AGG|128 ROT/PREFIX|
| Component |96 proposals|32 AGG, all VALID|64 ROT/PREFIX|

All first-valid predictions, gates and failures are preserved. CPU independently replayed8392 new quality S4/decode arrays, while the already-closed19200 collector predictions and old donor/control replays were reused. All64 new main and32 component achieved/installed trajectories were reconstructed at matching saved ticks; entry and physical metrics were independently checked where supported. Original V3/V2 scoring and executor arithmetic remain shared frozen functions, not independent physics validation.

Two reporting-only integration faults were repaired: the standalone CPU entry lacked qualified GR00T import paths, and typed reused CSV retention fields were Python dict literals while new rows were JSON. An isolated launcher restores the existing worker paths without calling its model; the analysis reader safely parses both saved representations. Original sealed code, arrays, metrics and qualification rules were unchanged; no scientific query or training update was repeated.

## 3. Fitting and old-domain retention

These are medians of full-H40 `xyz_rmse_m` and `so3_rmse_rad`, not prefix errors.

| Point | OLD_FIT224 XYZ mm / SO3 rad | FACTOR304 XYZ mm / SO3 rad | LEGACY440 XYZ mm / SO3 rad | Legacy REJECT |
|---|---:|---:|---:|---:|
| ROT2048 |9.939818 / .036871|10.977004 / .037925|15.121491 / .046878|0|
| PREFIX4096 |7.467644 / .018446|9.105805 / .020922|17.586511 / .067752|0|
| AGG4096 |7.123958 / .044186|6.732815 / .038281|13.326206 / .024064|0|

AGG passes the original ROT-relative legacy standard for both median errors and REJECT count. The old-FIT SO3 regression versus PREFIX is real but does not imply LEGACY failure. Only32 of224 old-FIT logical rows have exact actual AGG consumption;192 do not. On those32 consumed rows, AGG fullH40 XYZ/SO3 medians are4.902754mm/.009774rad; on192 not-consumed rows they are7.496741mm/.045592rad. These strata are recorded for all three points.

| NEW_FIT task,96 rows per point | PREFIX full XYZ mm | AGG full XYZ mm | PREFIX prefix XYZ mm | AGG prefix XYZ mm | Paired AGG-minus-PREFIX full / prefix mm |
|---|---:|---:|---:|---:|---:|
| M1 |33.423881|8.383060|10.936421|2.701186|-22.875615 / -8.508847|
| M3 |11.608396|6.468402|3.194467|2.119578|-6.484017 / -1.548816|

FullH40 SO3 improves .059010→.017970rad for M1 and .056336→.016493rad for M3. Paired fullH40 SO3 deltas are-.044709 and-.033326rad. Prefix, suffix, endpoint, command/cell and per-family changes remain in the complete tables. Marginal-median differences are not paired-median changes or significance tests.

Q0 and Q1 differ: M1 fullH40 XYZ medians are6.617→7.270mm at Q0 and60.988→9.822mm at Q1; M3 is8.145→9.126mm at Q0 and21.214→4.923mm at Q1. Thus the aggregate position improvement is not uniform across these buckets. These results do not establish fitting quality of the1932 consumed Q2/Q3 records or104 component-source records omitted by NEW_FIT192.

## 4. Original task acquisition, reliable completion and safety

| Point | M1 E0 original/reliable /16 | M1 E1 original/reliable /16 | M3 E0 original/reliable /16 | M3 E1 original/reliable /16 | Safe /64 |
|---|---:|---:|---:|---:|---:|
| ROT2048 |11/0|11/0|0/0|0/0|64|
| PREFIX4096 |15/0|15/0|0/0|0/0|64|
| AGG4096 |16/1|16/3|1/1|1/1|64|

AGG original command counts are M1[8,8] in both backends and M3[1,0] in both. Reliable command counts are M1 E0[1,0], E1[1,2], and M3[1,0] in both. Each task/backend still needs13/16 and each command6/8. AGG improves reliable count from0 to6/64, but every reliable subgroup fails qualification. All192 logical episodes run40s safely; no gate refusal, dangerous collision or execution stop is hidden.

For AGG M1,29/32 cases leave the pose region after first completion at some point; some can subsequently return. Median completion-to-first-exit is11.98s in E0 and17.05s in E1. During35–40s, position bounds fail9/16 and7/16; rotation bounds fail15/16 and12/16. Counts overlap. Only1/16 and3/16 satisfy the complete reliable definition. Final achieved XYZ medians are39.735mm and32.024mm; a final median does not replace all251 hold-window ticks.

M3 retains28 MISSING_ENTRY and2 TASK_CRITERIA_FAIL cases. The four entered cases are family031; c0 reaches all12 ordered bins and succeeds in E0/E1, while c1 reaches10 bins and fails with shortcut/reversal evidence. No failed entry is rearmed or its angle origin reset. The two original successful c0 cases complete at15.64s/26.04s; their first later reverse occurs at18.62s/27.58s. Frozen V3 first completion and full40s aggregate reverse are different recorded quantities: later reversal is retained, not erased or used to rewrite historical first completion. `M3_TIMING_INTERPRETATION.json` binds these actual timings. The unchanged reliability definition separately tests original completion by35s and the35–40s pose window.

## 5. Reference, installation and achieved state

`SAME_TICK_REFERENCE_INSTALLED_ACHIEVED.csv` contains6400 actual AGG queries, with original model0.4s reference, installed reference and achieved state at the matching endpoint tick, including XYZ and SO3. Rejected/unexecuted prefixes never become achieved predictions. Static task terminal distance is distinct from teacher-path progress; differences between unrelated minimum distances are never called tracking error.

E0 same-tick XYZ/SO3 tracking is exactly zero in all32 AGG episodes, yet most reliable task conditions fail. E1 episode RMS tracking has median1.502697mm/.002263rad and maximum11.243078mm/.009518rad. Tracking error is therefore not a necessary explanation for E0 failure; this does not identify a unique neural cause or establish WBC behavior. Complete source/current/model/installed/achieved transition rows preserve later drift and failures.

## 6. Component diagnostic behavior

All32 new proposals are VALID, full8s/20-query executions. These are the first two TRAIN geometries, with explicitly placed initial states. They are not original-task acquisition or independent transfer. The learning data include component-source support; exact diagnostic-input exposure must not be inferred solely from shared geometry.

| Start,8 cases per point | PREFIX / AGG goal-pose dwell median s | PREFIX / AGG final XYZ median mm | PREFIX / AGG final SO3 median rad |
|---|---:|---:|---:|
| M1_TERMINAL |3.25 / 8.00|65.051 / 24.692|.130486 / .060427|
| M3_ENTRY |0 / 0|77.201 / 84.395|.014942 / .028493|
| M3_MIDARC |0 / 2.05|67.901 / 24.566|.025583 / .071035|
| M3_TERMINAL |2.07 / 7.51|65.346 / 35.538|.020887 / .026442|

Terminal and midarc behavior improves on several metrics, while entry remains poor. Median dwell is not an all-case hold PASS, and angle-only component bins are not original ordered270-degree task qualification. All component denominators, exclusions, reversals, radial departures and stop fields remain in COMPONENT96.csv.

## 7. Budget, model and result preservation

Actual new counters:4112 completed updates/74040 training heads;27592 neural inference attempts/110368 S4 heads;288 virtual starts;26240 observation pairs;6784 teacher calls. All are within the original caps. The query split is19200 collector +968 offline +384 NEW_FIT +6400 main +640 component. No technical inference attempt is missing a first-valid output. No new WBC/dynamic physics/hardware, monitor, mail or merge.

Both complete18-file Trainers have fixed-revision full-file readbacks:

- AGG2048: `306c8fcfc162fde03b71fc2c9bd055e10dcfcbde`, `arms/PI0270/AGG/checkpoint-2048`, tree `b4f3f5825c5111b8d69697f310de06072bb218eb8f15d961846c1f67be0063e0`.
- AGG4096: `f862abb26041c8e7d355470d6e9d48899f78b9ce`, `arms/PI0270/AGG/checkpoint-4096`, tree `0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66`.

Public repository: `leoyuer/pi0246-gate-pass-campaign`. Required result parts all have anonymous complete-file and all-member readback:

| Part | Fixed revision | Members | Tree SHA256 |
|---|---|---:|---|
| collection |b66feabfc40992f03e0bb63d376c417188771c9f|116928|7368b9ca21a750d6a688aee8b780b5cdd6e21db40a290a652a0c0af105c51326|
| newfit |e1f03fef4e4eab553fad341ebb0828497b6cf5ba|1344|7d2523b13ded7f6affc4cb3e68e3a7202965d422ada69be72eabfe25f26ed33e|
| AGG4096 |a801462bf705c4bf5d8a05c25ad4e64d763ab393|35472|e2e927590f10f2d7b60841dcd3ae8a0e47a197f06a1364bf5e5ec66c7e6be227|
| component |feb01221b2c74de0100d1cdd6e90c9e0c1e874bc|3264|8d691140d143ef4c2c2c8fa8d1d099d79adcd03442e7a140ffd9730ff4515d4c|

Their prefixes are `evidence/PI0270/learner-recovery-results-20261004/{part}`. Training-data archive is separately fixed at `d16394d8e32d535465828ef5d572b491a373e759`, prefix `evidence/PI0270/learner-state-recovery-20261004/training-data`; original source/data/processor and useful models remain protected. No historical archive was redownloaded to repeat already-closed readbacks.

The fixed-first-family all-eight-case video contains800 frames from saved96x96 RGB,5fps/2x playback, including failures and late drift. Full decode passed. Owner visually inspected every chapter start/end and frame99 at full resolution; the entire movie is not claimed visually watched. VIRTUAL/NOT_WBC/NON_REALTIME and original/reliable outcomes are visible. No new render/query was used for presentation.

## 8. Evidence index and scientific decision

- PRIMARY_VERDICT SHA256 `e05b056434fc57f4604209d60bc18f25da427a1682ff1b45f5df38a701ab32a1`.
- CPU_REPLAY SHA256 `9789ec9604c1973823b8abde2335df07fb3ee4d4f12026eda6e928eaa8ed3a79`.
- PRIMARY_COMPLETION_AUDIT SHA256 `e738e6e747c6c51797ac4ebf858ad551f29344b3a7ad4f40cbbfce9e4f10f187`.
- OFFLINE2904 SHA256 `85830d60d4bb369454acb500b1553be56e5db41c83f786a94b982bdfb5782faa`.
- NEW_FIT384 SHA256 `604b1dff08b42a6d1c1352a4dc87ca6be02c5b66009b5c374f88509d6494c924`.
- VIRTUAL192 SHA256 `a6b6e80c89fb9e7906e9de276b15b31af90e6c01d0ca44d39f18639803822363`.
- COMPONENT96 SHA256 `a7aa066b99fc224efc182ca17806f11142624d67003126a254f53f663e0e16cd`.
- Video SHA256 `aa36c8826c61a101a9e564dd91fb86bf5941b1d6f3ec464f7a885a82e249a309`; VIDEO_VISUAL_QA SHA256 `0552af7c1ffc133b3a6c06107260ac5be495265cf6cb8c1c99c63ed36599d958`.

AGG learned a substantial correction on the measured new support, improved some terminal behavior and restored original-donor legacy qualification. Reliable success improved0→6/64, but M3 entry/acquisition and M1 late orientation/pose retention remain unresolved. NEW_FIT improvement, terminal component improvement and task reliability are different outcomes. This is evidence of useful learning with an incomplete feedback capability, not evidence that training failed to execute or a generic claim that the model cannot generalize.

After actual final summary readback and the single terminal canonical, PI0271 A0 should enter the authorized fixed late-support/K8/visited-state diagnosis. No new training, seed/loss search, physical transfer or additional episode is authorized by this interpretation. E0/E1 are mathematical executors; eight opened development families are not blind TEST or two robot embodiments. Old PI0269/PI0268 FAIL and PI0267 secondary WBC PARTIAL remain unchanged.

Terminal after final publication readback: **PI0270_LEARNER_STATE_RECOVERY_STUDY_COMPLETE — STOP_FOR_RETENTION_OR_EXECUTOR_TRANSFER_DECISION**. This completes PI0270, not the full conditional PI0271 goal. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.

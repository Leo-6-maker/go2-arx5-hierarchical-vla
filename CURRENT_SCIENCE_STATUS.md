# Current Science Status

## 2026-10-24 实际训练增量

同一作业 **1481574** 已获 FIRST_UPDATE 放行；02:48:21 SGT 的原生回读确认 **B2598（从2048新增550次更新）、17次实际查询、2次加载、unknown=[]**，驻留owner和心跳正常。B4096及目标资格仍未完成，core64=0。运行胶囊36925d27未改变；冷态终局接线已独立资格且只允许真实F/Exit0和owner自然成功退出后使用。此后新增进度以同一身份链原生回执为准。

[实际首次更新和终局接线](reports/PI0294/design/parallel-science-20261022/B1481574_FIRST_UPDATES_AND_COLD_TERMINAL_CONNECTED_ACTUAL_20261024.json)。下方审核保留01:46基线及后继计划，不覆盖本段更晚的实际更新；计划不重置本轮预算和截止。

目标接线新增[同一 owner 的 quota／暂停 STOP IO](reports/PI0294/design/WDR1_TARGET_OWNED_QUOTA_PAUSED_STOP_CPU_20261024.json)，显式绑定实际冻结82004c31时钟；未知写不重发，完整状态／FSM／leg history不变且新鲜pause ACK后才返回停止回执。当前仅冷态代码与CPU资格，尚无新目标child、物理执行或任务资格；core64仍为0。

[冻结SDK实际链接输入](reports/PI0294/design/WDR1_ACTUAL_FROZEN_MPC_SDK_LINK_INPUTS_20261024.json)已定位并核对部分约束／预测函数体；它们可用于原实现薄封装的接续，尚不证明header/class ABI、完整链接来源或目标前缀安全。未重编／修改WBC，未新增目标模型、child或物理执行。

## 2026-10-24 审核摘要

完整审核基线36925d27，增量复核至 **5d9f2a82a6aa4830481f7197298f666cdc4aad0c（01:46:45 SGT）**。原生读数01:45:53、回执01:46:11；本页是固定时间快照，不是实时集群监控。

| 项目 | 最后可证状态 |
|---|---|
| 源域学生确认 | N1：T1 10/20、T2 4/20，尚未达到该轮各14/20 |
| A/B | A4096完整；B完整恢复点2048，历史2970不是完整checkpoint |
| 当前作业 | Vanda1481574已转R，server-resident owner健康；MODEL_START已放行，FIRST_UPDATE未放行，未观察新增更新 |
| 失败与成本 | 保留B2970的922丢失更新及1481489 pre-CUDA失败；当前campaign9666更新，3522未保留 |
| PBS | 17/28已实际采用；已用DEC1/global14/25，余原3matrix提交位置 |
| 源终点评估 | 旧B0 14/48（9成功5失败），新A/B完整终点评估未完成 |
| 目标与主矩阵 | 目标本轮原生资格未完成；core64=0，NOT_RUN |
| 最近截止 | 10-25 02:29:56 SGT停新作业／候选；10-26 02:29:56硬收尾 |

[详细审核](docs/reviews/2026-10-24_GR00T_WBC_PROGRESS_AUDIT.md) · [长程计划V3](docs/roadmap/2026-10-24_GR00T_WBC_LONG_HORIZON_PLAN_V3.md) · [当前接续交接](docs/handover/2026-10-24/GR00T_WBC_NEXT_EXECUTION_HANDOFF.md) · [最新实际运行／启动放行回执](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5d9f2a82a6aa4830481f7197298f666cdc4aad0c/reports/PI0294/design/parallel-science-20261022/VANDA_PRECUDA_RETRY_PBS17_28_UNIQUE_GPU_SUBMISSION_ACTUAL_20261024.json)

沿用冻结WBC／任务数据优先、Vanda-only及原科学协议。当前运行源码仍为36925d27，GitHub文档更新不热切capsule；已批准且实际采用的范围不重复索权，后继提案不自动增加计算额度。

发布并发复核：**5c9a6de763c6f74b7731809f72b2dd54e6ca47de，2026-10-24 01:59:22 SGT** 新增[1481574共享终局契约及精确身份CPU验证](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/5c9a6de763c6f74b7731809f72b2dd54e6ca47de/reports/PI0294/design/parallel-science-20261022/B_PRECUDA_RETRY1481574_SHARED_TERMINAL_CONTRACT_CPU_20261024.json)。它尚未产生原生终局审计／结算／关闭，未宣称首次更新；运行capsule仍为36925d27，上述科学与训练实际状态不变。

## 历史状态记录

以下原文保留；其中“Latest”、R/Q、未采用／未提交等字样只描述当时快照，不覆盖上方带日期的最新回执。

**Latest: N1allocation06 failed its pre-model engineering gate,not learning.**
[Exact raw-sample/resource receipt and repair](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
673223F/exit1,653GPU seconds;source36164,parent40664,reservations0.
Zero new models/NN/updates/physics.75actual owner reads spanned600.095s,
but81were required;duration-only collection under-counted real I/O overhead.
Native abort stopped the pump at72/80cycles;next checksum timed out at60s.
Repair keeps both81reads/600s,900s cap and all watchdogs;no seventh submitted.
Task matrix remains25qualified/32with sixDEVunrun and one exhausted mid-gap.

**Actual fifth terminal and first N1 cold continuation:** [paired result/resource receipt](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
673008F/exit0,increment11392GPU s,source35511settled. Matrix25/32qualified;
CORRTRAIN3/4,DEV/F0-T1 andF0-T2 individually audited32sSUCCESS. SixDEV
conditions unrun,no complete DEV rate;exhausted midpoint gap retained.
55c0610d actualCI SUCCESS,both cold native scopes pass,84old process
records closed,1899actual NN outputs accounted,13installed S1 zero-live
rejects preserved with no synthetic ACKs. Single N1job673223 submitted
auto/1H200/max8h,nativeR/released at latest readback,pre-model entry not yet
ready;zero new training,no valid-case replay.
ActualN1 600s/80cycles stillNOT_RUN;must pass before loading any model.
WBC/S1/TF-R3STOP unchanged,selected/deploymentNONE,originalN1clock preserved.

**Cold-bound N1 continuation plumbing prepared, no new job yet.**
Immutable common06-12 staging/activation and full recorded effect closure
are CPU-tested;live consumers,unknown effects or unsealed transport plans
reject advancement. Private same-owner finite adapter prepared,not launched.
506-test full suite plus21post-guard focused tests pass;native600s/80cycles
stillNOT_RUN. CORR2048DEV/F0-T1 andF0-T2 both independently audited32sSUCCESS,
zero native faults. Other six DEV conditions not run;no fullDEVrate claim.
Fifth673008 still originale418/request;cold activation awaits actual closure.

**CORR2048 TRAIN4 actual audited result: 3/4 success, T1=1/2,T2=2/2.**
[Matched endpoint task outcomes and complete raw-score/video index](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
All4 raw/video and closed fresh-input/model/action/ACK chains pass;native
safety faults0,TeacherOFF. F0/T1 rejects under originalS1at24s;the other
three complete32s successfully. BC2048TRAIN4also3/4with opposite T1layout
success;this is no overall TRAIN benefit,causalDAgger or DEV claim.
Matrix23/32qualified,CORRDEV8 inprogress,irrecoverable512F1/T1gap retained.
No newPBS or active-source mutation.245446c2 exact-head CPU CI SUCCESS;
actualN1 600s/80cycles/sitefault qualification stillNOT_RUN.

**N1 full transport producer/owner IO phase CPU-validated, native evidence pending:** [endurance/fault ledger](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
Actual-size recorded dualRGB,existing wire codec,compute checksum pump,
80fixture-only ACKs and all native fault producers now have execution paths.
Full raw journals/wire/CoreACK/process closures must join600s before load;
no inline PASS substitute.496tests/372subtests and local mature-client real
20s/EOF poison-channel bench PASS;NOT site native qualification. Existing
owner site adapter prepared,finite manager/cold effect closure still pending.
No newPBS/hot source switch;fifth673008R at CORR2048DEV8,4096updates,pending0.
Source-only task results remain separate from qualification and WBC success.

**Actual scientific milestone, not yet a complete behavior result:** [both2048 endpoints and actual matched consumption](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FINAL.json).
CORR2048 sealed94ad4d0e,combined4096formal updates,pending0. Each arm has
65536actual visits;32768T2+16384retained originalT1cursors match exactly.
CORR16384C1visits=8306student-anchor+8078subsequent expert-state visits,
4parents/8segments,T1only. Same originalM1/recipe/frozen parameters and
2048positive-gradient records per arm verified. Fixed96expert/12historical
student outputs and actual event exposure audited,zero newNN/physics.
Expert-fit metrics are not hiddenDEV or closed-loop success. FullCORR
TRAIN4 audited/DEV8 executing;matrix23qualified/32,irrecoverable midpoint gap stays.
Fifth673008R unchanged;no sixth. Goal and nativeN1 qualification incomplete,
selected/deploymentNONE,WBC/TF-R3STOP unchanged.84e7cee1 actualCI succeeded.

**Native heartbeat producer implemented, not yet deployed or qualified:** [N1 endurance and fault ledger](reports/PI0294/20261016T191445Z-localpc-mp1/ENGINEERING_ENDURANCE_AND_FAULT_INJECTION.json).
Future premodel entry requires600s/81real post-read samples,exact same-job
native identity and immutable raw-journal join;an inline PASS cannot replace
missing/tampered raw evidence.473PI0294 tests pass locally,not native proof.
Full transport producer/router and isolated native fault evidence remain
required;no newPBS/deployment/model or physical execution. Same healthy
fifth673008R,CORR1944at snapshot;endpoint and paired science still pending.
Original data/recipe/physics/S1/WBC and MP1/N1 clocks unchanged.

**Latest native progress and reader correction:** [same fifth job still running; reader03 terminal](reports/PI0294/20261016T191445Z-localpc-mp1/N1_READONLY_READER_TERMINAL_AND_SOURCE05_PROGRESS.json).
673008R,CORR1828/2048,one legitimate in-flight update,fivePBS,original4h.
No sixth/restart/hot code change. Reader03's previous live snapshot is
superseded: actual read-only SSH child SIGPIPE(-13),empty output,no blind
retry,zero audit intents;540CPU upper settled,reserve0. Pipe cause remains
unknown. Independent Core metadata read later succeeded;not endurance.
F1/T2 midpoint replacement remains an audited S1 negative,F1/T1 spent.
Endpoint/full paired science and nativeN1 qualification remain pending.

**Latest actual control repair:** [read-only serial auditor recovered without restarting science](reports/PI0294/20261016T191445Z-localpc-mp1/N1_READONLY_SERIAL_AUDITOR_ESTALE_RECOVERY.json).
Reader01 had actualNFS116 metadata failure,not a policy or neural failure;
its one complete audit retained. Reader02 failed local import protection
before START,zero audit effects,cost settled. One isolated immutable
replacement reader03 verified live and same native673008R read after START,
CORR1605at frozen snapshot. Only12remaining fixed endpoint audits permitted;
old deadline and all valid/engineering outcomes preserved. Main science
controller/codee418bb79 unchanged,no newPBS/model/query/physics. Metadata
recovery excludes mutation retries and heartbeat writing. NativeN1endurance,
CORR2048/full paired science remain incomplete;selected/deploymentNONE.

**Latest actual science/engineering boundary:** [N1 generic admission and site receipt](reports/PI0294/20261016T191445Z-localpc-mp1/N1_GENERAL_RUNTIME_ADMISSION_CPU_AND_SITE_READBACK.json).
Original fifth673008 stillR;CORR1345 at frozen readback,not endpoint2048.
No new model or physical attempt;current e418bb79 unchanged. Public N1
runtime layers/zero-work skip/known-unstarted probe resume pass CPU tests,
including108array exact equivalence to unchanged original probe primitive.
Native600s/80cycles,qualification producer,general finite manager and cold
deployment remain pending;newN1 model load is fail-closed until actual proof.
Project auth was autonomously refreshed and independently read back valid;
prior failed query below is history,not a current request for user login.
Nativeauto/small queue supports the authorized future8h upper,current4h
job untouched. No newPBS,new method,science-budget reset or WBC change.

**Current authority/runtime boundary:** [N1 single adoption and effective cumulative ceilings](reports/PI0294/20261016T191445Z-localpc-mp1/N1_ADOPTION_AND_EFFECTIVE_LIMITS.json).
N1 control adoption03:08:01SGT Oct18;three registered sites read back the
same fixed bundle/delta without changing active manifests. Source05/673008
is healthyR and CORR training beyond512,not endpoint completion. NewN1
source36h/12PBS and parent80H200h/48A40h include old used/reserved costs;
current fifth remains old-bound4h/no-sixth until terminal. Future N1native
code/deployment/600s heartbeat/80transport qualifications NOT_RUN. New
Hopper dispatch project check failed;in-flight training unaffected. No new
PBS/query/physics by adoption. OriginalMP1 T0/S1/WBC/STOPs unchanged.
N1 cutoffsOct19 09:08:01/13:08:01/15:08:01SGT,not reset on resume.
F1/T2 sole replacement now audited originalS1 negative;F1/T1 spent gap retained.
Known T1 saved-label supervision reuse is not heldout generalization.
Older "latest" paragraphs below are preserved historical snapshots.

**Latest public evidence:** [four CORR512 actual attempt clips,partial panel](reports/PI0294/20261016T191445Z-localpc-mp1/S_CORR512_FIRST4_AUDITED_VIDEOS_PUBLIC.json).
Fixed READY9ba0fe42,1132paired frames;Core full bytes and Hopper token-free
READY/index PASS.2valid negatives+2engineering partials cover3conditions,not
completeTRAIN4. F1/T1 rights spent,F1/T2 NOT_RUN. Matrix4new+12preserved URLs,
all non-URL science unchanged,not16successes. Publisher9cf27435/402tests/CI
39055006711 does not relabel original video source03646938 or source04 runtime
8e50bcc2. Both metadata publishers terminal/reservations0,no newscience/PBS.

**Latest diagnostic:** [BC512/CORR512 actual saved pair and student identities](reports/PI0294/20261016T191445Z-localpc-mp1/SAVED_BC512_CORR512_PROBE_COMPARISON_AND_STUDENT_IDENTITIES.json).
96expert windows/8true TRAIN episodes,actual identical input/noise/precision
and recomputed metrics verified. Prefix XYZ means T1 2.703->2.817mm,T2
2.986->3.010mm;not improved descriptive means,no causal mechanism decision.
Each additional BC2048/CORR512 student set has12original E0 states verified;
36outputs across3models are only12unique states/4parents. All trajectory gates
ACCEPT do not establish S1 clearance or recovery. New-model same-state expert
label joins remain pending;four invalid initial-label exclusions retained.
Three read-only auditors terminal,4.821Hopper CPU seconds settled,reservation0,
no model/physics/PBS/update. CORR2048 and full final comparison remain pending.

**Latest submission:** [actual native672907 Q,last original source slot](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_SINGLE_NATIVE_SUBMISSION_AND_QUEUE.json).
One qsub,CFP04-CF-007/auto/1GPU/12CPU/128GiB/max4h;GPU unavailable,Q not R.
SourcePBS4/4,parent7/12,no automatic fifth. Source23728s/parent28228s settled
plus14400s same-job reservation;45min closeout and parent final4h protected.
Same finite controller is live with fresh heartbeat;PBS_READY/release/compute
heartbeat qualification NOT_RUN. CORR512/cursor16384 not resumed,13conditions
unattempted,spentF1/T1 and valid negatives not replayed. Original140GiB output
estimate used,no new180GiB rebinding or required deletion. Full goal incomplete.

**Previous boundary:** [actual same-root source04 cold adoption before submission](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_COLD_BOUNDARY_ADOPTION.json).
Both native entries PASS with immutable8e50bcc2 capsules;400pytest tests and
254subtests/actualCI39052726122 PASS. Current manifest pointers cold-advanced,
old bytes sealed in history,old STOP713fef5c retained,old scopes blocked.15frozen
numerical+12learning/data/plant modules byte-identical;no new budget,T0,attempts,
GPU/PBS/model/physics. Source23728s,parent28228s,sourcePBS3/4,CORR512 unchanged.
Compute heartbeat qualification NOT_RUN;unchanged300s gate requires native
measurement before model load. Last slot04 only,no automatic fifth submission,
no thirdCORR512F1/T1 attempt or negative replay. Failed-checkpoint inventory
found no eligible large failed payload:actual deletion0/freed0,valid states kept.

**Previous terminal:** [source03 F/1,partial-effect closure and unresolved heartbeat diagnosis](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE03_TERMINAL_PARTIAL_EFFECT_CLOSURE_AND_HEARTBEAT_DIAGNOSIS.json).
Native672829:409wall/412conservativeH200seconds,26draws,0updates;finite owner
settled and exited. Source23728s,parent28228s,reservations0. CORR stays512,
cursor16384. SoleF1/T1 engineering replacement ends10.4s/5200ticks,0nativefaults;
both attempts raw-audited with18checks/full449+209frames,not qualified policy
outcomes. All26actual draw/action/ACK chains and known sequence26inbox-only
orphan close without resend;1926mutation intents COMPLETE. Original STOP
713fef5c retained on both hosts,no fourthPBS. Compute-read heartbeat value
unrecorded,NFS cause only hypothesis;8f689fe3 helper382CPU tests PASS but NOT
integrated/deployed/native-qualified. Original300s watchdog unchanged. Matrix
20attempts/18qualified/1engineering cell(exhausted)/13unattempted. SourcePBS3/4,
originalfamily caps/T0 remain;cold qualified scope still needed for final work.

**Previous queue milestone:** [native672829 submitted/Q and actual engineering qualification](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_CONTINUATION03_ACTUAL_QUEUE_AND_ENGINEERING_QUALIFICATION.json).
SourcePBS3/4,parent6/12,1GPU/max4h/auto;not allocated or resumed at this snapshot.
Same full CORR512/cursor16384 and original03646938 scientific runtime;new cold
management1b1806d9 passed373CPU tests/CI39047585619 and both native entry checks.
Only output estimate and finite metadata recovery changed,no scientific recipe,
safety/heartbeat or original MP1 budget/T0 reset. Exact owned queued memory
225->128GB decrease read back,no cancellation or additional submission.
Latest native matrix has18verified policy cells,1engineering-incomplete cell,
13unattempted cells across all32. ParentGPU27816s settled/14400s reserved;
source23316s settled,3960CPU seconds finite-control upper remains reserved.
No original valid-negative replay;remaining CORR512/F1 and2048 still required.

**Previous allocation status:** [allocation02 F/1 and preserved exact resume](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_ALLOCATION02_NATIVE_TERMINAL_AND_PRESERVED_RESUME.json).
CoreLab SSH loss interrupted the finite owner;Hopper enforced the unchanged
heartbeat watchdog. Both controller and read-only adjunct are terminal;CoreLab
connectivity recovered,physical workers terminal,all current reservations0.
Source GPU23316s,parent Hopper27816s settled;BC2048+CORR512=2560updates.
Two CORR512 F0 negatives are audited;F1/T1 is ENGINEERING_INCOMPLETE22.4s,
not a policy failure. F1/T2 andCORR2048 remain pending,original budgets intact.
[Failed-checkpoint check](reports/PI0294/20261016T191445Z-localpc-mp1/FAILED_CHECKPOINT_RETIREMENT_CHECK_01.json):
three registered storage areas inspected,no eligible large failed payload,
actual deletion0/freed0;required parents/fixed models/exact resume preserved.
No newPBS or model execution in this reconciliation. Prior milestones below
do not mean the old allocation is still running. Final goal remains incomplete.

**CORR512 actual learning and matched consumption:** [sealed state and first audited negative](reports/PI0294/20261016T191445Z-localpc-mp1/S_CORR512_ACTUAL_MATCHED_CONSUMPTION_AND_FIRST_VALID_NEGATIVE.json).
512actual updates,537 changed trainable tensors,frozen tensors unchanged;
full state31f0514b. Both initial1030tensor inventories equal originalM1 and
actual recipes match. Each arm16384actual visits,T1/T2=8192each;all8192T2
cursors and4096retained originalT1cursors match exactly.4096CORR T1correction
visits come from4student parents/8segments,including2062student-anchor and
2034later expert-state visits,not4096independent demonstrations. F0/T1 is
validS1 refusal1.6s:19raw checks/33decoded pairs/5actual draws/4ACKs+1zero-live
rejection PASS,native faults0. Other fixed conditions remain in progress.
352tests/CI39043547480 PASS;consumption audit terminal1.397CPU seconds,
reservation0. No newPBS,budget/clock/recipe/runtime change or candidate chosen.

**Current mainline:** [complete BC2048 endpoint and independent CORR start](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_COMPLETE_DEV8_AND_CORR_INDEPENDENT_START.json).
BC TRAIN3/4,exposedDEV3/8:T1DEV2/4,T2DEV1/4 over four development groups.
DEV has four originalS1 pre-execution refusals and one safe full32s object
goal-zone miss,not five executed collisions. Zero engineering gaps/native
faults,TeacherOFF.148 raw checks,3360 decoded camera pairs,423 actual draws,
419 executed ACKs and4 zero-live rejected prefixes independently verified.
[All eight actual DEV clips are now public](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_ALL_DEV8_VIDEOS_PUBLIC.json)
at fixed READY448ad9de,with Core full-byte and independent Hopper token-free
READY/index readbacks PASS. Earlier four TRAIN clips stay at READY53d71ad9:
all12 BC endpoint videos,including every negative,are available without any
old-prefix reupload or task-outcome change. No failed condition was repeated.
[Full32-cell behavior matrix](reports/PI0294/20261016T191445Z-localpc-mp1/BEHAVIOR_AND_FAILURE_MATRIX.json)
joins actual Core/Hopper registries,episode IDs,checkpoints,raw audits and
chains:16verifiedBC conditions plus1verifiedCORR negative,15pending conditions,
not a final matched comparison. The earlier77update snapshot is superseded
by the actual512proof above. Pure aggregator85c4a013 is not deployed into
the immutable03646938 science capsule. Native672471 is now terminal as recorded
above;no newPBS or resource-clock reset,actual GPU settlement is complete.
[Serial raw-audit harvesting and closed-session transport measurements](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_ALLOCATION02_SERIAL_AUDIT_AND_TRANSPORT_MEASUREMENTS.json)
are real,read-only and separately accounted. Residual wall time is not measured
neural inference or guaranteed recoverable GPU idle. CORR endpoint/exposure,
mechanism decision and final resource closure remain. TF-R3STOP,VandaNOT_RUN,
WBC/S1 and selected/deploymentNONE stay unchanged.

**B2 terminal BC evidence:** [actual2048 raw-event consumption](reports/PI0294/20261016T191445Z-localpc-mp1/BC2048_FINAL_RAW_EVENT_EXPOSURE.json).
All65536 visits join126 verified raw timelines. Sampled T1prefix8 covers
close96/96,open95/96,withdraw95/96;all events have H40 supervision. T2detach
andwithdraw cover30/30 each despite DEV1/4,so missing event supervision alone
is not established as the mechanism. No error/safety margin inferred from
coverage,no sampler change. Auditor terminal20.477CPU seconds,reservation0,
0newqueries/physics. CORR512 paired exposure now passes;terminal2048 remains.

## Previous Complete TRAIN4 And Video Milestone

**Current mainline:** [complete S-BC2048 TRAIN4 and actual public videos](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_COMPLETE_TRAIN4_AND_PUBLIC_VIDEOS.json).
3SUCCESS/1VALID_TASK_FAIL/0engineering gaps/0native faults:both T2 layouts and
F0/T1 complete32s/41of41 hold,whileF1/T1 is originalS1 rejection at1.2s.
All73 raw checks,1948 actual paired frames,244 fresh native draws and243ACKs
plus1zero-live rejection PASS. Four conditions/two seen TRAIN layouts,notDEV.
All four success/negative videos ACTUALLY HF-public at fixed READY53d71ad9,
Core full-byte readback and Hopper public unauthenticated READY/index PASS.
Eight fixed512 anchors join original actual inputs,cameras,arrays,weights,
precision and native-hook noise fingerprints;0newqueries/physics. Original raw
noise tensors not saved;no invented rehash. Same-state endpoint comparison
still waits CORR seal and separate original NI/shadow scope.334 CPU/CI PASS.
Native672471/same finite owner now evaluates exposedDEV8;CORR from original
M1 remains pending. No newPBS,budget/T0/WBC/S1/TF-R3 change,selectedNONE.

## Previous FirstT2 Success And Static Restore Milestone

**Current mainline:** [S-BC2048 F0 succeeds at both tasks;F1/T1 is a valid negative](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_FIRST_T2_SUCCESS_AND_ZERO_STEP_ANCHOR_RESTORE.json).
F0/T2 has independent18-check raw/video and80 fresh observation/generation/ACK
chains PASS:32s/641pairs,15.685cm push,3903 contact ticks,zero bilateral grasp,
41/41 hold and zeroS1/native faults,TeacherOFF. Same fixed2048 model also has
the previously auditedF0/T1 success. F1/T1 instead rejects before execution
at1.2s:19 raw checks,3ACKs+1 rejected prefix,zero native faults;valid negative
preserved. F1/T2 active at snapshot,DEV8/CORR pending,not completeTRAIN4.
Eight current512 saved full-state/history anchors actually restore with max
difference1.78e-15,no integration/query;static-only,not dynamic-prefix parity.
182.450 chargedCPU seconds/8 conservative starts,reservations0,B1 lease stays
CLOSED. Initial interpreter import gap preserved,same-bound existing-runtime
repair passed.330 tests/CI pass. Native672471/finite owner live,auth valid,
no newPBS,T0/recipe/WBC/S1/TF-R3STOP unchanged,selected/deploymentNONE.

## Previous First2048 Success Milestone

**Current mainline:** [S-BC2048 sealed / first autonomous F0T1 audited SUCCESS](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC2048_TRAINING_AND_FIRST_AUTONOMOUS_SUCCESS.json).
Complete2048 checkpointac0f4692 with full state;independent three-partition
audit65536 actual samples,T1/T2 each32768,126 original TRAIN episodes,C10.
The417 final resumed updates have finite positive gradients,unchanged493
frozen tensors,537 changed trainable tensors and original final scheduler.
F0/T1 completed32s,641 pairs,41/41 hold,lift8.136cm/carry16.978cm,zeroS1/native
faults,teacherOFF.18 checks,full641-frame video and80 fresh actual model/ACK
chains PASS. One seen layout only;remainingTRAIN3,DEV8 andCORR still pending.
Paired96-window/8-episode expert probe comparison is now strict512-to2048,
not legacyM1:identical raw input/noise/precision verified,prefix8 XYZ means
2.703/2.986mm to2.221/2.506mm. Not generalization or unique mechanism proof.
Mainline672471 still R,original finite owner/recipe/reserves intact;do not
restart2048 training or initializeCORR fromBC. Original512 valid negatives
andB1 newly verified table/pad shadows remain immutable;target STOP stays.

## Earlier Current512 Failure Attribution

**Current512 failure attribution:** [exact original history and first geometry](reports/PI0294/20261016T191445Z-localpc-mp1/CURRENT_SBC512_FAILURE_HISTORY_FIRST_GEOMETRY.json).
Four current S-BC failures now reproduce all27600 actual ticks with zero
highrate/control/reference difference and original64/124/63/150-tick rejects.
First contact is right_pad/table forF0T1/F0T2/F1T1,left_pad/table forF1T2.
All contacts are predicted shadows;original native faults0 and valid S1
negatives remain. No oldM1 cause substituted,no safety/plant/WBC change.
Eight current-policy-induced noninitial anchors saved full live/IK/controller/
teacher history,not justqpos/qvel. Original unseen RNG fingerprint not claimed;
independent snapshot restore/noise-receipt joins remain before new model draws.
B1 child is CLOSED,194.195 charged CPU seconds/28001 shadow ticks/4 conservative
diagnostic starts,zero new live conditions/queries/PBS/GPU and reservations0.
326 tests/native CI PASS. Source672471 remains independently active;fixed pair
and final review deliverables still incomplete,no optional intervention chosen.

## Earlier Mainline Resume And Event Snapshot

**Current native boundary:** [actual full-state resume and raw event audit](reports/PI0294/20261016T191445Z-localpc-mp1/RAW_EVENT_EXPOSURE_AND_ACTUAL_SOURCE_RESUME.json).
672471 is R,observed1743/cursor55776;native full1631 resume has first
microbatch52192/gradient1631/optimizer1632 and continuing scheduler. No2048
claim yet,512 valid negatives not repeated,CORR and DEV pending.
B2 actual raw-command/event joins cover126 TRAIN episodes and completed
512/1631 traces. T1 at512 misses close/open/withdraw events in sampled first8
for27/26/36 of96 episodes,but all events are in some sampled fullH40 target.
T2 detach first8 coverage30/30,withdraw29/30;do not call supervision absent.
Per-episode counts/hashes and96 expert probe strata are published;open-event
gripper errors illustrate why low overall XYZ is insufficient,not causal proof.
Audit is terminal20.136 CPU seconds,zero reservations/new inference/physics.
Original1200s CPU/wall family preserved through actual-schema repair,320 CPU
tests and native CI PASS. B1 exact-history reconstruction and current first
geom remain pending;no new24-query or sampler branch self-authorized.
Original MP1 T0/reserves/fixed endpoint,WBC and TF-R3 STOP remain unchanged.

## Previous Continuation Submission Snapshot (Before Native R)

**Previous native boundary:** [B0 saved-label join / source continuation02](reports/PI0294/20261016T191445Z-localpc-mp1/B0_SAME_STATE_LABEL_AUDIT_AND_SOURCE_CONTINUATION02.json).
Project authentication restored and actual CFP04-CF-007 membership passed.
672471 is Q,one H200/max4h/native128GB,not training yet;same native request
had one owner-only queued memory decrease from site-rewritten225GB. Source
PBS2/4,MP1 H200 actual16295s/reserved14400s,original clock/recipe retained.
Same-owner finite controller will resume full1631 state and handle remaining
fixed panels without repeating the four valid512 negatives;no2048 claim yet.
B0 all12 historical E0 draws joined,8 qualified noninitial expert H40 labels,
4 initial exclusions and two true parents/task. Prefix8 mean XYZ T1=6.95mm,
T2=12.55mm;these are action errors,not RMSE,tracking or causal proof. Same
input pixels/history/ACK commands independently checked;S-BC S1 NOT_RUN and
clearance NOT_MEASURED remain. Zero new inference/physics;private16-file HF
delivery verified on both hosts. Both CPU workers terminal,reservations0.
311 CPU tests/native CI success. B1/B2,BC/CORR endpoints,DEV and all four final
review deliverables remain incomplete. Runtime03646938/WBC/TF-R3 unchanged.

## Previous Native Boundary (Before Auth And Continuation02)

**Previous native boundary:** [adopted review /1631 partial /12 existing student
draws](reports/PI0294/20261016T191445Z-localpc-mp1/REVIEW_ADOPTION_STUDENT_PROBES_AND_ALLOCATION01_PARTIAL.json).
672080 isF/0,11795 conservative H200 seconds,not currently training. Complete
state at1631/cursor52192 and exact two-session consumption passed independent
audit; T1/T2=26096 each,correction0. Both child leases remain ACTIVE but no
workers or GPU/physics reservations remain. Next original-budget submission
awaits fresh project authentication (last query15006); no second source PBS.
Reviewfb16d101 is adopted without changingruntime03646938,clock or recipe.
The12 historical E0 student draw identities all passed; actual S-BC S1 and
qualified same-state expert-error joins are NOT_COMPARABLE/pending,not filled
from old-model evidence. Four new failures' geom and actual event joins await
bounded diagnosis. Fixed pair/terminal evaluation remains the mainline.

## Earlier Source Pair Snapshots (Not Current Native State)

**Latest fixed panel and real continuation:** [S-BC512 TRAIN4 complete / actual
full-state resume](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC512_FULL_TRAIN4_AND_RESUME_AUDIT_ENGINEERING.json).
S-BC512 has0/4 task successes: four valid original-S1 failures at
1.6/20.8/10.4/22.4s,zero native faults and zero engineering-missing rows.
All four independent raw audits,1108-frame video decodes and142 fresh
observation/native-generation chains passed,including138 executed ACKs and
four rejected zero-live prefixes. These valid negatives will not be repeated.
Same-job exact resume is actually at515/cursor16480; first resumed cursor16384,
finite gradients and continuing original scheduler are verified. Fixed2048
criterion,recipe and budget stay; no full2048,CORR/C1,DEV or target result yet.
The read-only resume-partition auditor passed289 CPU tests and real512
single-partition readback; real multiple-session audit awaits completed state.
672080 is R,runtime03646938 unchanged. Vanda camera stays CLOSED NOT_RUN.
Original MP1 clocks/caps and TF-R3 STOP persist. selected/deployment NONE.

**Independent saved-probe evidence:** [S-BC512 expert-state audit and legacy
parent gap](reports/PI0294/20261016T191445Z-localpc-mp1/S_BC512_SAVED_PROBE_AUDIT_AND_PARENT_PROVENANCE_GAP.json).
All96 windows from8 TRAIN episodes have exact raw inputs,noise,weight hashes
and independently recomputed metrics. Prefix8 XYZ means T1/T2=2.70/2.99mm;
all96 trajectory-gate ACCEPTs are not S1 or live task success. Four T2 H40
windows have spurious gripper transitions,only one inside the executed8-point
prefix; these do not uniquely explain live failure. Strict M1 comparison is
NOT_PROVEN: its first legacy NPZ lacks actual input fingerprint fields.
No resampling or fake metadata; M1's prior task successes remain untouched.
Both CPU auditors are terminal,7.017 CPU seconds total under their original
600-second family bound. Native training observed668/cursor21376 continues.

## Earlier MP1 Snapshots (Not Current Native State)

**Live source / terminal camera:** [actual learning and original-window camera closeout](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE_LEARNING_STARTED_AND_VANDA_CAMERA_QUEUE_CLOSED.json).
Source672080 is R with real S-BC updates and verified balanced committed sample
prefix; CORR/new policy outcomes are not available yet. Camera1463728 expired
without allocation,was withdrawn at that boundary and is CLOSED/NOT_RUN,zero
GPU and zero reservations. Source learning continues independently of target.

**Recovery videos delivered:** [fixed public HF byte verification and queued-memory delta](reports/PI0294/20261016T191445Z-localpc-mp1/PUBLIC_RECOVERY_VIDEOS_AND_QUEUE_MEMORY_DELTA.json).
All ten attempts/3405 decoded paired frames are indexed,including failures and
two initial-only clips. Same source job672080 remains Q; memory-only request
was lowered225GB to128GB with native readback,no newPBS or training recipe change.

**Current mainline:** [matched Source Pair first native submission](reports/PI0294/20261016T191445Z-localpc-mp1/MATCHED_SOURCE_PAIR_FIRST_SUBMISSION.json),
job672080 Q, CFP04-CF-007/auto, one H200/max4h; no training result yet.
Both bindings passed, fixed parent/C1 and task-balanced sampling retained.
This is a new MP1 source study, not reopening E1 or the closed M2 recovery.

**Recovery-family final:** [eight-condition actual panel and resources](reports/PI0294/20261016T191445Z-localpc-mp1/RECOVERY_FINAL_ACTUAL_PANEL_AND_RESOURCES.json).
CORR seen-TRAIN T1 is 2/2 successful scenes; BC has one success and one
engineering-missing T1. Fixed total: 3 SUCCESS, 2 valid original-S1 failures,
2 engineering-missing, 1 NOT_RUN. No complete T2 or DEV/WBC success.
All ten attempts independently audited; 240 complete fresh observation/ACK
chains and 480 actual model-input camera hashes verified for three successes.
671899 is F/0; all three recovery children are CLOSED, 4,500 conservative
H200 seconds, zero updates/reservations. No fourth recovery PBS. Read-side
metadata assistance remains explicit; original MP1 budget/T0 and TF-R3 STOP
persist. Vanda camera qualification remains separately queued, not PASS.

## Historical Progress Snapshots

Observed 2026-10-17 after E1 and both MP1 recovery-child lease closures by the existing single local-PC owner.
This is an actual-state summary, not a replacement for immutable run receipts.

**Second independently verified complete model case:** [M2-CORR512 F0/T1](reports/PI0294/20261016T191445Z-localpc-mp1/M2_CORR_FIRST_COMPLETE_T1_AND_RAW_AUDIT.json)
also completed32s,641 pairs,41/41 hold,0 S1/native faults,18 raw checks and80
cross-host actual ACK chains. This oldC0 model's success on F0 and BC's F1
success do not establish a paired gain, freshC1 learning or generalization.
Remaining CORR evaluation is still running under the single7200s allocation.

**Latest actual qualification:** [original M2-BC512 F1/T1 full32s SUCCESS](reports/PI0294/20261016T191445Z-localpc-mp1/M2_BC_COMPLETE_T1_AND_CLOSED_LOOP_QUALIFICATION.json)
has641 camera pairs,41/41 hold,0 S1/native failures,18 independent raw checks,
and80 cross-host observation/generation/actual ACK chains with160 input-image
hashes matched. P0 is qualified for this explicitly metadata-reader-assisted
paused-sim transport, not unassisted/real-time performance. Scientific runtime
remained4d430556. Other M2 conditions remain in progress; no complete T2, DEV,
generalization, correction gain or target result is implied.

**Subsequent actual third-run progress:** [first 29.35s engineering partial and metadata observers](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_FIRST_PARTIAL_AND_DIRECTORY_OBSERVERS.json).
671899 is R; first BC F0-T2 stopped at the original 900s wall bound, not a
complete32s success despite 28/28 observed hold checks. Next BC F1-T1 is running.
Directory-only CPU observers have explicit cache observer effects and separate
sealed sources; no model/action/physics/watchdog code was hot changed. No P0
PASS or unassisted end-to-end performance claim is established by this snapshot.

**Latest direct human scope revision:** [third recovery-family submission](reports/PI0294/20261016T191445Z-localpc-mp1/THIRD_RECOVERY_ADDITIONAL_GO_AND_NATIVE_SUBMISSION.json)
is explicitly permitted once, at most two hours, without changing MP1 total
budget/T0/deadline or prior attempt counts. Child03 passed both native entries;
671899 is Q, 7,200 GPU seconds reserved, no model loaded yet. BC F0-T1 is
excluded as exhausted; BC3/CORR4 remain eligible. Previous children01/02 stay
CLOSED and their engineering conclusions/costs remain. No fourth submission.

**Later independent native admission:** [sprint adoption and camera binding](reports/PI0294/20261016T191445Z-localpc-mp1/SPRINT_ADOPTION_AND_VANDA_CAMERA_BINDING.json)
records a new camera-only A40 job1463728, currently Q (site node_pool capacity).
This does not reopen the two closed recovery children or TF-R3 physical STOP.
No new model/physics/WBC work has run; 900 GPU seconds and 3,600 CPU seconds
are reserved until native terminal reconciliation. Runtime EGL remains unproven.

## Active Authority

**MP1 is the only new incremental resource family.** The user directly adopted
release `c8cdf60e41e31da5796b44a064ef2f22d594a491`.
The concurrently published E2 documents remain non-additive references; there
is no E2 owner, second budget or 28-day clock.

- MP1 run: `20261016T191445Z-localpc-mp1`.
- T0: 2026-10-17 03:14:45 SGT. New submissions/candidates stop October 23
  03:14:45; new experiments/training blocks stop October 23 19:14:45;
  hard closeout October 24 03:14:45 SGT. Restarts do not extend these times.
- New upper bounds: 48 H200 GPU-hours, 32 A40 GPU-hours, 12 new GPU PBS
  submissions per site, 24,576 optimizer updates. Native site limits win.
- Old E1 and Vanda jobs retain their original capsules, budgets and leases.
  Their reservations are not spendable MP1 balance.
- [Actual adoption and three-host readback](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json).
- [Fixed execution entry](docs/handover/2026-10-17/MAINLINE_ACCELERATION_MP1_EXECUTE.md).

MP1 has completed CPU-only source-loader qualification and one Vanda native
build, then actually ran recovery jobs 671817 F/1 and 671826 F/0 using the
existing M2 weights with zero optimizer updates. Both recovery children are
now CLOSED. Total conservative H200 cost is 453 seconds; four actual starts,
600 live plus 600 shadow integrations, five generations, three executed ACKs,
two completed draws not installed, pending/reservations zero. All four raw
evidence audits passed, but there are zero valid full policy results:
three engineering-incomplete conditions and five NOT_RUN across eight fixed
conditions. No 32s relay qualification or M2 behavioral gain is established.

The first native failure was a heartbeat-file read in the recovery entry;
the mature bounded control reader was reused in a new immutable child.
The next attempt actually located a 30.301976s stall inside Hopper RPC ledger
accounting, exceeding the unchanged 20s watchdog. Lock wait versus
read/write/fsync is not yet separated. The branch is stopped, with both of
its two PBS opportunities used; no third equivalent recovery submission.
The controller also advanced to an initial-only CORR case before the owner
stopped the cross-model branch; this charged deviation remains in the matrix.
[Actual phase timings](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_ACTUAL_RPC_PHASE_TIMING.json),
[eight-condition matrix and four audited attempts](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_E1_M2_RECOVERY_BEHAVIOR_MATRIX.json),
[real qualification failure](reports/PI0294/20261016T191445Z-localpc-mp1/MP1_REAL_RELAY_QUALIFICATION.json),
[closed child resources, not whole-MP1 closure](reports/PI0294/20261016T191445Z-localpc-mp1/E1_RECOVERY_RESOURCE_FINAL.json).
Continue independent saved-T2 diagnostics and separately qualified Vanda
work under the original MP1 clock. Do not train new weights to bypass this
evaluation fault. Fresh permission and capacity remain actual admission checks.

The subsequent registered CPU-only saved-T2 audit actually completed with
zero new physics, model queries or training. Both original M1 T2 trajectories
already entered the unchanged all-corners goal zone (F0 16.70s / F1 16.15s).
Last saved TCP-object distances were 0.04690m / 0.04240m, below the original
0.08m clearance requirement; objects remained in-zone and nearly stationary.
This supports investigating post-push disengagement, not declaring complete
T2 success or a unique collision cause. Original scores recomputed exactly.
That first saved-only receipt left the illegal geom unknown; the subsequent
separately admitted exact-history shadow result below now resolves it.
No new autonomous negative episode was run.
CPU was conservatively charged at its 600s process cap, not advertised as
exact whole-process usage. [Actual saved-state/action audit](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_SAVED_PROGRESS_AND_ACTION_AUDIT.json).

**Exact-history T2 attribution is now complete, not just prepared.** Child
`saved-m1-t2-shadow-01` reconstructed all 18,000 saved historical integrations
with zero maximum difference, including the original static IK history.
The two model prefixes reproduced their original 56/52-tick S1 rejections:
F0 table-left_pad and table-right_pad; F1 table-right_pad. Maximum forbidden
contact overlaps were 0.03726mm / 0.01903mm. These trigger the original
any-arm-table-contact rule, not the separate 4mm penetration threshold.
Both same-state expert and expert-XYZ-only 0.4s prefixes passed; SO3-only
and gripper-only substitutions still rejected. This distinguishes the tested
Cartesian path, not a unique Z bias or permission to calibrate/relax S1.
The audit-only expert remains in approach history; short-prefix PASS is not
qualified full T2 Recovery, a model-intent label or new correction supervision.
[Actual geom/component receipt](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_FIRST_REJECTION_ATTRIBUTION.json)
and [event/contact figure](reports/PI0294/20261016T191445Z-localpc-mp1/M1_T2_TIMELINE_AND_CONTACT_DIAGNOSTIC.png).

The child is CLOSED with worker terminal, original input hashes unchanged
and reservations zero. New shadow cost: 19,123 integrations (18,000 exact
history + 1,123 diagnostic), Core CPU meter 189.659s including the 180s
management allowance; new GPU/PBS/model/optimizer/live/autonomous-eval zero.
Its two shadow reconstructions were conservatively charged to the MP1 source
start allowance. MP1 source-budget starts are now 6 (4 actual new evaluation
attempts + 2 shadow reconstructions), total integrations 20,323, H200 453s;
the original MP1 clock and target TF-R3 STOP remain unchanged.

**Original T2 TRAIN30 event coverage has also been actually audited.** All
30 admitted demonstrations / 30 distinct scene groups (F0/F1/F2 = 8/11/11,
not 30 OOD categories) contain every original push stage. Their 18,030
overlapping H40 anchors include detach 756 (4.19%), withdraw 1,500 (8.32%),
and 1,926 cross-stage windows containing detach. No window lies entirely in
detach because those segments last only 1.10-1.35s, shorter than H40's 2s;
this is not missing detach supervision. Raw stages exactly match the frozen
manifest buckets. All 640 supervised commands plus the unlabeled terminal
observation were explicitly distinguished and original input hashes rechecked.
[Actual episode/event/window coverage](reports/PI0294/20261016T191445Z-localpc-mp1/T2_TRAIN30_EVENT_AND_H40_COVERAGE.json).

The fixed Source Pair sampler forecast gives 1,324 detach + 2,678 withdraw
visits among 32,768 planned T2 visits per branch. This is a deterministic
future schedule, not actual training consumption or evidence of sufficiency.
No S-EVENT trigger or main recipe change follows automatically from these
counts. One CPU audit failed its initial 640-versus-641 observation check;
the bounded explicit-terminal repair passed. Both workers are terminal,
charged at 600 CPU seconds each as conservative process upper bounds,
reservations zero; new physics/neural/training/GPU/PBS zero.

**The measured RPC accounting hot path has a scoped code repair.** Runtime
`4fda127a` keeps successful-reply telemetry in memory and settles RPC count
plus process CPU once at exit; neural/optimizer/physics charges, journals,
guards and original 20/60/900s contracts stay durable and unchanged.
Poisoned channels now publish a separate engineering stop on both child
roots; native recovery/Source Pair guards reject it before the next model.
An existing marker prevents new panel binding. This is not a native safety
STOP and does not relabel previous policy or engineering outcomes.

Both registered Core/Hopper CPU probes actually held a synthetic ledger lock
in another process while emitting 80 memory replies, then made one final
shared-ledger transaction. All workers and lock holders are terminal, and
uncertain/repeated settlement is rejected. Native probes use no real model,
neural/physics counter or RPC wire; they prove this telemetry-lock isolation,
not actual SSH/NFS/32s qualification or the cross-model model-stop behavior.
Heartbeat and guard reads still do I/O; complete-loop speedup remains unknown.
A killed session requires registered CPU-bound reconciliation, not zero cost.
[Implementation, actual probes and limits](reports/PI0294/20261016T191445Z-localpc-mp1/RPC_REPLY_LOCK_ISOLATION_CPU_RECEIPT.json).

The first CPU probe on each site failed before lock setup on a Path expression;
the explicit role-entry test repair passed on the sole bounded CPU retry.
All four attempts are terminal and conservatively charged 600s/site/attempt
(1,200s per site), reservations zero. Native queue recheck finds no own
PI0294 running/queued GPU job; old recovery leases remain CLOSED, 453 H200s
unchanged, no third recovery PBS. A new legal end-to-end route is still required
before claiming P0 or a completed M2 behavior comparison.

**Campus connectivity is restored and the old E1 native states are reconciled.**
Job 670729 is F/exit 1 after the original owner-heartbeat timeout, not a
model/safety result. Both old jobs are terminal: 16,363 conservative allocated
GPU seconds charged, their old reservation released only after native proof.
The final third E1 job **671655** is now F/exit 0. All three allocations
are natively reconciled: 19,675 conservative GPU seconds charged and zero
GPU reservation. No fourth E1 submission is permitted even with unused
seconds. All source/neural/controller/stdio workers were terminal and
physical reservations zero before the separate CPU artifact-closeout stage.
Both E1 leases and registry entries are now
CLOSED after native/resource/artifact readback. E1 is a bounded closed
execution with explicit scientific comparison gaps, not a successful full
M2 behavioral comparison or a reopened resource window.
[Final resources and closed leases](reports/PI0294/20261016T150228Z-localpc-e1/RESOURCE_FINAL.json).
[Complete execution summary](reports/PI0294/20261016T150228Z-localpc-e1/E1_RUN_SUMMARY.md).
[Actual final allocation, training and consumed exposure](reports/PI0294/20261016T150228Z-localpc-e1/LAST_NATIVE_TERMINAL_AND_ACTUAL_M2_EXPOSURE.json).
[Restored-network reconciliation and exact final allocation](reports/PI0294/20261016T150228Z-localpc-e1/NETWORK_RESTORED_AND_LAST_NATIVE_ALLOCATION.json).
The preceding interruption remains preserved:
[Actual interruption and resumption order](reports/PI0294/20261016T191445Z-localpc-mp1/LOCAL_NETWORK_INTERRUPTION_01.json).

Offline E1 checks repaired the last-allocation controller's middle-hop cache
path and bounded pipe-write timeout, plus final condition/attempt reporting.
All 228 CPU contracts and a local OS-pipe controller fixture passed; these
are not native pipeline qualification or a measured speedup. The saved M1
T1 audit metadata still reconciles to two seen-TRAIN successes, not new runs.
[Checks and exact native readmission preconditions](reports/PI0294/20261016T150228Z-localpc-e1/LAST_ALLOCATION_ENGINEERING_CHECK_01.json).
No third E1 submission, native source switch, new MP1 GPU or physical action
was made during the disconnected interval. That historical network block
is resolved, not deleted or treated as completion.
[Blocker and explicit completion gaps](reports/PI0294/20261016T150228Z-localpc-e1/OWNER_BLOCKED_NETWORK_AUDIT_01.json).
Original clocks and total caps remain unchanged. The final immutable scope
retains all M1 valid outcomes, reuses completed M2-BC weights without training,
and admits three settled engineering replacements plus the unstarted fourth
BC condition and fixed C0 M2-CORR branch. It does not hot-insert C1 data.

**BC recovery did not produce a valid comparison.** Its first replacement
stopped after 6 simulated seconds on a 20-second file-RPC timeout; a late
response was then incorrectly reused on the same channel, causing the next
three conditions to fail response-ID checks before physical integration.
All four remain engineering failures, not valid policy negatives. No more
BC replays are made in E1. Both SSH file endpoints subsequently exited.
CORR completed its fixed 512 updates/16,384 samples in the same allocation,
with 537 changed trainable tensors, frozen parameters unchanged, and 512
finite positive gradient records. Its first autonomous case stopped at
10.8 simulated seconds on another original 20-second RPC timeout, with
zero S1 rejection/native fault. The poisoned-channel latch then prevented
the other three physical cases from starting: NOT_RUN, not three failures.
No valid complete M2 comparison exists. Both model weights are preserved.
The single local controller was replaced only at the settled BC/active
CORR-training boundary, using the same owner lock and original deadlines.
An uncertain channel now latches invalid before any later physical case.
The finite direct-open metadata helper actually ran and ended, costing
1.160 CPU seconds, no model queries or physics. It did not establish a
complete pipeline repair; original timeouts and scientific sources stayed
unchanged. Independent main-evidence audit accounts for all 22 actual
attempts, including three zero-integration engineering attempts. Audit
PASS is not task success or complete policy-outcome coverage.
[Actual failure chain, timing caveats and controller continuity](reports/PI0294/20261016T150228Z-localpc-e1/BC_REMAINDER_ENGINEERING_FAILURE_AND_CORR_CONTINUITY.json).

## Source Results

**M1@1024 autonomously completed T1 in both predeclared seen TRAIN layouts.**
Both runs lasted the full 32 seconds with Teacher OFF, original D0/S1/scorer,
no native safety fault, 41/41 terminal hold observations, and independent
raw/action/ACK/contact/score/video audit PASS. T2 was validly rejected by the
original S1 in both layouts. This is not DEV, Go2/WBC or real-robot success.

- [Complete M1 TRAIN4 and independent audits](reports/PI0294/20261016T150228Z-localpc-e1/M1_1024_COMPLETE_TRAIN4_AND_T1_AUDITS.json).
- Terminal E1 H200 job `670729.hopper-m-02` completed M2-BC's actual 512 updates,
  with 537 changed trainable tensors and 512 finite nonzero gradient records.
  Its three attempted evaluations are engineering-incomplete, not task outcomes;
  the first reached 25.6 simulated seconds before the original 900-second
  relay limit. M2-CORR training is now complete but its autonomous panel is
  engineering-incomplete. Both branches have the fixed M1@1024 parent and
  their previously sealed data choices; valid outcomes were not rerun.
- Actual consumed traces verify BC T1/T2=8192/8192 and CORR=12288/4096.
  CORR consumed 8192 original and 8192 C0 samples, from six correction
  segments/two independent student parents. Equal updates are not
  task-matched exposure; no causal correction-gain claim is supported.
- All 12 predeclared terminal-model conditions are explicit in the
  [actual behavior matrix](reports/PI0294/20261016T150228Z-localpc-e1/MODEL_BEHAVIOR_MATRIX.json):
  four valid M1 policy outcomes, five engineering-incomplete conditions,
  three NOT_RUN conditions. All 22 actual attempts have accounted evidence,
  but not all have valid task results. The six fixed probe contexts now have
  648 original-prefix S1 shadow checks, without new neural queries.
- On the noninitial fixed student states, CORR T1 prefix8 XYZ error is
  2.19 mm versus M1 6.90 mm, with original S1 prefix execution 6/6 versus
  4/6 across all fixed student states. These overlap C0's same two TRAIN
  parents and do not prove independent generalization or autonomous gain.
  [Actual student-state diagnostics](reports/PI0294/20261016T150228Z-localpc-e1/ON_STUDENT_DIAGNOSTIC_COMPARISON.json).
- [Saved M1 T2 object progress and original final rejection](reports/PI0294/20261016T150228Z-localpc-e1/M1_T2_SAVED_REJECTION_AND_OBJECT_PROGRESS.json):
  actual horizontal motion 13.28/14.61 cm before original S1 rejection,
  without lifting; no complete task success. That historical saved-only
  receipt did not know the specific illegal geom pair and did not inherit it
  from M2048. The separately charged MP1 shadow receipt above now resolves
  those pairs; the old E1 evidence and result remain unchanged.
- E1 C0: 833 full H40 windows from two E0 student episodes, T1 only. It is
  state aggregation, not new M1 on-policy DAgger.
- Four real M1 collection episodes are terminal. Two required qualified
  expert takeover after actual student prefixes; two produced no correction.
  Saved-data repair produced 832 new windows, 1665 including C0, with
  independent source and actual Hopper processor/loader checks passing.
- [Repaired C1 fixed revision and actual consumption](reports/PI0294/20261016T150228Z-localpc-e1/C1_INTERVENTION_REPAIR_ACTUAL_DELIVERY.json).
- **E1 M2 still uses C0**, frozen before the repaired C1 became ready.
  The new C1 is available to a later explicitly bound MP1 branch, not hot
  inserted into E1. T2 Recovery remains unqualified (full recovery 0/2).

The original M1@256 precision comparability gap, one disclosed unintended
repeat, invalid initial fork histories, old E0 negatives and all costs remain
preserved. No claim of a flawless full E1 comparison is made.

All four fixed inference checkpoints and 38 decoded main/fork clips now
have immutable public HF revisions and READY readback. The private upload
hit quota and its empty prefix was reconciled; the private repository was
not made public. Only two Cosmos metadata paths were sanitized; weights
and statistics stayed unchanged, with complete upstream license notices.
Public payloads exclude optimizer/RNG and private paths/assets. M1@1024
and both M2 full native training states remain on Hopper; M1@256 was
inference-only originally. Public inference/resume qualification is not
claimed. [Actual fixed artifact index](reports/PI0294/20261016T150228Z-localpc-e1/ARTIFACT_DELIVERY.json).

## Target Platform

Native Vanda refresh: hardware job `1462438` and saved-input inference job
`1462451` are F/exit 0, with CLOSED leases. Native allocated GPU walltime is
58 and 137 seconds respectively, charged to their old scope, not MP1.
The former passed actual A40 CUDA checks; the latter loaded old M2048 and
made 17 saved-observation queries (one warmup, 16 timed), median 0.2356s,
peak allocated 13.35 GB. This is neither a live loop nor H200/A40 numerical
equivalence, and does not qualify M1/M2 or native EGL rendering.
[Original native terminal receipts](reports/PI0294/20261016T191445Z-localpc-mp1/PRIOR_VANDA_ENGINEERING_NATIVE_TERMINALS.json).
Failed old builds are preserved.
MP1 CPU build `1463204` is F/exit 0: container tests, native compilation,
before/after controller-source checks, and linked-library checks passed.
The simulator and two controller binary hashes are recorded in the
[actual build receipt](reports/PI0294/20261016T191445Z-localpc-mp1/VANDA_CPU_NATIVE_BUILD_ACTUAL.json).
The build used 1,533 native CPU seconds and zero GPU, physics, model queries
or WBC launches. It does not establish runtime safety or task success.
Healthy queued/held jobs were not cancelled. Container/package network
downloads were not pre-metered: the ledger retains an explicit pending
audit and 32 GiB conservative reservation, not a measured zero-transfer claim.

The old TF-R3 safety STOP is unchanged. MP1 permits only evidence-backed,
bounded owned task-interface requalification, not WBC/plant/PD/safety changes
or a cross-host bypass. Source research does not wait for target construction.

## Next Admitted Work

After E1 closure, GitHub CI at d30354e5 failed four fixtures because pytest
loaded a second module alias and string-based mocks patched the other alias.
Tests now patch their actual imported module objects. Remote CI at
0218b645 passed; default local pytest is 827 PASS/14 explicit SKIP, PI0292
108 PASS and PI0294 243 PASS. No native scientific capsule was hot patched.

An earlier MP1 engineering action actually ran on both hosts: 96 same-host
synthetic packet/heartbeat/ledger filesystem cycles each, with no model,
PBS or physical start. The maximum component latency was 15.74ms Core
and 24.71ms Hopper. No 20s stall was reproduced; this is not a cross-node
NFS, SSH or 32s closed-loop test. Direct open is not a proven speedup.
Guard/invoke/accounting/output phase tracing was then used by the two real
recovery children and located the accounting stall described above, without
watchdog or mutation-retry changes.
Initial CPU entry/path errors and a pre-dispatch immutable-request collision
are preserved; failed cost is conservatively charged, not silently zeroed.
All probe processes are terminal, reservations settled, E1 still CLOSED.
[Actual timings, limitations, CI and costs](reports/PI0294/20261016T191445Z-localpc-mp1/ACTUAL_RELAY_CPU_DIAGNOSTIC_01.json).

E1 cannot submit again or reopen its closed leases. Continue within the
already adopted MP1. Both existing-model recovery PBS slots are now spent;
the failed shared relay branch is stopped, not queued for a third equivalent
attempt. A future usable evaluation route still requires actual qualification,
not another same-host smoke or training new weights to bypass the fault.
No valid M1 outcome is replayed.
Then bind S-BC2048 on original v3.0 TRAIN126 and the fixed M1@1024 parent;
use the matched S-CORR branch from qualified frozen correction data. Preserve 1:1
task exposure, including when only T1 has correction. Actual TRAIN126 has
96 T1 and 30 T2 episodes; the task-first sampler preserves 1:1 exposure.
The real source-reader and exact-cursor-resume CPU checks passed. With only
T1 correction qualified, S-CORR uses 50:50 original/correction within T1,
giving 25% correction overall, not a task-unbalanced 50% overall mixture.
Next Vanda work is runtime qualification, not another compilation-only claim.
No new DEV labels, WBC tuning or BASE padding.

The user additionally permits HF public storage for new owned artifacts.
This does not make existing private repositories public or grant rights to
redistribute private WBC/vendor assets. Credentials stay private; publication
requires sanitization and license review and does not block legal local use.
Historical selected remains NONE; real-robot deployment remains NONE.
# Latest Actual Source04 Terminal and Direct Fifth-Slot Authority

Source04 native672907 is F/1,381wall/391conservative seconds;source24119,
parent28619H200seconds,reservations0. CORR remains512/cursor16384,combined
updates2560. F1/T2 engineering partial8.4s/4200ticks/169paired frames passes
18raw checks,not a policy result.21ACKs,22draws,one known uninstalled action,
1970journal entries closed;both engineering STOP bytes preserved.
Observed negative67.9ms heartbeat age identifies a pre-read clock race.
Post-read repair07b04707 passes406tests/CI,keeps original0..300s predicate;
native sustained qualification remains pending. Direct human reply authorizes
sourcePBS4->5 only,one extra<=4h within original12H200h/T0/MP1 caps and1536
remaining updates. No sixth,no F1/T1 third attempt,no valid-negative replay.
Cold binding and single submission are NOT_YET_EXECUTED.
[Exact terminal and audit receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE04_ACTUAL_TERMINAL_AND_CLOCK_REPAIR.json).
[Exact additional-slot card](reports/PI0294/design/SOURCE_PAIR_ADDITIONAL_PBS05_AUTHORIZATION_20261018.json).
Matrix now21attempts/18qualified cells/2engineering cells/12unattempted,
16prior public clips retained. Previous source04Q snapshots are historical.
Full matched pair remains incomplete;TF-R3STOP,VandaCLOSED_NOT_RUN,selectedNONE.

# Latest Actual Fifth Submission:673008 Q

Human-authorized fifth job submitted exactly once,CFP04-CF-007/auto/1GPU/max4h.
SourcePBS5/5,parent8/12,no sixth. Same finite owner live. Both native cold
entry checks PASS at e418bb79,560source members,15numerical/12learning-data-
plant modules unchanged;both prior STOPs and old histories retained. Original
birthstage4 remains byte-identical,effective5 requires sealed direct human card.
416tests/274subtests and actualCI39057571128 PASS,ten exact blob readbacks.
Source24119s/parent28619s settled plus14400s reserved,not actual GPU use.
CORR512/cursor16384 not yet resumed,remaining1536updates. Q is not native
sustained heartbeat qualification or task completion. Only original F1/T2
replacement;no F1/T1 third,no valid-negative replay,no T0/budget/S1/WBC reset.
[Actual cold and single native submission receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_ACTUAL_COLD_AND_SINGLE_NATIVE_SUBMISSION.json).
Full paired science incomplete;TF-R3STOP,VandaCLOSED_NOT_RUN,selectedNONE.

# Latest Actual R and Historical Student-Label Comparison

Fifth673008 is actually R. Forced compute/server heartbeat qualification before
model load PASS with real post-read clocks,original0..300s gate unchanged.
CORR512F1/T2 original sole replacement is executing real prefixes;its terminal
raw/chain audit and sustained transport outcome are still pending. SourcePBS5/5,
no sixth,remaining1536updates,no F1/T1 third or valid-negative replay.
[Actual native admission/prefix snapshot](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_NATIVE_ADMISSION_AND_REAL_PREFIXES_IN_PROGRESS.json).
[Actual three-model same-state saved-label comparison](reports/PI0294/20261016T191445Z-localpc-mp1/THREE_MODEL_HISTORICAL_STUDENT_LABEL_COMPARISON.json):
12historical states/4parents,8qualified labels/4invalid initial exclusions.
At matched512,T1 prefix XYZ macro BC6.954->CORR2.502mm;T2 XYZ12.550->11.793mm,
gripper0.2675->0.2726. Only2TRAIN parents/task,C1 overlap unresolved,not heldout
generalization,full recovery,live S1 clearance or a final paired comparison.
BC2048 is descriptive at a different update count. New four CPU-only workers
terminal,8.0704actual CPU seconds charged once,2400reserve cleared;all16private
HF members/model verified on Core. No newNN/physics/teacher labels/model-weight
downloads. Auditf29f748a passes418tests/276subtests/actualCI39058058261;running
sourcee418bb79 unchanged. CORR2048/full pair/mechanism incomplete,selectedNONE.

# Latest Actual CORR Resume and Audited F1/T2 Negative

673008R resumed exact originalCORR512 fullstate. Fixed observed progress533,
cursor17056;optimizer/scheduler/RNG/sampler/stats preserved,stop2048 unchanged.
F1/T2 sole replacement is valid originalS1 failure12s/6000ticks/241pairs,
19raw checks and31draws/30ACKs/1zero-live rejected prefix verified,native faults0.
Original8.4s engineering partial retained,no further F1/T2 retry;F1/T1 no third.
Matrix22attempts/19qualified cells/1engineering gap/12unattempted finalcells.
[Exact audit and resumed formal-update receipt](reports/PI0294/20261016T191445Z-localpc-mp1/SOURCE05_CORR512_F1_T2_AUDITED_AND_EXACT_TRAINING_RESUME.json).
HistoricalT1probe parent overlap with actualC1 consumption now verified2/2,
3115of4096correction visits at512;T2overlap0/2. Error improvement is explicitly
not heldout generalization;exact anchor/window training overlap not asserted.
Current sourcePBS5/5/no sixth,originalT0/caps/45min closeout unchanged.
Concurrent N1 docs preserved by ordinary merge,expanded budget not native-bound
or consumed by current healthy job. Full pair/mechanism incomplete,selectedNONE.

# Latest CORR512 Raw Events and Supervision Reuse

[Actual event and executed-label reuse audit](reports/PI0294/20261016T191445Z-localpc-mp1/CORR512_ACTUAL_EVENT_EXPOSURE_AND_SUPERVISION_REUSE.json)
joins16384committed visits,4096C1=2062student-anchor+2034expert-after states.
Four historicalT1qualified expert-label windows exactly recur in C1 and were
consumed2060times;T1probe error gain is training-supervision fit,not heldout
generalization. Whole model processor tensor byte identity is not claimed.
OriginalT1prefix-close/open/withdraw uncovered52/49/57 versusBC51227/26/36;
coverage changes alone do not prove a sampler bug or license a recipe change.
All30T2episode event/visit summaries match BC512 exactly. C1stage/detach/
withdraw remain unknown,not zero.96saved CORRexpert draws join raw-event
strata;no newmodel/query/physics/teacher labels. Two workers terminal,
17.5037metered CPU seconds,one shared180s local upper,1800CPU reserves cleared.
Audit83405c41 passes426tests/285subtests/actualCI39059634999;runninge418bb79
and healthy673008 unchanged. CORR2048/full paired final still pending.


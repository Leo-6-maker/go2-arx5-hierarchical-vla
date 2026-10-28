# E1 Actual Results and Explicit Comparison Gaps

Scientific execution and artifact delivery are terminal. Both E1 leases are CLOSED,
with native resource and registry readback. No registered E1 workload remains active;
the closed leases admit no further E1 experiments.
This is a bounded execution receipt, not a claim of a complete M2 policy comparison.

## Authority and Identity

- Run: `20261016T150228Z-localpc-e1`; one existing local-PC owner.
- Original E1 authority: `c851cc791e1fbb3efb293f7075f0e771518cf5c0`.
- Adopted priority review: `114f0786a329fae95aadd89f9d5d96c3ccb143d8`.
- Original T0, deadline, v3.0, EEF10, D0, S1, four-step inference, score and WBC were not changed.
- MP1 retains its already registered T0/budget. No MP1 GPU or physics execution is claimed here.
- Selected/deployment: **NONE**. No DEV, Go2/WBC transfer or real-robot success is established.

## The Predeclared Conditions

| TRAIN condition | M1@1024 canonical | M2-BC@512 | M2-CORR@512 |
|---|---|---|---|
| F0/000 T1 | Autonomous 32s SUCCESS | Engineering incomplete | Engineering incomplete |
| F0/000 T2 | Valid original S1 rejection | Engineering incomplete | NOT_RUN |
| F1/000 T1 | Autonomous 32s SUCCESS | Engineering incomplete | NOT_RUN |
| F1/000 T2 | Valid original S1 rejection | Engineering incomplete | NOT_RUN |

There are four valid policy outcomes, five engineering-incomplete conditions and three
unstarted conditions. BC has seven actual attempts across four conditions, not seven
independent conditions. The complete [matrix](MODEL_BEHAVIOR_MATRIX.json) retains all
22 actual attempts, including other checkpoint contexts, four collection episodes and
the disclosed scope deviation. Independent evidence accounting passes for all actual
attempts; the three zero-integration attempts have initial-frame evidence, not task scores.

Both M1 T1 successes retain Teacher OFF, original safety/scoring and 41/41 terminal hold
observations. The two M1 T2 episodes actually moved the object 13.28/14.61 cm without
lifting before rejection. Their final forecasts report `ILLEGAL_CONTACT_OR_PENETRATION`
at offsets 56/52 shadow ticks. Specific geom pairs remain unknown; old M2048 causes are
not substituted. See [saved T2 evidence](M1_T2_SAVED_REJECTION_AND_OBJECT_PROGRESS.json).

## Actual Learning and Consumption

| Branch | Updates | Consumed samples | T1 / T2 exposure | Correction |
|---|---:|---:|---:|---|
| M1 | 1024 | 32768 | Task-balanced SMALL8 | None |
| M2-BC | 512 | 16384 | 8192 / 8192 | None |
| M2-CORR | 512 | 16384 | 12288 / 4096 | 8192 fixed C0 samples |

Both M2 branches use the same fixed M1@1024 parent and new optimizer. Each has 537
changed trainable tensors and 512 finite positive projector/DiT gradient records, with
frozen parameters unchanged. All 2048 authorized E1 updates are accounted for.
The actual sampler traces prove a task-exposure confound: equal update counts do not
make this a task-matched causal correction comparison. C1 was not inserted into E1.

C0 has 833 H40 windows from six segments/two E0 student parents. Repaired C1 adds 832
windows from two M1 takeover parents, giving 1665 windows/four independent parents.
These are not 1665 independent demonstrations or student intervention anchors. C1's
fixed delivery and real Hopper reader pass remain available for a newly bound MP1 branch.

## Fixed TRAIN Diagnostics, Not Task Success

| Model | Expert prefix8 XYZ T1/T2, mm | Noninitial student XYZ T1/T2, mm | Student T1 original S1 |
|---|---:|---:|---:|
| M1@1024 canonical | 2.583 / 2.873 | 6.897 / 13.401 | 4/6 |
| M2-BC | 2.639 / 2.733 | 7.042 / 13.016 | 4/6 |
| M2-CORR | 2.579 / 2.795 | 2.192 / 12.333 | 6/6 |

XYZ errors are episode-macro means. Expert diagnostics use 48 anchors/task; student
errors use four noninitial actual-ACKed expert H40 comparisons/task from two parents.
The two invalid initial Oracle comparisons/task remain unavailable, not zero-error labels.
S1 uses all six fixed student states/task. All three models pass 6/6 student T2 prefixes.
The complete six-context check contains 648 real original-S1 shadow evaluations, no
new neural queries or main starts. These fixed TRAIN states overlap C0 ancestry:
the T1 fit improvement is real but is not held-out generalization, new C1 learning,
or autonomous task gain. See [actual student diagnostics](ON_STUDENT_DIAGNOSTIC_COMPARISON.json).

## Engineering Outcome and Resources

The last BC replacement stopped at 6s on the original 20s RPC timeout; three subsequent
zero-step response-ID failures were preserved. CORR stopped at 10.8s/27 ACKed prefixes
on the same original whole-call timeout, with zero S1 rejection/native fault. Its new
poison latch prevented three later physical starts. Partial lift/carry is not success.
The finite visibility helper cost 1.160 CPU seconds and did not establish a full repair.
Completed CORR RPC medians were 0.0223s Core / 0.0379s Hopper; the unknown timed-out
call is not included in these successful-call statistics. No unique stall cause is proven.
Posthoc read-only reconciliation accounts for all 47 final-panel neural draws: 42
ACKed executions and five completed but uninstalled draws (including the three zero-step
BC attempts). All were charged originally; pending draw state is empty. This does not
change the historical timeout or authorize replay. See [reconciliation](UNKNOWN_RESPONSE_TERMINAL_RECONCILIATION.json).

All three E1 native allocations are F; last job 671655 exited 0. Conservative charged
GPU time is 19675s, GPU reservation zero, submissions 3/3. Unspent seconds do not
authorize a fourth submission. Main starts are 22, recovery starts 16, live/recovery
integrations 247400 and shadow integrations 563150. Instrumented CPU accounting,
including the initial management allowance, is 7858.56s Core and 12305.20s Hopper.
Observed owned disk is about 2.23/83.80 GiB respectively; continuous peak monitoring
and exact whole-account OS CPU are not claimed. [Final counters and closed leases](RESOURCE_FINAL.json)
record zero CPU/GPU/physical reservations and no cancelled healthy foreign jobs.

## Artifact Delivery

The private HF model upload hit its storage limit; readback confirmed an empty target
prefix, with no payload or READY commit. The original private repository was not made
public. Under the existing explicit public-storage permission, audited sanitization
exports inference-only weights/configuration and complete upstream license notices.
Weights/statistics are unchanged; only two private Cosmos model-name paths are relocated.
Optimizer/RNG state, credentials, private runmaps, raw assets and WBC are not published.
Full M1@1024 and both M2 native training checkpoints remain intact on Hopper; M1@256
is the original inference-only checkpoint, not a full-resume state. Public resume/inference
qualification is not claimed. All four fixed model revisions have READY readback in
the [artifact index](ARTIFACT_DELIVERY.json), together with hashes and public URLs' components.

All 38 independently decoded main/fork clips have public fixed-revision delivery.
They include three initial-frame-only clips, assisted collection and invalid-restoration
fork evidence; not all clips are autonomous task trajectories or successful demonstrations.
Existing fixed probes were transferred and actually consumed at Core with 652 members.

## Continuation Within Existing MP1

Preserve these results and gaps. Qualify a reliable bounded relay before spending more
physical attempts, then recover any admitted existing-model comparisons in a separate
MP1 child at a normal boundary, never by reopening E1 or replaying valid M1 outcomes.
The prepared S-BC/S-CORR source pair uses original TRAIN126/stats and qualified frozen
C1, with task-balanced exposure; it has not trained yet. Diagnose current M1 T2 geom
and phase from exact saved history, not a new Teacher project or safety relaxation.
Vanda next needs runtime/camera/native safety qualification, not another compilation.
TF-R3 STOP, WBC/plant/PD/safety and absent BASE supervision remain unchanged.

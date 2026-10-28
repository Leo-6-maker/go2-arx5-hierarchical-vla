# PI0286 completed two-arm scientific comparison draft

**Both prescribed1024-update arms completed with actual native training CPU PASS. Neither fixed endpoint qualifies under the original source gates.** C0 TRAIN2/8 fails7/8; C1 TRAIN8/8 passes, but DEV5/16 fails13/16. No fresh CONFIRM_C was started and no source policy is qualified. Root actual sealed COMPARISON_CLOSE=BOTH_ENDPOINTS_CLOSED and original select=NO_DEVELOPMENT_QUALIFIED_CANDIDATE are now verified (sealed final receipt SHA ae3e4d81072a1dc664b21cb908972e8185fb0fa8668e1ddb55c1f8e550e6bb1e; local owner readback wrapper SHA 5935069e274db90d81519f731ebea15db6d72a7c40e0356f96c4e19b76457e1b). Scientific execution is stopped; only CPU/HF/video archival remains. This is the read-only scientific draft for owner publication.

| Actual evidence | C0 REPLAY1024 | C1 CORRECT1024 |
|---|---|---|
| Parent | Full immutable DOSE6144 / Adam6144 | Same parent, independently reset |
| Native formal updates |1024, CPU PASS |1024, CPU PASS |
| Real fresh canary |PASS8 vs4+fresh4 |PASS8 vs4+fresh4 |
| Source exposure |8192 expert chunks |4096 expert +4096 admitted suffix chunks |
| TRAIN first-valid CPU |**2/8** |**8/8** |
| DEV first-valid CPU |**NOT_RUN: original TRAIN gate failed** |**5/16** |
| Original gate outcome |TRAIN<7/8 |DEV<13/16 |
| DIAG96 original gate |95 ACCEPT /1 WARN /0 REJECT |96 ACCEPT /0 WARN /0 REJECT |
| Fresh CONFIRM_C |NOT_RUN |NOT_RUN |
| Qualification / target deployment |NONE |NONE |

## What is actually paired

Closed formal journals prove identical native training noise and time hashes at all2048 matching microbatch indices; each arm consumes8192 chunks with the same overall phase exposure[2048,2048,2048,1638,410]. Data differ by the prescribed intervention. Saved TRAIN8 arrays prove eight identical initial scenes, actual RGB/state/instruction packets, and all640 identical noise/time/K4 inference packets; model point is excluded from the noise identity. Later states/images may diverge through the different policies. C0/C1 paired TRAIN outcomes:2 both successes,6 C0-failure→C1-success,0 opposite changes. This is a fixed TRAIN-domain result, not independent generalization.

Old parent TRAIN7/8 and DEV13/16 used a different noise namespace. Neither7/8→2/8 nor13/16→5/16 is a strict paired causal estimate. The new C0/C1 comparison is limited to one parent, one training seed and one prescribed data intervention; no unique recovery mechanism or population-wide benefit is established. C0 DEV was not observed and must not be presented as0/16 or used to invent a matched DEV advantage.

## Actual correction admission and exposure

Exactly32 fixed requested slots were faithfully restored:28 physical successes,27 native-codec admitted.3s14/16 admitted;12s13/16. Exclusions remain four physical failures and285118@3s with three genuine SO3 failures (max0.010654909013566369rad). Historical15/32 remains sealed as an implementation-added zero-preclip verdict, superseded by V2. V3 actual sampler preserves50:50 source weighting and equal branch draws within eligible source/buckets, original phase quotas/event preference/eight strata. No failed prefix or failed branch became a positive label.

| Source | phase0 | phase1 | phase2 | phase3 | phase4 | total |
|---|---:|---:|---:|---:|---:|---:|
| C0 expert |2048 |2048 |2048 |1638 |410 |8192 |
| C1 expert |1024 |1024 |1024 |819 |205 |4096 |
| C1 corrected suffix |1024 |1024 |1024 |819 |205 |4096 |

## Stored native loss, gradient and LR

| Metric | C0 | C1 |
|---|---:|---:|
| First16-update native loss mean |0.016680527 |0.132817075 |
| Last16-update native loss mean |0.009791979 |0.026623198 |
| LR peaks(interface,DiT) |5e-5,5e-6 |5e-5,5e-6 |
| Other31 rows and Adam moments |Exact frozen / zero |Exact frozen / zero |

C1 global stored native loss mean0.031412686, min0.002332880, max0.597764790. All2048 microbatches mix the two sources;1608 also mix phases. Per-source/per-example losses are **UNOBSERVED** and cannot be recovered from the aggregate scalar. The440 source-mixed/single-phase batches provide only composition-conditioned descriptive scalar losses (phase0n126,1n115,2n121,3n72,4n6), not causal phase/source effects. Loss magnitudes across different data distributions are not a qualification metric.

Thirty-two saved gradient samples contain row-module and DiT norms; no gradient was recomputed. Mean/max saved norms: action_decoder0.509474/1.842812, action_encoder0.384304/1.689435, state_encoder0.056281/0.218409, DiT7.998212/44.555953. These are the logged audit fields, not evidence of violating optimizer gradclip1. Actual LR2048 samples span0..5e-5 and0..5e-6; last[1.30580056e-10,1.30580056e-11]. Native masked active reduction1600 and frozen scope were checked by original training CPU audits.

## Source failure decomposition

C0 six failures:285100/106 lack sustained grasp/lift;285103/104/107 have aggregate acquisition histories but never enter target zone;285101 reaches zone7.56s but retains grasp/lift at all41 final hold ticks and fails table/static/release/TCP-clear. These stages/predicates do not uniquely establish learning or runtime cause.

C1 DEV cumulative original funnel:16 requested→14 sustained grasp→12 grasp+lift→11 complete acquisition histories→5 all41 stable-place→5 released→5 withdrawn→5 complete task. Eleven failures split five incomplete acquisition (2missing sustained grasp,2missing sustained lift,1missing horizontal history) plus six later place/hold failures. Among those six, overlapping failures are zone5/table2/static2/not-grasped2/TCP-clear4/green0; do not add overlapping counts as independent causes.

| Failed DEV layout | Earliest incomplete history / later outcome | Original hold | Target→installed RMS(m) | Installed→same-issued achieved RMS(m) |
|---|---|---:|---:|---:|
|285200 |GRASP_WITHOUT_SUSTAINED_LIFT |41/41 |0.000064081 |0.008052561 |
|285204 |ACQUISITION_HISTORY_WITHOUT_ZONE |0/41 |0.011886141 |0.010141961 |
|285205 |ACQUISITION_HISTORY_WITHOUT_ZONE |0/41 |0.000060058 |0.008452266 |
|285207 |ACQUISITION_HISTORY_WITHOUT_ZONE |0/41 |0.014232724 |0.007155862 |
|285208 |ACQUISITION_HISTORY_WITHOUT_ZONE |0/41 |0.003641185 |0.010005603 |
|285209 |ZONE_REACHED_BUT_FINAL_PLACE_RELEASE_WITHDRAW_HOLD_FAIL |0/41 |0.006831672 |0.006143373 |
|285210 |ZONE_REACHED_BUT_FINAL_PLACE_RELEASE_WITHDRAW_HOLD_FAIL |4/41 |0.000067968 |0.007976200 |
|285212 |MISSING_SUSTAINED_GRASP |0/41 |0.011446841 |0.007534166 |
|285213 |GRASP_WITHOUT_SUSTAINED_LIFT |0/41 |0.000047348 |0.009324390 |
|285214 |MISSING_SUSTAINED_GRASP |0/41 |0.017251867 |0.005499173 |
|285215 |NO_HORIZONTAL_HISTORY |0/41 |0.015447678 |0.006640022 |

Preserved counterexamples and late failures:285200 has41/41 final task-hold samples but no sustained lift history and remains FAIL;285212 has a transient grasp first-true11s but no sustained grasp;285213 transient lift6.12s does not satisfy sustained lift.285209 reaches zone5.43s then first loses it6.86s and fails all final zone ticks.285210 first satisfies T1hold only29.84s and passes4/41 final samples; this remains FAIL.285208 lift first-true27.76s does not rescue its missing zone/final release. First-true values are individual original predicates, not a newly relaxed sustained-success gate. Target/install/achieved RMS describe actual tracking and do not prove a neural cause.

## DIAG96 original metrics — descriptive only

| Saved metric | C0 first8 | C1 first8 | C0 H40 | C1 H40 |
|---|---:|---:|---:|---:|
|Pooled equal-length XYZ RMS(m) |0.014747059 |0.017891534 |0.027543615 |0.030207373 |
|Pooled equal-length SO3 RMS(rad) |0.087603201 |0.095711096 |0.128302460 |0.144208106 |
|Mean gripper MAE |0.047531764 |0.056447983 |0.017120743 |0.026856041 |

First point versus actual input state: C0 XYZ median/max0.010062750/0.038520231m, SO3 median/max0.011544726/0.177910278rad; C1 XYZ0.011625387/0.040455762m, SO30.015120508/0.127491934rad. First-point gripper error was not stored separately and remainsUNOBSERVED.

C0 WARN: UID T1-TRAIN-285106-20, phase0/control20/task query0, angular speed2.1214336123rad/s versus original warning threshold2.0. C1 has no WARN or REJECT. Both native decode error counts are0. Raw normalized OOB counts stay diagnostic and cannot silently replace original ACCEPT/WARN/REJECT decisions. C1 improves fixed TRAIN successes while these aggregate raw-teacher DIAG errors do not decrease; these are different outcomes, and neither replaces DEV or fresh confirmation. Historical two DEV raw-target rotation/clipping cases remain unmodified under the same statistics; no normalization repair was performed.

## Closed final cumulative budgets and runtime

| Counter | Actual closed/prefix snapshot | Original cap |
|---|---:|---:|
| Formal updates |2048 |2048 |
| Discarded canary |32 |32 |
| Technical completed updates |0 |32 |
| All completed updates |2080 |2112 |
| Quality queries(two hosts) |4092 |10000 |
| Native S4 heads(two hosts) |16368 |40000 |
| Raw charged starts |81 |176 |
| Raw charged rgb |53267 |116160 |
| Raw charged control |53268 |116160 |
| Raw charged dispatch |106536 |232320 |
| Raw charged physics |266335 |580800 |
| Raw charged teacher |15680 |43000 |

Raw81 starts include80 unique quality conditions and the preserved engineering-incomplete/replacement pair for learner285109; the repeated slot is not a quality retry. Raw reservations count incomplete/live operations before dispatch and do not imply every reserved final operation completed.32 endpoint model episodes=(C0TRAIN8+C1TRAIN8+C1DEV16); learner16 and correction32 remain separate budgets. Fresh confirmation episodes0. Planner-call count is UNOBSERVED in read ledgers; do not infer zero from missing rows.

Hopper same-device own interval union **5083.990035s**; Core **3147.901836s**, no unmatched active start; after-science closure readback unchanged at14:25. These include original engineering attempts; nested canary/model children are not added again inside their parent reservation. Per-host14400s reserve leaves HOP9316.010s/Core11252.098s. Scheduler actual C0wall26:16 and C1wall47:33 are separate occupancy receipts, not extra intervals to add. C1 scheduler host peakmem62,223,724kB/vmem118,993,688kB; saved allocator allocated max26,203,665,408B/reserved max31,746,686,976B are different metrics. True GPU-compute utilization and full CPU-phase utilization remainUNOBSERVED.

## Full model recovery and owned storage

C0 full14 was actually read back at HF revision b0b8c65de8d5bbdf0946ccdda9ecb23348cf16c9, prefix arms/PI0286/C0_REPLAY1024/checkpoint-1024, tree e61e2d5288dab9a0408a67bd88583d022e3c85d42aef5ebfbe699957bcb3d7ab. Owner then retired only its local three large payloads20,463,648,768allocatedB; small evidence remained. C1 current full14 local Trainer/optimizer are protected; its model-only transfer excludes optimizer. No retirement by this worker.

Scoped actual own allocation **75,941,203,968B (70.725758GiB)**,100GiB cap headroom **31,432,978,432B**. Core42,591,764,480B/HOP33,337,806,848B unique regular-file allocation plus directories. Count once per(host,device,inode); external original parent/expert/pristine symlinks not traversed. This is a timestamped filesystem allocation snapshot, not personal quota or physical storage-backend dedup; future video/archival growth remains additional. Earlier three-canary106.35GiB projection was a genuine transient risk corrected by owner storage actions, not a scientific data failure.

## Limits and next experiment

No eligible new source policy: gates remain TRAIN7/8, DEV13/16 and freshCONFIRM26/32. No threshold, scorer, normalizer, labels, heldout denominator or old confirmation result changed. No blind confirmation, hardware, targetWBC, Go2, multi-task or deployment claim is made. A successor should be separately authorized and pre-register training-state coverage across acquisition/late hold, an equal-budget expert-only control and independent confirmation; the present results do not authorize another optimizer/model/physics branch.

Evidence files: TWO_ARM_ENDPOINT_COMPARISON.json; C1_CLOSED_SOURCE_PHASE_LOSS.json; TWO_ARM_CLOSED_EXPOSURE_NOISE_IDENTITY.json; TWO_ARM_TRAIN8_INITIAL_NOISE_PAIRING.json; C0_ACTUAL_FAILURE_AND_DIAG_SUMMARY.json; C1_ACTUAL_DEV_FAILURE_FUNNEL.json; C1_DIAG96_METRIC_SUMMARY.json; PI0286_CLOSED_FINAL_BUDGET.json; PI0286_OWNED_STORAGE_SUMMARY.json. This worker performed saved-array/small-ledger CPU reads only: new neural/gradient/optimizer/codec/render/physics/teacher/PBSsubmissions=0. Active code/seal/data and all original evidence were untouched.

### CPU budget uncertainty

No cumulative CPU-time quota or remaining CPU-time meter is present in the read ledgers.4threads/16GiB are per-worker bounds, not proof of remaining CPU hours. Saved loader/forward timing and PBS cput cannot be converted into precise free CPU capacity. Arithmetic unused GPU reservation and raw/query cap headroom do not authorize further science after the owner stop.

# SLA V6.1: actual prospective first4 and inherited V6 closeout

Snapshot **2026-11-14 17:46:13 SGT**. Fixed plan PR383 `6e6636577d53b6b25a92b1e36cf2e619e765eab3` remains unmerged. The user explicitly advanced **only the first4 new expert episodes** to test prospective recording and the native student entrance, while keeping old RP8 failed. Execution stays on Draft PR384. [Machine-readable evidence](ACTUAL_20261114.json) records each outcome, source, checksum, counter, cost and remaining boundary. The [15:49 report](../sla-action-contract-v61-20261114/EXECUTION_20261114.md) is retained unchanged.

## Four new complete PICK_HOLD episodes: actual4/4

Prospective producer source `a29f076963af048569f8779bb20fd5ad9fbf9aac`, TRAIN01/04/07/10, restart14: **4 requested /4 physical requests /4 terminal /4 complete expert PICK_HOLD episodes**, 86812 native integrations. Each has21703 integrations,61 final H40 plans,2440 supervised points,640 actually consumed points and321 body+wrist frame pairs. All four shut down cleanly, without native physics stops. Worker CPU totals2127.0702s; finite campaign wall1052.5015s. At most three new worlds ran concurrently, reserving one slot for the existing V6 evaluator. No failed stratum was replaced or retried.

This is complete **grasp-and-hold**, not PICK_PLACE, formalD100/D300 or autonomous SLA success. There were **0 SLA model queries /0 optimizer updates /0 training visits**. The first4 producer and each raw shard explicitly keep training admission false.

At every actual query, recording captures the **post-EEF_STREAM final plan**, pre-planner proposal, reference_before_query, actual query state/frame/mount/base and stable original command IDs. It then writes a real normalized FP32 file, decodes through the existing GripLogitCodec, writes real ACTION.npz and uses the **unchanged raw student compiler**. Both the original plan and FP32-submitted program pass the original full-H40 native certifier. Original installation ACKs and completed25-tick intervals confirm consumption. No expert regeneration follows raw compilation. The submitted FP32 program is kept separately from the original final plan.

The four episodes used original native binary `39e5136603f51c092b2f3e595245468dbb0c12dbca79818b0df08da8dd4af7dd`, original SDK/solver/controller and frozen full N20 statistics `024951afc22e03586635f17b5ee8f473aead1960f107d8768384643844afb9e1`. Callback replay was **disabled**. The experimental callback binaries below were not used for this collection.

Across244 plans, **9760 supervised points and2560 actual origin-command consumptions** remain separate. There are240 P10 queries with40 supervision/10 consumption each, plus four terminal P40 holds with40/40. A later query never fills an earlier plan's unconsumed suffix;renewal ACKs add no consumption. Each async writer committed823 events,with zero pending/dropped records. This is site evidence, distinct from the earlier unit-only40/10 case.

## Same four episodes, two independently frozen views

Converter source `ad7458193763bd96bb1de5a4a118f14f9bb7dce1` CLOSED0. Both views retain the same lossless original observations, query states, anchors, mounts and episode split, with **244 windows each**. Each separately fits TRAIN statistics after relative XYZ/SO3/logit transformation using all supervised target slots in those unique windows, without filtering by consumption or adding resampling multiplicity.

| View | Manifest SHA256 | Frozen statistics SHA256 | Max FP32 installed-reference error |
|---|---|---|---|
| PD4_EXECUTED_REF | `a06eab92e12f9b6eec91e04d45665cb9f65834d1d0f626a9b0a0b22571df3f4c` | `66ff4bdf39bfbf1db3d9ff394dad8a0a0384be1b4e22c0b80c1be2304cf6fe37` | XYZ3.201e-9m /SO3 1.796e-9rad /mechanical grip4.949e-10m |
| PD4_FINAL_PLAN | `f9482067c8634fec3dab197a50a9aa8ae1b0c37731d8bf32e446ce3cd883e603` | `31462fbeb99e907605eb203d359c78dedaf8f356f643adb725dc637c9a52fcbf` | XYZ1.662e-9m /SO3 6.975e-10rad /mechanical grip4.0000004e-6m |

EXECUTED_REF inverts transport using explicit reference_before_query, never installed[0]. Its all40 consumed-reference slots may span multiple original queries;rolling windows reuse physical commands in different anchor contexts and do not create extra physical consumption. FINAL_PLAN retains each original query's actual10/40 or40/40 consumption mask. Original nominal raw-shard H40 concatenations are not admitted as either view.

The CPU audit actually serialized/read FP32 files, decoded them, ran the original raw compiler and full geometric envelope for all488 view-windows. Audit wall17.6686s/CPU2.4261s,0 models/updates/physics. **The newly fitted PD4 statistics have not passed native replay or training admission.** Successful first4 transport used the old frozen N20 statistics. This is PD4, not PD20, and partial-mask loss qualification has not been implemented or claimed.

## Inherited V6: actual complete3000, full87 and freshRGB16

Original source `3f48cadfafac08e86b2bc14594481978bb6788f0` CLOSED0 at **3000 actual updates /96000 committed target visits /0 source visits /0 physical requests**. All12000 forward/backward attempts completed. Training recorded64 ordinaryRGB generations;the separate fresh process completed16 bit-exact ordinaryRGB reconstructions without a teacher or optimizer. All87 initial model/Adam/RNG arrays matched original N20 before update1;data,recipe and endpoint stayed unchanged.

Final COMPLETE SHA `b01d15817391f7e74971678ad3304060e8651dd7582761b1a0ba7611f115800d`. The independent follower read every saved file: **87 arrays /625851664 array-file bytes**, exact checksums, shapes, dtypes and finite values. RGB package manifest `34b610446a2fbf00a49b042b76fc8fc0cb557898a90c40ff7c2ba77c4c3da8b2`;freshRGB result `9a6954fc4b455e47abb5f8caafbf39a52e53d23d0da056f4f4f280b546a2b21e`. GPU-owned wall19368.1348s,worker childCPU13324.7223s. This closes the training/save/RGB endpoint, **not native policy success**.

The existing b5baee80 observer independently completed that readback and policy admission, then failed before panel registration because a local scene variable shadowed the reference-identity module. It is retained CLOSED1: **0 registered panels /0 physical requests /0 model loads /0 updates**. The earlier import failure is also retained. No training restart or valid-condition retry followed.

New source `d2f77ddeb648bf5e6645b3e5747806c7e9266203` only repairs that local name and adds a registration-path regression. Twenty-four relevant checks pass. A new finite observer03 verifies old observer terminal, both panels absent, unchanged inference dependencies and original closed3000 artifacts before dispatch. The same87 arrays were read by the new observer,0.5660s from cache. Original fixed TRAIN8 is now dispatched;at the snapshot **1/8 started,0 terminal**, first native preparation tick1750. Independent DEV48 remains0 started/48 planned. Running conditions are neither failed nor successful. The original checkpoint, layouts,restarts,seeds,horizon,raw consumer and native safety stay fixed. The follower retains its original stop rule for all-entry-rejected TRAIN8 and technical/safety failures;DEV48 is not guaranteed to dispatch.

## Old RP8 still failed; retain the later engineering negatives

Original first condition TRAIN01-P10-r5:00 failed state reconstruction at5703 before any oracle injection. The original startup first diverges at tick12. Its remaining seven fixed RP8 conditions remain unstarted. The original negative is not overwritten by the new episodes.

Recorded callback export captured4122 callbacks through5703. Two explicitly versioned candidate binaries reuse original SDK/solver/controller and unchanged control/physics math but **are different binaries**. A build-receipt finalization failure and a stale-catalog preflight both remain recorded with0 physics. Build02 source d51ebf80 compiled binary `dbaa7650ecf2d3daff20b6284e634e88f8380639ff238730b6d3c91fca6dc22a`;build03 source4e8fad01 compiled `d009f122002ddd019f628701b2fbf1787bb7463896b5df4357279c5780acde99`. Compilation is not native qualification.

Callback diagnostic02 reaches5260 ticks then fails EXACT_NATIVE_PAUSE_NOT_REACHED:the gate blocked sensor/clock refresh needed for the next policy. The new version moves the gate after the original clock/GetState/pause ACK,without extending safety or matching tolerances. Diagnostic03 reaches5703 but still fails the original1e-9 captured-state gate. Reference/past-reference/base errors become9.99e-16/2.55e-15/0,while qpos/qvel/history do not match. The first2503 state packets approach machine precision;**tick2504 first diverges in arm control indices12–17**,maximum0.185528. Logs show differing first MPC target captures at4.996s versus5.000s. That localizes the unresolved arm handoff;it is not proof of a specific complete solver-state cause. Original selected-policy ID and complete MPC restore are unqualified.

Across the original and two repair diagnostics: **3 native requests /3 terminal failures /16666 initialization integrations /0 oracle injections /0 oracle-consumed points /0 SLA queries /0 updates**. They are separate from demonstration-generation denominators. No reanchoring,state overwrite,tolerance relaxation or further blind reconstruction attempt. **Old RP8 remains failed;S16 remains0/500 updates,0/16000 visits,0/32 predictions,0/4 student diagnostics.**

## Retained endpoints, next actions and boundaries

The generation pool is now **122 full /141 terminal generation-worker requests**,137 positive-integration worlds,4 zero-tick failures,2852786 integrations. This includes the new4 and the old118/137 pool;RP8 engineering diagnostics are separate. FormalD100/D300 and PICK_PLACE admission remain false;all earlier contact and pacedPLACE foot-displacement failures remain.

Original2EP SF remains3000/96000 committedT/0S (96016 gradient visits including discarded16),full105/726615568 bytes and freshRGB16 verified. It is not a matched20EP control. BOOK21 remains500/0T/16000S,full87/625851664 bytes and freshRGB16 verified,0 target native requests. Old N20 TRAIN8 remains0/8 (7 entry H40 rejects,1 initialization technical failure),and oldDEV/TEST are unchanged. H200716146 stays unknown without duplication;other users' healthy GPU activity is untouched.

Next receive the original V6 TRAIN8 and conditional independent DEV48 outcomes. Old RP8 needs a verified complete control snapshot or prefix before another attempt. New PD4 statistics still need the complete native chain before training;the authorization to advance first4 does not automatically expand collection toPD20 or start optional training arms. No S16 or matchedSF was started behind a failed dependency.

Relevant callback/capture/dual-view checks and the latest24 registration checks are overlapping regression sets,not a summed capability metric. GitHub CI NOT_RUN. Frozen WBC/legs/gains/cost/solver,native2ms/safety,oldDEV/TEST and historical negatives remain. No healthy capsule hotpatch,RL,real robot,force push or auto-merge. The goal remains active and incomplete.

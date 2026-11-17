# V5.2 actual progress — E1 complete, real E2 running

As of `2026-11-17T16:38:43.459925+00:00`,B1 and B2 each completed **500 genuine updates/16,000 optimization visits**,full-state checkpoint restoration and independent ordinary-RGB16 restoration. The campaign added1,000 updates/32,000 optimization visits,8 separately charged diagnostic windows and792 offline model generations.

**Real MuJoCo E2:9/24 new requests started,8 terminal,1 active.** In the complete first F1/T0 triplet,B1 and B2 both achieved genuine grasp/lift/hold plus required base motion;B0 stopped on illegal contact,with0 following integrations. This is5/8 among closed requests,5/9 among launched requests with1 pending. It is a prefix,not promotion. No candidate is selected.

| Stage | Actual state |
|---|---|
| D0 |320 inputs,319 actions,1,376 C20 labels,8 experts/550 plans;0 new generation/integration |
| Pretraining split |36 sealed new layout clusters,DEV32/TEST40;205,308 passive FK checks;no new dynamics |
| B1/B2 |500/16,000 each;all87 GPU reload bit-exact;all28 active arrays changed |
| Independent RGB |B0/B1/B2 each16/16 bit-exact;no optimizer loaded |
| Saved TRAIN/CN38 |All three96/96 and180/180;CLOSE20+36 per arm,0 omissions/late events |
| ROLL96 |B0=93/96,B1=93/96,B2=96/96;original rejected3 repaired0/3,1/3,3/3 |
| E2 native |9 started,8 closed;5 genuine successes,1 contact STOP;15 remain unstarted |
| Confirmation/E3/DEV/TEST |Conditional;0 started |

See [actual matched endpoints and costs](E1_PAIR_ACTUAL.md),[offline distributions/gates](E1_OFFLINE_ACTUAL.md),[real native prefix and counters](E2_PREFIX_ACTUAL.md),[D0 evidence](D0_BOUNDARY_CONTACT.md),and[retained native-profile cold repair](E2_PROFILE_COLD_REPAIR_ACTUAL.md). The complete24-condition matrix and fixed selection/negative decision remain unfinished.

Fixed plan `90d39196bdc345b977ee54ed64e750cf3561aa72`;reviewed PR403 head `dbf222b41cb666bff4dd5412b864d023aec69399`. Training source `8ebdcbb7ab39c0f707c5b2ba94b0b097abc937e5`;current immutable inference/native source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`. Common SHIFT7 parent `ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb` and frozen statistics `f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e`. Source publication and offline legality are not task-success evidence.

The healthy server service `wm02-sla-mobile-f2-v52-e1-e2-followup-20261117-02` runs under verified linger with no frontend dependency. Its fixed deadline is unchanged at epoch1794967938.2895093(Nov18 10:12:18.289509 SGT);the campaign/allocation deadlines are unchanged. It automatically runs the remaining fixed E2 suffix, with no scientific replay, extra training, confirmation, E3 or DEV/TEST automatically scheduled. PID/cgroup,private logs and cooperative STOP entries are retained in the private detach receipt.

Both zero-update E1 environment failures and the zero-scientific-effect native-profile repair are preserved. Old followup01 is terminal;new02 retains the reviewed native interpreter instead of the learner interpreter. No healthy capsule, frozen binary or saved result was hotpatched.

WBC/legs/MPC/solver/arm PD,original completeH40/P10/TTL/2ms,collision/STOP and scores remain fixed. Pure MuJoCo uses the project-owner authorization already recorded in V5.1;no new third-party signature is required. Real hardware,RL,oracleBASE/phase,action clipping/smoothing and scientific retries remain forbidden. PR4033/8,F2 0/4,failed V5.1 pilot and old RP8/JOIN10/R3/history remain unchanged.

New [live H40 rejection attribution](E2_REJECTION_PREFIX_ACTUAL.md) identifies B2/F1-T2 interior index34 at0.261575m/s²;head/tail remain legal. The [conditional evaluation implementation](CONDITIONAL_EVALUATION_READY.md),source `eecf53b3`,passed61 focused CPU checks but is not deployed or dispatched. The active capsule remains source `2797d0c2`.

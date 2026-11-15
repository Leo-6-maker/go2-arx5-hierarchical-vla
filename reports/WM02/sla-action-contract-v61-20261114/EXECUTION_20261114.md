# SLA V6.1: first native state-reconstruction failure and actual endpoints

Snapshot **2026-11-14T15:49:18.392684+08:00**. Fixed plan `6e6636577d53b6b25a92b1e36cf2e619e765eab3`,PR383,unmerged. Execution remains Draft PR384;the goal is unfinished. The independent implementation worktree was based on the current reviewed runtime plus prospective HOLD50 source58d87107,not main. The four plan/entry files were copied exactly on dc288f5b. All running source capsules remain immutable.

## Actual native finding: RP8 has not passed

The first preflight found an archived scene-builder digest mismatch before a worker existed:0 physical requests/updates. The original producer was independently CLOSED0,its exact builder bytes still matched its proof,and the model/frame/role were unchanged. The existing archived-builder validation path was used in a fresh source/version after verifying that custody. The failed preflight and journal intent were retained and reconciled;no unknown effect was replayed.

First actual condition: **TRAIN01-P10-r5:00,PREGRASP,anchor5703**. Runtime `2b82156584c6bb8326ed362111171c21d16a06e8`;one native worker request,one terminal failure,5703 initialization/preparation integrations,**0 oracle injections/0 oracle-consumed atoms/0 SLA queries/0 updates**. Clean shutdown,no native safety stop. Remaining seven fixed RP8 conditions are unstarted after the first-world diagnostic failure;they are not counted as failures or successful replays.

The same native binary,solver,controller,model,libraries,frozen files and seed pins were used. Replaying startup to the same absolute tick did not reproduce the recorded control state. At5703,TCP translation differed **1.1686219647mm**,rotation **0.0028801069rad**,and reference/history/base also differed. The predeclared existing captured-state/frame component gate was1e-9;no widening,reanchoring,state overwrite or failed-condition retry was performed. A read-only comparison of both original state journals found all state fields bit-exact for only the first11 packets;**qpos/qvel/ctrl first diverged at tick12**,before any SLA label. Both runs' first two recorded leg-policy input/history hashes agree. This localizes the mismatch before label injection;it does not prove a specific scheduling/RNG/solver cause or that the label itself is rejected.

The current `control_snapshot.py` is explicitly an offline typed container with `native_exporter_bound=False`;`native_state_snapshot.py` provides coherent sensor reads,not a complete restore. The old MuJoCo-only replay explicitly disclaims live controller/solver restoration. Historical Python-engine fork/restore code is another backend and supplies no GO2/OCS2 restoration qualification. A complete original control-state restore or verified replay remains necessary;metadata or qpos/qvel reconstruction cannot fill this gap. Next investigate the tick12 control/clock divergence and actual exporter coverage under the frozen low-level stack,before any new versioned diagnostic.

## Actual student-file conversion: 1220/1220

A separate finite Core CPU worker CLOSED0 under source `9af0b639`. All1220 real normalized FP32 containers were written/read,decoded with the existing GripLogitCodec,serialized to real ACTION.npz files and passed through the unchanged raw student compiler and original full-H40 geometric envelope. Original recorded BASE7 was used. Dataset `29fd45ff31a2031f70f5d32402e6bfefe2f8b644e8e81ad1d2003d4537a1a6f5`;full N20 frozen statistics `024951afc22e03586635f17b5ee8f473aead1960f107d8768384643844afb9e1` were not refitted.

Maximum installed-reference roundtrip error:XYZ3.28784e-9m,SO(3)1.86271e-9rad,mechanical grip4.0000004e-6m (declared logit epsilon envelope). Stages47PREGRASP/384DESCEND/80CLOSE/135LIFT/574HOLD,total1220. Actual wall74.2781s/CPU3.6175s;0 model loads/updates/training visits/physical requests. **This worker did not run the native certifier or restore control history.** It cannot establish RP8 or a working policy.

## Fixed S16 and final-plan masks

S16 is fixed at16 distinct TRAIN windows/eight layouts;RP8 at five stages plus three seams,selected before model outputs. Selection SHA `07ef8e034aace1331bd37c176a8e94d6e0ba3a0a38c25fe542fe8935bccb6a42`. S16 remains **0/500 updates,0/16000 target visits,0/32 RGB predictions,0/4 student TRAIN diagnostics** because RP8 is unsupported. No new dataset or training budget masks this native-state failure.

The new diagnostic hook records the post-EEF_STREAM final program separately from the pre-planner proposal and explicitly stores reference_before_query. The student oracle uses the raw compiler without expert regeneration. Supervision and consumption sidecars are separate,with original query/command/tick identity;renewal ACKs do not count. Pending native certification has zero admitted supervision. The unit case legalH40/P10 has40 supervised/10 consumed points,**not现场证据**. This actual failed anchor produced **0 final-plan files,0 newly supervised points,0 consumed oracle points**. No masks were hot-added to existing V6 loss/data;partial-label learning remains unimplemented.

## Inherited V6 and other actual training

V6 source3f48cadf remains healthy on GPU1: **2002 actual updates/64064T/0S/0 physical requests**,fixed3000/96000T. All87 initial model/Adam/RNG arrays matched the original N20 before update1. New checkpoints500 and1500 have now each been independently read in full:87 arrays/625851664 array-file bytes,exact hashes/shapes/dtypes/finite values and unchanged initial bindings. Their COMPLETE SHAs are respectively `f6d2f48597725169c35d301d097b0c371bce594fc84c442139e18d5343f99ee0` and `3f74b1a2a61e3474ef4b01d66dd9730e45e597c06c8c6e2bb79a87f8ad0faf4f`. These are intermediate saves,not3000 closeout or endpoint qualification.

The existing b5baee80 follower is still waiting without a GPU owner,model or physical request. It must receive original3000 closeout/full87/fresh ordinaryRGB16 and run the fixed TRAIN8/independentDEV48. RP8/S16 do not select its checkpoint or change its recipe. Original N20 negative TRAIN8 remains0/8 with seven full-H40 rejections and one initialization technical failure;oldDEV/TEST and historical negatives remain unchanged.

Old two-episode SF is CLOSED0 at **3000 updates/96000 committedT/0S**;actual gradient visits96016 include the retained discarded16. Full3000 state `e88776ab91534ceef4e69b8d2388f15c2ac0978d94c8c85e4241616208f8d715`:105 arrays/726615568 bytes independently read,freshRGB16 bit-exact. It is not matched data against20-episode V6.

BOOK21 source-only FM is CLOSED0 at **500 updates/0T/16000S**. Full500 state `5961e0f1c7f2405a7eb40280900f5f36f16d5414ab959ec4acc0763da6b9f846`:87 arrays/625851664 bytes independently read,freshRGB16 bit-exact.0 target native requests;no target book-skill success established. Readback adds0 updates/visits/models/physics. H200716146 remains unknown and is not duplicated;last A40 scheduler refusal is retained,not treated as current training.

## Prospective HOLD50 closed:40 full /50 requests

Source58d87107 campaign CLOSED0: **50/50 actual supervisor requests and terminal worlds,40 full demonstrations/10 failures**,996650 integrations,26021.3533 worker CPU seconds,10207.2683 campaign wall seconds,814317444 qualified shard bytes. All50 positive-integration worlds;no failed stratum was retried. Outcomes and failure mechanisms are retained per condition in the machine-readable report.0 SLA model loads/updates/training visits. These new raw shards remain outside current V6/S16 training admission.

Combined with the old78/87 generation pool: **118 full/137 terminal generation-native-worker requests**,133 positive-integration worlds,4 zero-tick failures,2765974 integrations. RP8 is an additional separate diagnostic denominator. This remains a PICK_HOLD component;formalD100/D300 and PICK_PLACE admission are false. The old pacedPLACE foot-displacement negative remains retained.

## Validation and boundaries

62 relevant regressions passed for the initial native route;32 affected checks were rerun after the archived-builder repair;final replay tests8/8 include refusal of changed builder bytes/live producers. These overlaps are not added into a new unique-test total. GitHub CI not run. Runtime preflight,CPU file checks,actual gradient updates,ordinaryRGB restoration and native outcomes are distinct evidence classes.

Frozen WBC/legs/gains/cost/solver,native safety/2ms and oldDEV/TEST remain unchanged. No hotpatch of a healthy capsule,no additional camera/SFT2000/GR00T qualification,noRL or real robot. The goal remains active and incomplete. [Machine-readable actuals](ACTUAL_20261114.json) retain hashes,denominators,costs,negative conditions and next actions.

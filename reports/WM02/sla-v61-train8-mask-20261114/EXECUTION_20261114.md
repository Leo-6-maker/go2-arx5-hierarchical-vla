# SLA V6.1: fixed TRAIN8 prefix, mask loss and saved-prediction diagnosis

Snapshot **2026-11-14 18:20:39 SGT**. Fixed plan PR383 `6e6636577d53b6b25a92b1e36cf2e619e765eab3`,execution Draft PR384,goal unfinished. The actual evidence covers the fixed evaluation,isolated mask implementation and a readback of existing model outputs. [Machine-readable actuals](ACTUAL_20261114.json) include all outcomes,decision identities,limits and costs;[first4/complete3K evidence](../sla-action-contract-v61-first4-20261114/EXECUTION_20261114.md) remains unchanged.

## Real student panel:0/5 complete tasks,130 confirmed commands

The same finite observer03 and original TRAIN8 group are confirmed live by their actual process handles. **8 planned /6 physical native requests started /5 terminal /1 running /2 unstarted**. All five terminal outcomes are valid policy failures,with0 technical failures and0 hard safety stops. The completed rows contain **18 genuine SLA queries /130 original solver-gripper-confirmed commands /3250 student-execution integrations**. Including28515 preparation integrations,the closed-row total is31765. Counts exclude the running sixth world's unfinished model/physics work.

| Original condition | SLA queries | Confirmed student commands | Native integrations | First failure |
|---|---:|---:|---:|---|
| TRAIN01/r3 | 10 | 90 | 7953 | Full-H40 EEF preview rate/acceleration gate |
| TRAIN01/r4 | 3 | 20 | 6203 | Original joint acceleration certificate |
| TRAIN07/r3 | 2 | 10 | 5953 | Original joint acceleration certificate |
| TRAIN07/r4 | 1 | 0 | 5703 | Original joint acceleration certificate |
| TRAIN13/r3 | 2 | 10 | 5953 | Original joint acceleration certificate |

Every consumed command has its exact original decision-file hash and contiguous25-tick interval verified;the failed next proposal is not counted as installed or consumed. Closed pair CPU totals1880.3675s,with per-condition native/neural wall/CPU in the JSON. All remaining original conditions stay in the denominator;no success-selected substitute,retry,checkpoint selection or extra training. IndependentDEV48 remains0/48 started;its original conditional stop/deadline rules remain effective.

The model is still the closed V6 endpoint **3000 updates /96000 target visits /0 source visits**,source3f48cadf,COMPLETE `b01d15817391f7e74971678ad3304060e8651dd7582761b1a0ba7611f115800d`. Full87 saved arrays/625851664 bytes and freshRGB16 were already independently verified and were not requalified here. Native runtime remains d2f77dde. No Teacher fallback,expert regeneration,endpoint basis,clipping or safety relaxation.

## Existing ordinaryRGB outputs: small position error is insufficient

A finite CPU readback sourcef7754260 reused the **16 already saved step3000 predictions**,original eight fixed TRAIN probe windows ×two noise draws. It verified original array hashes and the bound dataset/statistics/label bytes. **0 new models /0 new RGB generations /0 updates /0 visits /0 physical requests**. Audit wall1.6736s/CPU0.0317s;whole worker wall10.7524s/childCPU0.2514s.

All16 XYZ RMSE values are below0.65mm and mean SO3 errors below0.00043rad. Yet only **12/16 pass the original policy-frame EEF geometric rate/acceleration limits**. PREGRASP3/4,DESCEND4/4,CLOSE3/4,LIFT0/2,HOLD2/2. Two fail translation acceleration and two fail mechanical finger speed;largest nominal-grip MAE is0.03356. No limits changed. All40 points,including the unused suffix,remain checked.

These are **one episode/eight windows**,not S16,independent development or a native/joint certificate. The CPU checks preserve the nominal-to-mechanical grip conversion and original50ms derivatives. Actual native execution still needs its original frame/reference/base history,full compiler,joint/IK/contact certificate and WBC chain. The finding supports retaining executable-trajectory checks in addition to pointwise error;it does not identify a complete cause or justify new optimization budget,smoothing or data changes by itself.

## New mask loss: implemented separately,synthetic autodiff checked

`masked_plan_loss_v61.py` adds a separate FM32 loss;no existing engine/inference module imports it. Host admission requires FP32 `[B,40,32]` targets and boolean `[B,40]` supervision,nonempty rows and finite supervised values. It rejects all-false rows before model work. Unsupervised targets are replaced before flow noising,and their action keys are excluded from attention at every layer while original state tokens and H40 positions remain. The loss averages each row over its supervised knots and all32 dimensions,then averages rows equally. All-true batches use the original reduction. Active7/padding25 diagnostics stay separate;consumed_mask is absent from the learning API. This changes no native H40 gate.

Two finite Core **CPU synthetic attention/autodiff assays** closed0,sourcesa37e0f9a thena62d041a after adding explicit valid-forward assertions. Both load **0 real SLA parents/adapters/teachers/optimizers**,perform0 optimizer updates/visits/physics and hold no neural GPU owner. The second checks:

- All-true loss difference0;parameter-gradient max difference1.4901161193847656e-8,within the explicit1e-6 relative/1e-7 absolute comparison.
- Replacing every unsupervised target by1e6,NaN ornegative infinity preserves valid-point forward predictions,loss and parameter gradients bit-exactly;unsupervised target gradients are exactly0.
- Invalid attention keys are absent;original state-token visibility and H40 positions remain.
- Unequal supervised lengths normalize per row;FM32 still includes padding25. A bypassed all-false device row yields nonfinite loss rather than a fabricated zero-loss update.

Final assay wall45.2932s/CPU3.0908s;whole worker wall167.5824s/childCPU7.0069s,including import/CPU scheduling overhead. Earlier assay wall8.6156s/CPU3.1227s;whole worker wall93.4201s/childCPU6.4878s. These are **synthetic mathematical checks,not real-SLA gradient or training qualification**. The new loss is not yet bound into a PD data batcher/learner;no dataset was admitted or candidate trained. Real-model qualification within the declared budget remains required. Latest33 relevant local regressions pass;overlapping earlier sets are not added. GitHub CI NOT_RUN.

## Remaining scope and next actions

Continue the untouched original TRAIN8 and conditional independentDEV48. Keep **old RP8 failed** and S16 at0/500 updates,0/16000 visits,0/32 predictions and0/4 student diagnostics. Original control-state restore remains unqualified;the old three diagnostics/16666 initialization integrations/zero oracle consumption are retained separately.

The user-authorized first4 expert PICK_HOLD episodes remain4/4,9760 supervised/2560 consumed points;the total generation pool remains122/141. Two PD4 views each244 windows have frozen-statistics FP32 CPU checks,**not new-statistics native qualification or training admission**. PD20,matched N20 spatial training,new real-model mask qualification and new candidates were not automatically started. Existing2EP SF/BOOK endpoints,oldN20 0/8,oldDEV/TEST,contact and PLACE failures stay unchanged.

Frozen WBC/legs/weights/gains/cost/solver,native2ms and all safety gates remain. No healthy capsule hotpatch,RL,realrobot,force push or auto-merge. The goal remains active and incomplete.

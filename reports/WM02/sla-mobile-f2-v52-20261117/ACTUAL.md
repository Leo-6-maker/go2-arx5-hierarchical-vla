# V5.2 actual progress — B1 sealed, B2 running

As of `2026-11-17T15:03:02.534508+00:00`, B1 has completed **500 actual updates /16,000 optimization visits**. B2 is at the verified prefix **100 updates /3200 visits** on the same GPU. The campaign has generated96 new TRAIN predictions. **New native requests/integrations are0/0; no new mobile-grasp success has been established.** The full E1/E2 objective remains incomplete.

Plan `90d39196bdc345b977ee54ed64e750cf3561aa72`; execution PR405 independently inherits reviewed PR403 head `dbf222b41cb666bff4dd5412b864d023aec69399`.

| Stage | Actual state |
|---|---|
| D0 | 320 inputs,319 actions,1,376 C20 labels,8 experts/550 plans,2,501 saved contact states;0 new generation/integration |
| Pretraining split freeze | 36 new layout clusters: DEV32/TEST40;205,308 static initialization-state FK checks;0 new dynamics |
| B1 lambda0 | 500/16,000 complete;4,000 micro4 backward calls plus one diagnostic over4 windows |
| B1 restoration | Actual all87 GPU reload bit-exact;all28 adapters changed;checkpoint/package bytes separately verified |
| B1 TRAIN96 | 96/96 original H40;20/20 CLOSE events within100ms;maximum50ms |
| B2 lambda.1 | Verified prefix above;fresh Adam and same true SHIFT7 parent |
| Independent RGB16 / CN38 / ROLL96 | Pending finite followup;scientific failures cannot remove an E2 control |
| E2 | Fixed24 requests preregistered;0 started |
| Confirmation/E3/DEV/TEST | Conditional;none started |

See [B1 endpoint identities and measured costs](B1_ENDPOINT_ACTUAL.md), [D0 boundary/contact evidence](D0_BOUNDARY_CONTACT.md), and [B0 saved-prediction baseline](B0_OFFLINE_REUSE_ACTUAL.md). B0 remains96/96 TRAIN,180/180 CN38 and93/96 ROLL;its three historical H40 rejections still block promotion but do not remove its eight E2 conditions. No saved ROLL prediction is an executed trajectory or a same-state expert correction.

The two saved pre-update states were independently byte-compared: all87 model/Adam/RNG arrays match, together with parent,visit map,LR,noise,time,dropout and first indices. Training source is `8ebdcbb7ab39c0f707c5b2ba94b0b097abc937e5`;matched LR SHA `7f10a93cbac3bda375845b99b1f50c4eb516cb5d49a8d5e2e17f6e1b956298c9`. The only objective change is lambda0 versus.1, with margin.20 and original execution threshold.25. Both endpoints are fixed500;step250 is recovery-only.

B1 model manifest: `b9ba974c374b3c3a09d7742004fc799337d5e7f1b9c9c7d290af066766e4706c`;result: `6ab3b21ed91789aa883a88d8e875396c20b30910013f403b24377e33e6b4adf6`. True common SHIFT7 parent: `ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`;C20 manifest: `4bcc5ffcd8920d874599a54cca73792c50a49ea6013d65ca7826817a9f80d7de`;statistics: `f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e`.

Measured B1 service cost:3369.86s wall/2829.06s CPU,host peak14,365,962,240 bytes,swap peak0,JAX live-buffer peak9,714,233,344 bytes. GPU busy-seconds were not separately measured. D0 cost6.13s wall/3.02s CPU;split preparation278.73s wall/54.95s CPU. The split uses predeclared local outward offsets8–38mm with unchanged material/camera/robot,minimum5mm separation from190 historical initial centers and between new clusters. This is the declared local population;no TEST model output has been read. The CPU-only input-format audit read186 input archives,90 truths,276 baseline archives and verified16 existing RGB output hashes,with0 new generation/update/native request.

Server followup `wm02-sla-mobile-f2-v52-e1-e2-followup-20261117-02` is actually live under linger,with an independently advancing heartbeat and no frontend dependency. Source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`;spec `9d0d02c39749219116498a85ec0323e0a2c78cf89e6cee9813a3d7f3421eaf01`. It waits for B2's closed actual500 endpoint,then performs exactly600 new offline generations (48 independent RGB restorations plus B1/B2's276 saved-input conditions each) and the fixed24 E2 requests. It has no automatic retry or automatic confirmation/E3/DEV/TEST. Original deadlines remain;detailed PID/cgroup/log/stop receipts are private.

Two zero-update technical failures are retained: B1-01 missing upstream dependency path,and B1-02 inherited native LibTorch ABI search path. Both ended before model load,optimization visits or generation. Their B2 siblings were verified unstarted and closed. New03 identities fixed the offline environment;no old capsule was hotpatched or scientific outcome replayed.

Frozen WBC,legs,MPC,solver,arm PD,H40/P10/TTL,2ms integration,collision/STOP and scoring remain. Hardware,RL,oracle BASE/phase,output clipping/smoothing and scientific retries remain forbidden. PR403's3/8 student success,F2 0/4,failed V5.1 pilot,and RP8/JOIN10/R3/history remain unchanged.

Next actual deliverable remains both fixed500 endpoints,independent ordinary-RGB restoration,the complete24-condition paired native matrix and fixed selection/rejection,then the plan's conditional successors.

The zero-scientific-effect [native profile cold repair](E2_PROFILE_COLD_REPAIR_ACTUAL.md) preserves the original native NumPy/MuJoCo/ROS stack. Old followup01 is terminal and retained;new02 is independently verified live,using the exact same deadline and unstarted600+24 suffix. Training units and model endpoints were untouched.

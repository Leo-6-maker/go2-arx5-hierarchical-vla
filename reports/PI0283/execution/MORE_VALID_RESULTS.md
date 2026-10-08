# PI0283 MORE: valid results, EVENT continuation in progress

Authority: canonical 6048886543, release 508e3db7b63d3c47758794f3ffbad87b577e6e9d. The user removed wall-clock deadlines; finite updates, queries, physics, GPU time and technical recovery caps remain unchanged. This is an intermediate report, not PI0283 completion.

## Actual quality

| Point | Fixed TRAIN prefix fit | XYZ median | SO3 median | Gripper median | Persistent source success | First success |
|---|---:|---:|---:|---:|---:|---:|
| ACTION2048, retained predecessor | 1/24 | 17.335 mm | See original PI0282 receipt | 0.00452 | 0/4 | 2/4, one control sample each |
| MORE1024 | 1/24 | 22.881577 mm | 0.014252 rad | 0.005615 | 0/4 | 0/4 |
| MORE2048 | 2/24 | 15.243788 mm | 0.012808 rad | 0.001099 | 0/4 | 1/4, t=10.80 to 10.85 s |

All eight MORE episodes completed the original 32-second task window. Independent CPU computation reproduced the native on-top/static/not-grasped predicates, exact current observation inputs, saved noise bytes, installed references and achieved TCP. All four conditions at each point have zero success samples in the original 41-sample 28–30 s qualification window. No DEV or CONFIRM was opened.

The MORE2048 first-success condition is 10003. It lost success after 0.05 s and remained grasped throughout the final hold window. MORE2048 has sustained grasp in 3/4 and any on-top in 2/4. A descriptive >2 cm/0.3 s lift measure passes 2/4; the historical lift rule was not recorded and is explicitly UNOBSERVED, so this descriptive result is not a new qualification gate.

## Exposure and optimization evidence

MORE consumed exactly 4,096 microbatches / 16,384 planned draws: 12,285 unique UIDs and 4,099 repeat draws. Each phase has 4,096 draws. Existing-command event counts: close-first8 1,472; open-first8 792; close-tail-only 2,705; open-tail-only 2,702; stable-closed 4,068; stable-open 4,655. Categories can overlap; they are not a disjoint denominator.

The independent complete-Trainer audit verified exact parent moments and global Adam steps, both actual LR groups, native loss/reduction, input/noise/time identity, row10-only projector updates and DiT updates, with frozen tensors and other31 rows unchanged. Native loss first16-update mean 0.0362233, last16 mean 0.0333367. Reduced loss and a better median do not establish task success or a normalization defect.

## Conditional EVENT, within original budget

The frozen transition test was met by actual saved outputs: TRAIN fit 2/24 <18/24; release error >0.1 in 4/4 fixed release probes; approach XYZ >0.01 m in 4/4 fixed approach probes. EVENT1024 uses the complete MORE2048 parent, carried Adam moments, the sealed event plan, unchanged native loss/precision/scope, and the authorized phase schedule.

Job 653762 completed MORE but stopped before any EVENT forward because a legacy guard demanded an unauthorized extra EVENT canary. The bounded second technical recovery reuses the authentic MORE carry-mechanism proof and the full MORE2048 parent audit; it does not fabricate an EVENT canary or repeat MORE. EVENT-only job 653917 began on H200 at 12:40:01 SGT. Technical training recoveries are now 2/2; prior GPU time is conservatively retained as 5,823 s, the EVENT allocation reserves 3,177 s and Corelab reserves 5,400 s, totaling the unchanged 14,400 s cap.

## Complete recovery and archive

Both MORE checkpoints remain complete local Trainers on Hopper and have public fixed-revision full-file readback:

- MORE1024: `arms/PI0283/MORE/checkpoint-1024`, revision `54ba8f2ba4001ef587b58a5d9353b3b34947aae6`, 14 files / 20,468,175,924 bytes.
- MORE2048: `arms/PI0283/MORE/checkpoint-2048`, revision `11e413b687bb101f5066bb3d8ccc3f97acf84e77`, 14 files.

Repository: `leoyuer/pi0246-gate-pass-campaign`. Only consumed duplicate export tensors and completed, unqualified Corelab inference views were retired after identity checks; unique datasets, observations, training histories and full recovery sources remain protected. Representative two-camera videos use saved actual observations and explicitly identify source Panda/native PD/non-realtime/not WBC.

Next: complete EVENT1024, independently audit its full Trainer, execute the original DIAG48 and TRAIN4, then route the unchanged qualification gates. Execution completion and quality PASS remain separate; selected/deployment=NONE.

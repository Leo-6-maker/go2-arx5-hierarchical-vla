# PI0260-R1: fixed DEV stratum exhausted

**BLOCKED_SCIENCE — PI0260_ELIGIBLE_SCENE_SHORTFALL.** H0 passed, but only **4/20** DEVELOPMENT families satisfy both commands of both behaviors; the fixed requirement is **6**. The original20-proposal stratum is exhausted. No TRAIN/VAL borrowing, camera change, reference/IK change, replacement sample or new model branch was performed.

Scientific authority90cbb926a842f08c3f9a78756d7a24e0d7a63f78; workflow amendment1f27d9c14ce0a91518930a8d8a2f6f0ffc2dd468. Review mode **SAME_OWNER_READONLY_REPLAY**. Historical PI0258/PI0259 failure verdicts remain unchanged.

## Actual result

| Stage | Result |
|---|---|
| H0 | First rig `[.65,0,.75]`,84 actual image pairs; original cue gates and saved-array checks pass |
| Fixed DEV proposals |20 evaluated,80 sequences attempted; no replacement |
| Complete geometry |45/80 sequences: M1 29/40, M3 16/40 |
| All-four complete geometry families |8/20; the remaining12 fail this first stage |
| Raw physical gate |All32 sequences in those8 families pass eligibility; other families not rendered |
| Observation eligibility |4 of those8 families pass;4 fail M3 endpoint visibility |
| Jointly eligible families |000,004,009,013 — **4/20, short by2** |
| TRAIN / KIN-VAL / processor / canary |NOT_RUN: DEV admission failed |
| Base6000 / G6000 /1760 evaluation |**Do not exist. Overall model objective is not complete.** |

All35 incomplete trajectories end at the fixed solver's `IK_RESIDUAL` condition. This is failure of the prescribed bounded continuation on these proposals, not proof of global kinematic impossibility. No alternate branch or restart was attempted. All saved accepted output states passed replay of the original collision checks.

The four complete-geometry observation failures are:

| Family | M3 c0 visibility | M3 c1 visibility | Required |
|---|---:|---:|---:|
|005|91.875%|91.875%|≥95% plus required states/quartiles|
|006|91.250%|91.875%|same|
|014|89.375%|89.375%|same|
|018|91.875%|91.875%|same|

Failures are the black/white endpoint falling below4 actual pixels during motion (roughly8.8–11.2s across these sequences). The fixed calibration poses passing H0 did not establish visibility throughout every fresh trajectory. No conclusion about GR00T capacity, model quality, physical tracking or WBC success follows; no model was queried.

## Saved-array verification

DEV manifest `65bca938ec673605392cd54cefbdf9ca0de7bd934ce914d986e16c1f0fd4f059` binds4710 files/222,609,912bytes. Same-owner replay receipt `024592ae2c19020151424464310da9b9382d9d89b09a65789e6c50910083186c` reconstructs the same4-family result from raw arrays:

- 11,237 saved states: actual FK, hard/velocity bounds, original full-link/obstacle/ground FCL collision replay.
- 2,134 unique production image pairs: original96×96 RGB, masks/depth, FK/StateV1, complete observation indices, camera transforms, original pixel rules and shared starts.
- 4,192 original physical-gate windows, including terminal audit windows.3,520 are active generation windows.
- Independent algebraic reference reconstruction matches saved references exactly; FK delta0; StateV1 max delta3.33e-16.
- Original proposal order, numeric exclusions, stratum partition and selected-family order checked; negative tests reject producer-PASS contradicted by arrays, missing first quartile, incomplete geometry and failed mandatory states.

Shared low-level FK/FCL/gate primitives are disclosed; this is not a second agent's independent signature. The pinned URDF SHA256 remains96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25.17 private mesh file hashes were inventoried at closeout; no preexisting per-mesh comparison is claimed. Private assets were not published.

## Actual cumulative usage

84 calibration pairs;80 new kinematic attempts;11,399 bounded IK solver calls;2,134 production pairs;3,520 active raw windows. Model queries, optimizer updates, WBC/physics/TEST/DEV-F/hardware all **0**. Two pre-image CPU entry errors consumed0 image/IK/model budget. No duplicate dispatcher or live training job was found. Remaining allocations do not authorize scientific redesign or stratum borrowing.

## Durable artifacts

- [H0 fixed HF archive](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/1e019362f25f0308cfbca7ada2efd5e88c9a6a0e/evidence/PI0260/H0-motion2-r1-20260930):317 members, tree5b88faf12cdf970446e9f3765f0b0da270e818470f1258dd6dcbc1aca9cfcddb.
- [DEV fixed HF archive](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/6173fde8b088f32e39b481e356ae0e7aa77f4bbd/evidence/PI0260/DEV-motion2-r1-20260930):4738 members,225,577,763 source bytes; treee5212ba604d9755fd464134ca017eaf9ec0c678851b83a971a96477985d7bead.
- Both are PUBLIC and passed anonymous complete-file and every-member SHA256/size readback. DEV compressed archive89,699,919bytes, SHA25631de686c54c98415817b567705d3c510c8ae4940eabe20f9450a56de024b6619. The archive helper's legacy `producer_H0_manifest_sha256` key carries the DEV manifest hash in the DEV receipt; it does not refer to the H0 manifest.
- PC results: `D:/Users/GR00T project/Projects/go2-arx5/PI0260/delivery/index.html`, with161-frame16.1s reference comparison and all20 proposal outcomes. It shows the first eligible versus first observation-failed family in frozen order, not selected model examples. All161 frames fully decoded; first failed video state at8.8s manually inspected (failed black endpoint3px versus17px in the paired eligible example).

The video is **KINEMATIC REFERENCE / NO MODEL PREDICTION / NO PHYSICS**. No microtuned-model comparison can be supplied because training was not admitted. selected_checkpoint=NONE; deployment_selected_checkpoint=NONE. No cleanup, monitor/mail or merge.

Current authority requires stopping this dependent branch. Continuing to the requested model endpoints needs a substantive scientific amendment to the unmet fixed condition; an ordinary execution GO or absence of an auxiliary reviewer does not resolve it. This is not `PI0260_KIN_MOTION2_PILOT_COMPLETE` and not a model-quality FAIL.

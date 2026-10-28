## PI0260-R1 — verified fixed-DEV shortfall; preserve evidence and stop dependent training chain

Continuation of H0 canonical5894006224. Science90cbb926 / workflow1f27d9c, unchanged cumulative PI0260 budget. Result commit **10bbef10a00c43a3197a0f6d9fb07a3469596c53**. [Full result and provenance](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/10bbef10a00c43a3197a0f6d9fb07a3469596c53/reports/PI0260/DEV_RESULT.md).

**BLOCKED_SCIENCE — PI0260_ELIGIBLE_SCENE_SHORTFALL.** Fixed20 DEVELOPMENT proposals exhausted; only **4 jointly eligible families**, versus required6. No borrowed TRAIN/VAL proposal or replacement design. H0 first-rig PASS remains true; PI0259 OBSERVATION_DESIGN_BLOCKED remains unchanged.

- 80 real sequence attempts:45 complete geometry (M1 29/40, M3 16/40).35 terminate at the original bounded IK_RESIDUAL condition; no alternate IK restart.12 families fail joint complete-geometry eligibility.
- 8 families have all4 complete trajectories and pass raw-gate eligibility.4 then fail M3 observation coverage: families005,006,014,018 have c0/c1 endpoint visibility91.875/91.875%,91.250/91.875%,89.375/89.375%,91.875/91.875%, below95%. Missing black/white endpoint pixels during motion are preserved, not filled.
- Only000/004/009/013 qualify,4/20. TRAIN, KIN-VAL, processor, canary, Base/G6000 and1760 model evaluation are **NOT_RUN**. No claim of model-quality FAIL or complete model experiment.

**SAME_OWNER_READONLY_REPLAY** independently reconstructs outcomes from saved arrays, with shared original math/FCL/gate primitives disclosed:11,237 FK/collision states,2,134 image pairs with camera/state/pixel checks,4,192 gate windows, all80 reference time grids, numeric split/exclusion/order and negative tests. Reference/FK deltas0; State max3.33e-16. DEV manifest `65bca938ec673605392cd54cefbdf9ca0de7bd934ce914d986e16c1f0fd4f059`; [replay receipt](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/blob/10bbef10a00c43a3197a0f6d9fb07a3469596c53/reports/PI0260/DEV_READONLY_RECEIPT.json) SHA `024592ae2c19020151424464310da9b9382d9d89b09a65789e6c50910083186c`. No second-agent signature is claimed or required.

Actual cumulative compute:84 calibration pairs;80 kinematic attempts;11,399 bounded IK solver calls;3,520 active raw windows;2,134 production pairs. New model queries/updates/physics/WBC/TEST/DEV-F/hardware all0. Producer and CPU replay exited; current user Hopper PBS listing empty. No healthy job was restarted. This is a real frozen-data condition, not missing auxiliary availability.

**PUBLIC durable evidence:**

- [H0 fixed revision1e019362f25f0308cfbca7ada2efd5e88c9a6a0e](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/1e019362f25f0308cfbca7ada2efd5e88c9a6a0e/evidence/PI0260/H0-motion2-r1-20260930):317 members, anonymous complete-file/member readback PASS.
- [DEV fixed revision6173fde8b088f32e39b481e356ae0e7aa77f4bbd](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/tree/6173fde8b088f32e39b481e356ae0e7aa77f4bbd/evidence/PI0260/DEV-motion2-r1-20260930):4738 members/225,577,763 source bytes, anonymous complete-file/member readback PASS. All failed proposals, raw arrays, receipts and display retained; private URDF/meshes/WBC source excluded.

Results display: `D:/Users/GR00T project/Projects/go2-arx5/PI0260/delivery/index.html`;161-frame16.1s video (fully decoded, PC SHA verified), first eligible versus first observation-failed M3 reference in frozen order. Watermarked **KINEMATIC REFERENCE / NO MODEL PREDICTION / NO PHYSICS**. At the first displayed failure8.8s, black endpoint is3px; original threshold4. The corresponding eligible example is17px. No new inference/rendered scene for presentation.

Per parent§4 and R1§5, stop this dependent branch. A substantive scientific amendment is required to change the unmet fixed population/design condition. Unused budgets or ordinary execution GO do not authorize sample borrowing or redesign. Overall goal is **not achieved**, not PI0260_KIN_MOTION2_PILOT_COMPLETE and not STOP_FOR_PHYSICAL_TRANSFER_ADMISSION. selected/deployment NONE. No broad cleanup, mail/monitor, WBC or merge.

# PI0259 Agent M: actual H0 calibration delivery

Authority: 240dedfd3954dd4cbbb48645a2aad5b19738e2c4 / canonical5892082658.
Source baseline: PI0258 closeout8e1276d, preserved unchanged.
Producer branch: pi0259/observable-kin-mt-execution-v1.

## Actual work and current gate

36 new, explicitly quarantined calibration sequences and144 actual96x96
body/wrist image pairs are complete. Native integration, WBC, model loads,
queries and optimizer updates are all0. No fresh DEVELOPMENT/TRAIN/KIN-VAL
sequence has been started. H0 is sealed and delivered to Agent V for independent
correctness review. No common rig currently satisfies the registered criteria.

| Geometry calibration | Complete/qualified sequences | Actual limiting evidence |
|---|---:|---|
|M1|10/12|Family002 both commands exhaust fixed continuation with residual failure|
|M2|0/12|Original x5_link7/8 contacts original grey obstacle during final segment|
|M3|8/12|Families000/002 both commands fail bounded continuation in early approach|

These are calibration outcomes, not fresh-data admission counts. Complete
desired references remain for all36. Failed sequences retain accepted prefixes,
every attempted internal solve and explicit NaN/unrendered suffixes. No failed
solution is propagated as an accepted state.

| Fixed rig index (protocol order) | Tasks supported at all shared starts | Later evidence |
|---|---|---|
|0 (first)|M3 only|M1 axes/target and M2 goal/obstacle counts fail necessary thresholds|
|1 (second)|M2,M3|M3 black endpoint falls below4 pixels in14 of32 available registered poses; missing geometry separately indexed|
|2 (third)|M3 only|M1 axes and some M2 obstacle counts fail necessary thresholds|

Rig1 has40 M1,36 M2 and32 M3 later-pose frames; all three rigs have6
shared-start frames plus6 exact reproduction frames.144 total. Even ignoring
missing geometry, rig1 cannot support two tasks: M1 fails at common start and
M3 has observed endpoint visibility failures. No fourth rig, marker resizing,
scene replacement or new trajectory has been tried.

## Findings, checks and preserved corrections

- Actual original default root/DOF FK gives p0=[.2901,0,.5966057] and R0=I.
  Old nominal/identity differences are exactly0 for this source state. The
  hypothesized default-start mismatch is not observed here; the new reference
  still derives both position and orientation from actual FK.
- New quintic segment timing, residual-function central Jacobian, negative DLS
  step and depth4 continuation follow the new authority. CPU fixtures cover
  nonzero root pose, analytic/full FK, near-pi branches, residual/geometric
  Jacobian sign (max discrepancy5.43e-11), failure propagation, accepted-prefix
  retention, quintic joins, velocity and collision negatives.
- Original hard bounds, collision components/exemptions, dimensions, task
  durations, wrist camera, gate and WBC remain. M2 collision is actual link
  geometry against the original box, not inferred from endpoint error.
- The initial camera report incorrectly required8 visible contour sectors at
  later poses as well as common starts. CPU-only replay corrected this to the
  protocol's common-start requirement. Original RGB/masks/reports remain;
  no new rendering or scientific attempt was used. Endpoint failures remain.
- Native camera process wrote all144 arrays and its final JSON, then exited
  with SIGSEGV during shutdown. Hashes, saved readback and144 rasters were
  replayed without rerendering. Exit failure is reported separately from
  individual valid data outcomes. Agent V independently checks the evidence.
- An earlier import-path error produced no scene or image and is retained.
- Some H0 fields are intentionally unavailable: no TRAIN-only processor,
  codec/model/recovery evidence can exist before the data gates. Raw gate
  records are provided for every available current FK; missing FK rows and
  terminal-hold audit rows are explicitly labelled.

## Immutable packet and archive

Corelab node-local root: /var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0259.
H0 manifest: handoff/H0_v1/manifest.json.
Manifest SHA256:
94954516f9b1e9159e7fe468c04d7155e6cc20eb4d21c601b8389e8ab2dcac62.
491 entries include raw arrays, cue-ID/geometry, view/projection, full root/link
FK, StateV1/native anchor, numerical solver ledger, checks and full outcomes.
Agent V has the same-machine private URDF/FK binding for independent CPU replay.

Public repo: leoyuer/pi0246-gate-pass-campaign.
Prefix: evidence/PI0259/H0-observable-fk-calibration-v1-20260929.
Revision: 6c0d169637f836190eeb7250eee018d9a392bb43.
492 public members /30753688 source bytes; full anonymous file/member readback
PASS. The private dependency binding is explicitly excluded, retained on
Corelab; public member manifest records this exclusion. No private URDF/mesh,
WBC source, model/tokenizer weights or credentials are published.
See H0_PUBLIC_RECEIPT.json for file hashes.

The three-rig contact sheet is display-scaled recorded RGB, labelled
KINEMATIC REFERENCE - NO WBC EXECUTION. No visual overlay enters policy RGB.

## Boundaries

Preflight GPU0 UUID fef928e1-b12e-2f20-4b2e-a868e6e0faad:8139/32760MiB
used; Corelab node-local946GiB/shared378GiB free; Hopper no user job and370GiB
free. QuotaUNKNOWN. Next writes remain far below measured disk capacity with
10GiB reserve. No scheduler job, unrelated process interruption, cleanup,
mail/monitor, TEST/DEV-F, native episode or merge.

Two6000 endpoints and full evaluation remain conditional and unachieved.
Agent V correctness findings will determine any same-contract engineering
repair. Current pixel/geometry outcomes cannot be retried for better scores.
selected_checkpoint=NONE; deployment_selected_checkpoint=NONE.

## Independent Agent V closure

Agent V returned CORRECTNESS_BLOCKED / COMMON_RIG_OBSERVABILITY_NOT_ADMITTED
for the exact H0 manifest. Unmodified companion AGENT_V_H0_RECEIPT.json has
SHA2561e16c79ad6d7c1f98318cf8d151378b27b050dcaeb2068c3874b815c8fabcde7.
It independently checked491 files,144 rasters and18 exact shared-start repeat
pairs. FK discrepancy max2.22e-16, State comparison max2.11e-8 and camera
rotation-matrix difference max2.88e-7.4466 cue/camera depth backprojections
match saved world sphere surfaces within0.194mm (tessellation diagnostic,
not a new scientific acceptance threshold). FCL contacts in V's receipt are
explicitly AS_RECORDED; V did not independently rerun the collision library.

The independent visibility counts agree with the producer's corrected report.
No reported pose/camera implementation defect supplies an authorized repair
that would change the observed admission failure. Agent M therefore stops at
OBSERVATION_DESIGN_BLOCKED, with calibration geometry failures also preserved.
H1/H2 and the two6000 endpoints were not released or completed. This is not
KIN_MT2/3_PILOT_COMPLETE or STOP_FOR_PHYSICAL_TRANSFER_ADMISSION.
Any additional camera/reference/geometry candidate requires a new scientific
amendment; unused budgets do not authorize a fourth rig or valid-failure retry.

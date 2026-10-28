# Local-PC Adoption Report

Status: **ADOPTION_READY_FOR_REVIEW**. Scope: **ADOPTION_ONLY**.
Observed: 2026-10-12, 23:08-23:18 Asia/Singapore.
Owner: this local-PC chat. No cloud owner, subagent, automation, or competing goal was created.

## Evidence Anchors

Repository: Leo-6-maker/go2-arx5-hierarchical-vla.
Active branch: pi0267/executor-decoupled-recovery.
Live read HEAD and handover release: `3d3cecaa5a5d8b9c6a17158912b6f5816702b678`.
Main navigation: `1c1a57f77007358c3e0231ab39677a727c244bb0`.
Old navigation was also read at `1f05bfc919b25d3146839529d2763ac847010d7d`.

Issue #11 was refreshed with since=2026-10-10T03:08:35Z, per_page=100, pages 1 and 2.
One incremental comment: [6126486784](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6126486784), updated 2026-10-12T14:39:38Z; page 2 empty.
Issue #47 incremental pages 1 and 2 returned only #6126502857; page 2 empty.
These are adoption/maintenance records, not new scientific GO.
PI0289 terminal #6093151230 was separately read in full. Its latest scientific close remains 2026-10-10 11:08:34 SGT.

## Research Understanding

The upper team owns independent object-task physics, expert EEF/gripper data, pretrained GR00T adaptation and reactive object-task evaluation.
The WBC owner independently develops the low-level controller. Future composition uses an explicitly frozen WBC identity; this adoption does not tune its gains, solver, reference preview or weights.

GR00T observes dual RGB, language and actual TCP/gripper feedback. Its EEF10 output is XYZ + row-rot6d + gripper, with relative EEF and absolute gripper, H40 at 20Hz, K4, followed by eight committed points and fresh feedback after 0.4s. Internal 132D masks do not mean 132 joints.
Desired, accepted, installed, committed and achieved states have distinct identities.
Object task success includes acquisition, lift, transport, placement, release, withdrawal and sustained completion; first arrival and trajectory error cannot replace it.

Panda/source, independent dynamic tool, fixed-base target geometry, and frozen-WBC execution must remain separate data domains.
Independent-task admission should have a new manifest and provenance contract. The existing target_mujoco validator correctly requires its original WBC gates; do not rewrite old FAILs or relax that validator to admit another source.

## Actual Scientific Terminals

| Experiment | Actual outcome | Boundary |
|---|---|---|
| PI0285 | PLACE4096 + DOSE2048 = 6144 formal updates; TRAIN7/8, DEV13/16, CONFIRM_A20/32 below26/32 | Valid confirmation quality FAIL; source Panda NON_REALTIME; no deployment |
| PI0286 C0 | 1024 updates; TRAIN2/8; DEV and new confirmation NOT_RUN | DEV is not0/16 |
| PI0286 C1 | 1024 updates; TRAIN8/8, DEV5/16; new confirmation NOT_RUN |27/32 correction admission is not generalization |
| PI0289 | safe native startup10.000178504s PASS; hold XYZ0.011672954m PASS, SO3 0.065199717rad FAIL; clock0.047803157s exceeds0.025s | Other five references NOT_RUN; formal target teacher/data0 |

PI0286 arms used a common parent and paired training/initial/noise identities. The older parent used a different evaluation noise namespace; parent-vs-arm score differences are not strict paired effects.
C1 training gains did not meet DEV qualification and its offline label error was not lower.
**selected=NONE; deployment=NONE.** Old deadlines, unused allowances and old HPC gates are not inherited.

The WBC comparison describes a different control structure: old IsaacGym learned body/arm actors versus new ROS2 Jazzy + MuJoCo + OCS2 + leg RL. The default native MPC is a simplified floating-arm kinematic optimizer; EEF-only7D or EEF+BASE14D pose JSON is distinct from legacy201D observations/15D actor actions.
GR00T EEF10 requires SE(3) decode and a separate gripper channel. Observation-time anchoring, reference expiry and sim/ROS/wall clocks still require explicit qualification.

TaskAnchor-inspired causal history/task-state conditioning and UNITAS-inspired metric calibration/object consequences are designs only.
Start with B0 native current observations, then equal-budget B1 causal history, then B2 predicted task state if justified. Object truth, teacher phases and success signals must not leak into formal Student inputs.
There is no TaskAnchor/UNITAS implementation or project benefit established by this adoption.

## Verified Resources And Assets

- local -> corelab-amd-server -> corelab-4080s and local -> hopper SSH are operational, using existing keys and host checks.
- Corelab shared/SSD projects, original private WBC delivery, SSD candidate and PI0289 original receipts exist. Local215,194,129B WBC archive hash exactly matches6dd13098... handover.
- Hopper official GR00T base cache, source and training venv exist. Official source HEAD is51d4c89f72fda44cbf77285c6a8114b52676b8a1 and was clean at read.
- C1 has14 local Trainer files totaling20,468,176,035B. C0 retains metadata and lacks both shards/optimizer locally. Corelab keeps model-only BF16 views. These are research artifacts, not selected policies.
- Both endpoints authenticate to HF as leoyuer through their existing explicitly configured token location. The historical hub is PUBLIC repo_type=model.
- Both endpoints read five exact revision/prefix inventories and five small JSON files, totaling50,732B of payload per endpoint. C0/C1 Trainer metadata hashes match archived receipts; PI0289 manifest matches the Corelab saved receipt. No large weights/video/archive download or HF write occurred.
- Source-bank/processor/statistics/correction manifest small hashes agree across endpoints. Source bank records160 admitted T1/T2 episodes,128 TRAIN used for statistics, no DEV in statistics. Original bulk RGB, arrays and77MB window index were not exhaustively revalidated.

## Resource Exceptions And Unknowns

Hopper qstat is available via its installed PBS client, version2024.1.0. Two own RUNNING jobs662514/662515 belong to a separate vla_wbc project by output path; no adoption allocation is claimed, and neither was touched.
Corelab GPU activity belongs to other users. Two owned PI0276 archival processes remain in stateT (paused/stopped), with old log modification. Their health and the existing AMD Codex task identity are not established; neither was resumed/killed.
No held go2-arx5 lock was found in the bounded lslocks view; empty historical lock files alone do not prove a lease.

Storage display: Core NFS1.4T free, SSD508G; Hopper scratch93G free at91% used, home20G free. These are filesystem readings, not personal quota or a new scientific storage allowance.
Personal quota remains UNKNOWN: Core command unavailable, Hopper quota mount errors/no usable headroom.
The task-data-preparation directory named mujoco-3.2.7 contains only pip package metadata in the bounded inspection; a complete independent physics runtime is not yet verified.
HF write scope, private destination for new restricted data, raw-history/calibration completeness and correction-data HF transfer pin need resolution before any reuse/collection GO.

## Branch Maintenance Actual Result

Nine heads remain. The four allowlisted live SHAs match; API comparisons place each candidate on its keeper ancestry; default/main excluded, protected=false, rulesets empty, open PRs0. No same-name candidate tags existed among164 observed tags.
**Tags created0; heads deleted0; dry-runs0. All four SKIPPED.**
This new local machine has no resolved Git authentication and no gh installation; authenticated HTTPS ls-remote failed. The connected GitHub tool can publish reports but has no tag-create/head-delete action. Therefore the required tag-before-delete and exact expected-SHA lease push could not be performed.
See BRANCH_MAINTENANCE_RECEIPT.json. Maintenance is explicitly partial, with this concrete blocker; nothing is reported as completed deletion.

## Next Review

NEXT_DATA_GOAL_DRAFT.md is **NOT_AUTHORIZED**.
Recommended first bounded GO: independent dynamic gripper + object contact task, complete episode recording and data provenance/admission, with model loads/forward/updates and WBC calls kept0. Use Corelab physics and Hopper only for later explicitly allocated neural work.
NETWORK_AND_EVAL_PLAN.md separates verified SSH/HF reads from the proposed dual-SSH relay. No relay, tunnel, model service or physical test was started.

This report is ready for adoption review. Wait for a new scientific GO. It does not restore any historical experiment.

## Read Register

All entries below were read at the fixed active SHA in the requested order, followed by bounded supporting source reads. Older main navigation is historical only.

- START_HERE.md (blob aa0c2e859ed41b904e8a3d273bc08cb050cf0357)
- docs/handover/2026-10-12_LOCAL_PC_ADOPTION.md (blob b80f9f79a52b60923e023f1f45d0703411ecc4d2)
- docs/handover/2026-10-12_BRANCH_RETIREMENT_PLAN.json (blob 7ae676493c9188dfba025cdd743ff6ec59a5c6e0)
- docs/handover/2026-10-12_BRANCH_CLEANUP_RUNBOOK.md (blob b88da313311496bf6a9bcf242c058318acdf4a63)
- docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md (blob 183f64ec39635bd9a555ed5fe2556a889a25fa00)
- reports/PI0285/r3/FINAL_REPORT.md (blob e9def1582217342759a56198df074a545c0ff077)
- reports/PI0286/final/FINAL_SCIENTIFIC_REPORT.md (blob 3d092e076054c6847a6161c56540865f1bf463ca)
- reports/PI0289/execution/REPORT.md (blob 105b5868c57f7a1f243a24224caa6ad17e0be022)
- reports/PI0289/execution/GATE_RECEIPT.json (blob 3459466ce441689d46b8dda2a4316aa4a542722e)
- docs/coordination/2026-10-10_WBC_PACKAGE_COMPARISON_AND_TARGET_LEARNING_REVIEW.md (blob 8e84c3f712d1151c542941dd89703dac6fa352ce)
- docs/coordination/2026-10-12_TASKANCHOR_UNITAS_DATA_AND_FINETUNING_REVIEW.md (blob efd51bd79e16980b016f04b30d3e0556d297cc9b)
- scripts/pi0286_execution/environment.sh (blob ba83f104ec108d2c6d7b8c50d96fc1bacd9cfb64)
- scripts/pi0286_execution/C0_REPLAY1024.pbs (blob b5ef8d1748c849d13c0c1eb89d8ff8e460b11772)
- scripts/pi0287_target_sim/owner.py (blob 43bc412d6600af4f0768d649dc96131255386a3a)
- reports/PI0286/final/FINAL_MODEL_INVENTORY.json (blob 7765f3b55d95fa1ffd1382fc5985cab043962422)
- scripts/pi0286_execution/policy.py (blob 54b781c810d69a2ffb7cfe5d4eede00e633085b6)
- scripts/pi0286_execution/data.py (blob fe7cf4366d435ebdffbcb4c4bd08cb799473313e)
- src/go2arx5_vla/target_mujoco/dataset.py (blob f6a7342a4a72bbe6a9317ab27fc8522facd6c621)
- src/go2arx5_vla/target_mujoco/teacher.py (blob 3120cbd2615e2921f057ead84898f9cd7ef28e8d)
- src/go2arx5_vla/target_mujoco/recorder.py (blob fb59b89c7274a09fe51f783eeaa763cf64483453)
- src/go2arx5_vla/target_mujoco/scorer.py (blob 095b5a7295fac632a35d45da8510ced06eb42807)

Additional reads: PI0286 final delivery/full-readback receipts, stage_common/transport_setup/contract and archive publishing scripts; main PROJECT_STATUS/RESEARCH_INDEX/REPOSITORY_WORKFLOW at fixed navigation SHAs.
Old launch/owner/doctor/archive code was inspected, never executed.

## Session Accounting

New PBS jobs0; GPU programs0; GR00T loads/forward/updates0; teacher0; physics0; WBC0; hardware0.
Remote checks were SSH, metadata/process/PBS/storage reads and limited HF GETs. Credentials were neither requested nor printed.
Private addresses, usernames, absolute resource paths and probe details stay in the local private runmap. The six committed files contain sanitized aliases and small evidence.


## Maintenance Completion Addendum: 2026-10-12 23:44 SGT

The earlier SKIPPED result above remains the historical first attempt at commit4944ea53d8efd0e65abefc31ee575a4fad110452. After the user resumed pushing and explicitly authorized local-PC GitHub configuration, the existing project GitHub account was verified as Leo-6-maker and a repository-scoped local Git credential helper was configured. It reads the existing Hopper credential transiently over authenticated SSH; no plaintext token was saved locally or printed. Authentication depends on Hopper SSH reachability.

Full, non-shallow Git ancestry metadata was fetched into an isolated bare repository without checking out or rewriting user files. All four allowlisted live SHAs matched, all were ancestors of their keepers, open PRs0, protected=false, rulesets empty, and bounded owned-process/lock checks found no active candidate branch dependency. Existing paused archive processes, other-project PBS jobs and AMD Codex were preserved.

All four same-name lightweight compatibility tags were actually created and read back at their exact expected SHA before deletion. Each head then passed its own exact expected-SHA lease dry-run and actual deletion, followed by authenticated readback showing the head absent and tag unchanged. Final API inventories confirm9->5 heads and164->168 tags. main, the active branch and the three independent-commit branches were retained. No tag was overwritten, keeper force-updated, local historical refs pruned, model/data removed, or scientific work started.

BRANCH_MAINTENANCE_RECEIPT.json now records the complete actual result and preserves the first-attempt pointer. RESOURCE_MAP.json records the resolved local authentication and its SSH dependency. All original scientific limitations and NEXT_DATA_GOAL_DRAFT=NOT_AUTHORIZED remain unchanged. selected/deployment=NONE; state ADOPTION_READY_FOR_REVIEW.

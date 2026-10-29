# GR00T + WBC agent entry · main integration

## Read before work

Read `START_HERE.md`, `CURRENT_SCIENCE_STATUS.md`, `docs/EXPERIMENT_ROADMAP.md`, `docs/CODE_MAP.md`, `docs/REPOSITORY_WORKFLOW.md`, and the exact current task protocol. Refresh Issue #11 science and #47 operations receipts before execution. The 2026-10-28 user-directed consolidation promotes `main` as the single integration/publication branch; `infrastructure/workspace/INTEGRATION_REF_CHANGE_20261028.json` supersedes older branch-routing text only.

The canonical human workspace remains `~/Projects/GR00T-WBC`. Read `docs/operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md`, `docs/operations/WORKSPACE_ENTRY_ENFORCEMENT_V1.md` and `infrastructure/workspace/ENTRY_POLICY.json` for unchanged workspace/admission semantics. Historical policy snapshots and active run seals are not rewritten by this promotion. Do not git-init the whole workspace, create new Home pi roots, or copy another scheduler/path registry.

## Publishing and execution are different

- Publish new bounded work to main through reviewed normal commits/PRs. The former `pi0267/executor-decoupled-recovery` remains only as a temporary compatibility ref until its publisher adopts main; do not silently delete or overwrite it.
- One integration publisher and one mutation owner per shared checkout. Fetch before writing, preserve concurrent commits, never force-push/reset/mirror overwrite. A merged repository does not mean all historical implementations are qualified for deployment.
- Immutable running source capsules never pull/checkout/move/delete. Code-management checkouts change only at a verified boundary with dirty/local-only work preserved. Pin full code/upstream SHA, source seal, entry release, dataset/model revision, processor/statistics, environment, authority, owner and run UUID.
- Repository merging, branch retirement, documentation and CPU tests are not new GPU/PBS/model/physics authority. Preserve all prior costs, clocks, leases, FAILED/UNKNOWN evidence and protected resources. Never restart a healthy or already closed job to reflect a Git change.

## Fixed platform route

Vanda runs legal GPU training and finite offline neural diagnostics. CoreLab 4080S runs policy inference and the existing MuJoCo closed-loop evaluator. **AMD is shared-storage and login infrastructure only**, following the direct human instruction recorded in [AMD infrastructure protection](docs/operations/2026-10-29_AMD_INFRASTRUCTURE_PROTECTION.md). It permits bounded SSH forwarding, small project metadata/Git work, limited reads and controlled archival transfers. Do not start Teacher production, physics, training, model inference, video encoding, bulk decompression/conversion, compilation or large recursive scans on AMD. The PC or a qualified compute node performs that work; historical AMD Teacher outputs remain valid and preserved. Runtime processes must not require Core-to-Vanda SSH, the other site's live queue or heartbeat. Reuse presealed finite allowances, local durable effect records and independent terminal reconciliation through the existing Journal/onceIO. Do not invent another scheduler or weaken identity/safety checks.

Never modify or restart AMD/compute-node NFS, autofs, SSH, network, mounts, exports, disks, host power or system packages as an experiment repair. Do not invoke desktop eject, UDisks/polkit/D-Bus disk-management actions, privilege escalation or the host Docker socket. Administrator-controlled recovery stays with the administrator. Preserve existing SSH configuration, private keys, authorized keys, permissions and credential stores; do not copy, rotate, delete, print or publish credentials. Use the existing key entry and already authorized memory/anonymous-pipe credential route only. A failure is a bounded observation, not permission to repair infrastructure, delete files, replay UNKNOWN writes or submit another job.

New Core execution must verify node-local absolute paths for code, interpreter/environment, libraries, model and data inputs, working/temp/cache directories, control state, locks, heartbeat, logs and output. A local cache symlink alone is insufficient. Stage/check before launch and archive after terminal verification; do not hot-migrate running or sealed historical jobs. This policy does not assert that current deployments already meet every node-local requirement, add a new scheduler or authorize infrastructure changes.

Use only already authorized fixed-revision HF model delivery; do not disclose private datasets, WBC/vendor assets, credentials, paths or environments. Source Git contains code, schemas, sanitized small evidence and indexes, not weights or raw videos. A public model route does not authorize publishing unrelated private material.

## Scientific invariants

- Prioritize measured behavior. Complete original C50/D25 and GEN1 evaluations; conduct matched raw-parent E0/E1/E2 research. Unknown/unrun outcomes are not 0% success; training completion and offline loss are not task success.
- Current E1/E2: same original pretrained parent, same twelve source TRAIN demonstrations, matched cursor/noise/optimizer schedule, fixed first512 of a1024-step schedule. E1 trains only target-category encoder/decoder leaves; E2 additionally trains declared shared action components. No extra C1/GEN1/target data or silent change of R0, horizon, scoring, augmentation or dropout semantics. Conditional continuation extends both own complete states, not only the losing arm.
- Read `docs/protocols/GR00T_TASK_DATA_FIRST_FROZEN_WBC_SCOPE_V1.md`. WBC code, weights, gains, costs, solver and leg policy remain frozen. Historical hold RMS failures remain diagnostic, not global task-data prerequisites. Do not start another WBC tuning, bias-compensation or attribution project.
- Successful demonstration admission requires actual object-task success, native safety, actual observation/submitted command integrity, clocks, labels and split validity. No future achieved state as hidden action relabeling, rejected command as expert label, unexecuted suffix as executed supervision, or fixed-base zeros as BASE labels.
- Go2+ARX5 target demonstration research is a separate version and scope. Training on target-WBC data changes the claim to few-shot target-domain adaptation; preserve the no-target-task-training DEC1 comparison independently.
- Ordinary engineering fixes may proceed only within the existing valid scope and cold runtime boundaries. True external permissions, unsafe effects or changed scientific variables require an explicit decision; routine CPU progress is not a substitute for native evidence.

## Handoff and maintenance

Deliver assistant-authored plans and machine cards in this GitHub repository; capture a full commit SHA, read back changed files and provide immutable links. No local-only handoff ZIP or invented PASS. New run admission uses existing RUN_MANIFEST/profile/lease/registered outputs before effects. Private keys and host-bound assets remain outside Git.

Keep current entry/status concise; detailed dated receipts stay in their original paths. Preserve historical source and negative results. Branch retirement requires an exact head, main ancestry, no open PR/protection/active publisher dependency, a verified recovery tag and expected-SHA deletion. Current cleanup deletion is pending until a real receipt exists. Root-only owner and paused monitoring remain as last instructed; this file does not restart subagents or automations.

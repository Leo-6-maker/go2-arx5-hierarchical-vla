# GR00T + WBC: October 12-14 Work Report and Owner Handoff

Report assembled October 14, 2026; fresh metadata-only server readback completed before publication. This is a user-requested code publication and handoff, not a new experiment GO. The incoming agent replaces the local-pc owner; no second cloud owner, agent, model service, tunnel or scheduled restart was created.

## 1. Read This Before Continuing

**Scientific state: BLOCKED_REAL_NATIVE_FINGER_JOINT_LIMIT_NO_P1_GATE. Objective incomplete. selected/deployment=NONE.** M512 exists and trained for 512 actual updates. Its first fresh-observation native ARX5 trial stopped at a real high-rate finger joint-limit violation. No M2048/M4096, DEV model trial, independent confirmation, regression panel or additional full-state training ran in R1.

Read in order:

1. This report and [code/evidence index](CODE_AND_EVIDENCE_INDEX.json).
2. [R1 final report](../../../reports/PI0294/20261014T025730Z-localpc-pi0294-r1/FINAL_REPORT.md), [resource ledger](../../../reports/PI0294/20261014T025730Z-localpc-pi0294-r1/RESOURCE_LEDGER.json), [actual model evidence](../../../reports/PI0294/20261014T025730Z-localpc-pi0294-r1/M512_POLICY_EVAL.json), [saved gripper audit](../../../reports/PI0294/20261014T025730Z-localpc-pi0294-r1/SAVED_GRIPPER_TRACE_ACTUAL.json).
3. [Original R1 task card](../../../reports/PI0294/20261014T025730Z-localpc-pi0294-r1/GO_TASK_CARD.json) and [human adoption #11](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-6159886188).
4. [Pending gripper safety proposal](../../protocols/PI0294_R1_GRIPPER_SAFETY_REQUALIFICATION_PROPOSAL_V1.md) and [machine task card](../../../reports/PI0294/design/GRIPPER_SAFETY_AMENDMENT_TASK_CARD.json). Their status is PROPOSED_AWAITING_EXPLICIT_HUMAN_ADOPTION; authorizes_execution=false.
5. [Unified workspace policy](../../operations/2026-10-13_UNIFIED_ENTRY_AND_GITHUB_SYNC_V1.md), AGENTS.md and private local handoff. Refresh #11/#47 latest comments, all pages as needed.

Historical sections of RESUME.md retain obsolete preparation statements such as authentication pending, no PBS submitted, or source46bf. The actual terminal evidence above supersedes those statements. Do not blindly run the historical Next Execution paragraph.

## 2. Repository, Runtime and Ownership

- Active branch: `pi0267/executor-decoupled-recovery`. Before this handoff publication, refreshed HEAD was `21db430c6f00915697d9a74bbc40ca68fef6b29c` (proposal/task-card only). The final handoff commit is recorded in GitHub issue receipts, avoiding a self-referential commit hash here.
- `main` readback was `ff50eeb9b9635b2932c765168d55fe0077880d04`: shared CPU CI/navigation, not current scientific runtime. Do not launch experiments from main or mutable branch HEAD.
- R1 actual runtime: `021d813fd669561a24ffbf89431a4a9812e0f2ca`; saved-gripper CPU sidecar: `aa9664cd7e55773ad7c6a91a561944f3df8bc04f`. The latter did not modify the runtime or numerical recipe.
- R1 source seal: `7f785c5b06ee2f1424a4709a07dbd5cac0327d55f1ffeea59730fcbda28c08ac`.
- Core manifest: `4db8828d1a1eab3e63c071f38c1c68ecacd678ff658821a559a56e2ecbba0d54`; Hopper manifest: `0762019a64c4462a0c5168692ffa16151d03f63d86e64eebdd84afdd4e76a26c`.
- Eighteen numerical files stayed byte-identical to neural parent `a171997a704072e127d16410ba584c8aa0ddd573`; common physics stayed identical to `f1fdf3acd481687cb275334ba512ce038eba51e9`.
- local-pc is the sole scheduler/publisher, SSHing each server separately. CoreLab owns physics, teachers, CPU data/scoring/video. Hopper owns neural inference/training within legal PBS. The servers do not communicate directly.
- WBC is the senior collaborator's frozen private executor. We did not run or modify it. Independent tool and fixed-base ARX5 results are not Go2, integrated WBC, Panda transfer or hardware success. TaskAnchor/UNITAS remain design references, not implemented achievement.

## 3. October 12: Adoption and Independent Tool Pilot

The fixed adoption release and research charter/PI0285/PI0286/PI0289/WBC interface documents were read; existing local SSH configuration and private runmaps were used to identify real routes. Existing authentication was reused without publishing credentials, re-downloading all historical models or creating another cloud owner. The initial adoption report, resource/HF maps and network plan are under `reports/handover/2026-10-12/local-pc-2317-sgt/`.

Four specifically authorized ancestor branch heads were actually deleted only after compatibility tags and exact-SHA checks. The historical receipt records nine to five heads and 164 to 168 tags. October 14 readback sees seven heads because two engineering branches were added later; the four retained compatibility tags still match. Main, active source and the three independent-commit historical branches were retained.

PI0290 commissioned a separate opposed-finger dynamic tool task in MuJoCo3.2.7: 2ms integration, 20Hz observations, H40 chunks, prefix8 execution, finite contact fingers and software CPU rendering. Four commissioning attempts retained a finger-velocity failure and negative controls; eight frozen pilot layouts passed, including four held-out layouts. Each pilot used 16,000 steps, 641 RGB pairs and 601 H40 windows: 4,808 total pilot windows. Independent FK/contact/calibration/codec checks and private HF raw/derived readbacks were completed. No GR00T model was trained in this phase. See `reports/PI0290/execution/REPORT.md` and D0-D4 receipts.

## 4. October 13: Native ARX5 Dataset and Training Readiness

PI0291 built a fixed-base six-joint ARX5 scene with actual finite motor/contact fingers, geometry, camera calibration, TCP/gripper codec and a teacher. The original goal was TRAIN64/DEV16; network-interrupted collection produced only TRAIN56, with eight TRAIN and all16DEV unattempted. Eight pilots passed. Interruptions and terminal PID/exit evidence were preserved; network failure root cause was not proven. The authorized 60-minute CPU closeout audited/exported the existing56, without recollection: 33,656 H40 TRAIN windows, zero DEV, full official processor/collator qualification and private HF delivery. PI0291 remains PARTIAL, not an80-episode completion. See `reports/PI0291/20261013T023717-localpc-arx5/FINAL_REPORT.md`.

PI0292 then collected the authorized24 successor episodes, DEV16 followed by TRAIN8. Together with the unchanged original56 this produced TRAIN64/DEV16, 38,464/9,616 H40 windows. Statistics were fit on TRAIN only; DEV clipping was reported rather than hidden by refitting. The v2 HF bank reused existing members, independently verified new transfers and passed cross-host CPU real-RGB/processor/collator comparisons.

Training preparation reused the official GR00T/Hugging Face Trainer through adapters, not a newly invented optimization engine: native loader, balanced consumed cursor, complete model/optimizer/scheduler/RNG inventory, smoke continuation comparisons, checkpoint capacity accounting, offline codec gates and PBS admission. The old project expired; user-selected CFP04-CF-007 was verified. Scratch capacity and real amgr expiry blocked submissions at different times. A compute-node PATH missing amgr was an engineering failure before model load, not a mount diagnosis or a scientific failure. User explicitly allowed up to two independent account allocations while this experiment stayed oneGPU, and one additional engineering retry. PI0292 ended at its original deadline: three of four permitted submissions, 17 allocated GPU seconds, no loaded model and zero updates. Do not reuse its expired submission allowance. See its RESUME.md and S3 terminal receipts.

## 5. CPU CI, Workspace and Storage Maintenance

CPU CI failures were traced to CPU dependency setup, Pillow/safetensors, shallow historical Git fixtures, import-time optional MuJoCo dependencies, engine checks mixed into ordinary pytest, portable runtime fixtures, HF mocks and contract/test assumptions. The fixes kept engine/private-source tests explicit and preserved historical fail-closed behavior. Shared fixes reached main via PR80 without merging active scientific runtime there. Current local handoff suite: **762 passed, 14 explicitly skipped, 58 subtests passed, one existing tensor-conversion warning**. Prior PI0294-specific suite:133 tests/96subtests; remote scientific CPU workflows38511924296 and38514322002 succeeded. This is CPU contract evidence, not neural or physics execution.

CoreLab's human entry is `~/Projects/GR00T-WBC`, with deployed entry/doctor and amd-management,4080-simulation,4080-cpu-data profiles. Phase2 initially copied22/24 paths (2,206 files) while keeping sources. Later verified retirement removed33 source paths/6,648 files, used non-force Git worktree removals and restored private Git/submodule recovery metadata before source retirement. Projects remained7 entries because protected dependencies were not deleted. New recovery metadata consumed5.24GiB; migration and reduced directory counts must not be called reclaimed space. **A universal shared `workspace doctor/prepare-run/sync` package was proposed, not implemented; generic Hopper entry deployment is not PASS.** Actual entry snapshots are in the verified-retirement report.

Under explicit user September-history abandonment authorization, real deletion followed reviewed current dependency separation:

- Earlier failed BF16 checkpoint deletion:44,017,266,688 allocated bytes (~40.99GiB), a separate operation, not counted twice.
- CoreLab September NFS:93.30GiB last-link allocated candidate total; observed available-space gain93.01GiB.
- CoreLab SSD:76.03GiB candidate total; observed available-space gain75.86GiB. Shared writers/Btrfs extent accounting limit causal interpretation; do not add these to personal quota as equivalent measurements.
- Hopper maintenance actor:282 roots,193,073 regular files,24,055 directories,796 symlink objects and10aliases retired. Personal quota940.7 to707.4 decimalGB: actual233.3GB (~217GiB) quota reduction. Stage22 full Trainer states196000/200000 were permanently retired under authorization without equivalent backup. A later15-file Xet cache cleanup removed175,456,256 allocated bytes, below quota rounding resolution.
- Current code/data/base/Cosmos/environment/WBC dependencies were excluded or independently checked. Hopper historical environment check still found seven missing declared packages; no speculative mass reinstall was performed. Historical base seven-file hashes, Cosmos ten-file inventory and CPU processor checks passed.

Detailed Core receipts live in `reports/maintenance/2026-10-13-september-retirement-corelab/`; Hopper's nine reports originated on fixed maintenance branch commit4f5fe8ea9b4309b2aa8792d60475a1f1daa7d0a8 and are integrated unchanged here, with provenance. No storage deletion happened during this handoff turn.

## 6. PI0294: Multitask Data, Parallel Workers and Actual M512

PI0293 already named another remote proposal; the new adopted experiment is PI0294, not a silent relabeling of old records. T1 pick/place and T2 push share the initial scene and differ through language. F0/F1/F2/F3 cover baseline, spatial, visual and combined variants with fixed splits.

- v3.0:80 new first attempts,78 admitted and2 rejected; plus80 legacy episodes =158 native episodes. TRAIN75,726/DEV19,232 windows.
- v3.1 includes v3.0:224 cumulative new attempts,215 admitted/9rejected; plus80legacy =295native episodes. TRAIN149,048/DEV28,247 windows. The extension was144 attempts, not224 extra. Six static FOV failures made zero integrations; three T2 audit failures were retained without replacement or relaxed thresholds.
- v3.1 full CPU codec/loader0,2,4-worker checks passed after a preserved2,400s qualification timeout and separately frozen recovery. It was **not delivered to HF or trained**; R1 still uses v3.0/statistics.
- Measured admitted episodes/hour:17.194 serial,31.600 two workers,49.171 four workers (1.838x/2.860x). Serial/parallel arrays, contacts, H40 and1,282 PNG pairs matched. This is data-production throughput, not an actual GPU-training worker-speed benchmark.

M512 used official GR00T source51d4c89f72fda44cbf77285c6a8114b52676b8a1, fixed base/Cosmos, frozen backbone, original learning recipe, microbatch2/accumulation16/effective32/workers0. Horizon6144 and warmup308 stayed configured, but only512 formal updates actually happened. It consumed16,384 samples, T1/T2 each8,192, with537 trainable tensors changed. Four-step native inference and deterministic CUDA/math SDP were used. Failed numerical engineering smoke was retained before strict full-state engineering recovery passed.

Parent run20261013T142457Z-localpc-pi0294 accounted fourPBS,16engineering updates,512formal updates,3,669allocatedGPU seconds and8,316.916974HopperCPU seconds. Actual512 training time2,199.881s, loader wait257.308s, optimizer throughput0.26884steps/s and GPU~35.4GB. The parent deadline closed unchanged; no claim of6144-trained performance or successful physical skill. See `reports/PI0294/20261013T142457Z-localpc-pi0294/`.

Private HF storage403 prevented original delivery. Under separate human public-storage permission, one sanitized public full Trainer checkpoint was published with upstream licensing/notices. Model/optimizer/scheduler/RNG numerical bytes stayed unchanged; three metadata files were separately sanitized. Anonymous fixed-revision file-set/size/LFS hash and small-file actual readbacks passed. Public independent neural restore has **not** been run. Permission for public model storage did not authorize publishing raw trajectories/videos or all private assets.

## 7. October 14 R1: Real Closed Loop and Actual Safety Failure

The adopted fresh R1 started2026-10-14 10:57:30SGT, science cutoff19:57:30SGT, absolute closeout20:57:30SGT. The final hour is closeout only; queue delays, authentication, publication and handoff do not reset clocks.

The local-pc relay was actually exercised: Hopper action from current observations, CoreLab executes an8-point prefix then emits post-action observations, exact request/prefix/ACK identities, replay rejection and server-specific admission. No server-to-server connection, listener or tunnel; no pre-generated replay labeled closed-loop. Teacher transport and model transport are distinct evidence. Hopper neural evaluation was not moved to CoreLab.

First new PBS665153.hopper-m-02 exited before the first forward due to NumPy32 seed compatibility:321 conservative allocatedGPU seconds and one failed pre-forward query preserved. A seed API compatibility repair kept the fixed Torch63-bit seed and18numerical sources unchanged. Second PBS665199.hopper-m-02 used the mature auto submission path, naturally routed to small, and produced48 offline model draws that passed the actual expert/model codec gate. Native second-job memory was225GiB despite128GiB requested; normalization cause remains unverified, not a newly established queue rule.

The first TRAIN F0/000/T1 trial sent six actual fresh-observation model actions: five complete ACKs, then81 partial steps of the sixth prefix. At1081steps/2.162s, `right_slide=0.045004271727453214m` exceeded frozen0.045m: native JOINT_LIMIT. Forty-four RGB pairs/20Hz samples were saved. A low-rate last sample marked safe cannot override500Hz safety/native abort. Three remaining TRAIN cases and model T2 were NOT_RUN. Offline gate PASS is not physical task PASS.

Saved-row CPU audit reconstructed controller forces with maximum absolute error0N. All48sent gripper prefix points were in[0,1], frozen slewed targets0.042..0.044m. At the stop the last command0.0771693open targeted0.04352m; right finger moved outward0.005746m/s despite inward-0.589112N control. This supports achieved dynamic overshoot, not proven bad units/out-of-range commands. Earlier grip transitions, contact, arm coupling, servo and integration contributions remain causally unisolated. No physical replay or safety modification was performed.

Original P1 remains FAIL, SHA0c154feb382f935a9f2b00b15b99b7f65e3e32ab6be47ecf04afff7e81bfe3e5. Both PBS terminal/reconciled; own resident stopped and parent finished, Core workers terminal, no qdel/unrelated healthy cancellation. Local route terminal-handshake failure was preserved and its5,000CPU-second reservation conservatively charged as an upper bound, not measured runtime. All old reservations are zero. A44-frame2.2s private video was fully decoded/hash-copied locally, not padded into a nominal32s trial or uploaded publicly.

Fresh handoff metadata readback: Core statusLIVE_STOP_SAFETY, gateFAIL_ACTUAL_M512_NATIVE_JOINT_LIMIT_MAIN_CONTINUATION_GATE_CLOSED; own active Hopper jobs0; PBS2/3 used,890allocatedGPU seconds,511.035013HopperCPU seconds,55query reservations =1pre-forward failure+48offline+6live; globalstep512/newformal0/newengineering0. Core charged5,240.044481548CPU seconds, one physics start,1081integrations,44RGB pairs,12,705,997accounted output bytes. The scientific goal remains blocked and incomplete.

## 8. Fixed HF Asset Index

Use exact revisions/prefixes and existing private authentication. READY is a fixed-revision marker, not evidence that a model skill passed. Do not repeat whole historical downloads.

| Asset | Repository / prefix | Fixed READY revision |
|---|---|---|
| PI0290 raw | private `leoyuer/pi0290-independent-dynamic-tool`, `PI0290/v1-c4853f3b` | `0fd9247ba5426eb54ebf152912432302b193b479` |
| PI0290 native derived | same, `PI0290/derived-native-codec-v1` | `f70516bd0cfa541e91db6ac23b3130fe04f1851a` |
| PI0291 partial56 | private `leoyuer/pi0291-arx5-fixed-base`, `PI0291/20261013T023717-localpc-arx5/v1-57a02b49` | `c25eeadf9be0aa7d46bb6f7b65e260cbf3a305ea` |
| PI0292 v2 | same; read BANK_V2_ADMISSION/HF_BANK_PROGRESS for exact member prefixes | `6075aac0cdc44b8d252d7540efbac208701c30b3` |
| PI0294 v3.0 | same, `PI0294/20261013T142457Z-localpc-pi0294/bank-v3_0-2fd28e00` | `6aa94f4f40d791bedd6b4e75489e4ede6504d10d` |
| sanitized public M512 | public `leoyuer/pi0294-public-artifacts`, `PI0294/20261013T142457Z-localpc-pi0294/public-M512-bc28535755c4` | `c698f54d0fc422ebfdd1e2ce1fc96552b6930c52` |

v3.0 bank manifestb8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df, statisticsfb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7; publicM512 manifestbc28535755c4034c8c5f0cc10cb127cf46893f205f49aa663d6b1e417e5421f2. Public24files contain16fullTrainer members with25,541,086,052payload bytes. Original private16members total25,541,023,437bytes/inventory6fdc10c2cfb13841bbabd444aa0ab8a4a6aef09adfbffae7d5fb4dee24747a8a. Base revision2fc962b973bccdd5d8ce4f67cc63b264d6886495; Cosmos9ce19a195e423419c349abfc86fd07178b230561.

## 9. Code Published and Private Material Retained

Reusable implementation remains in `scripts/pi0290_task_data/`, `scripts/pi0291_arx5_data/`, `scripts/pi0291_delivery/`, `scripts/pi0292/`, `scripts/pi0294/` and their tests: physics/teacher/data contracts, native codecs/loaders, audit/export/HF transfer, official Trainer adapters/fullstate checks, admission/accounting, dispatcher, resident/continuation, fresh-observation relay, public checkpoint sanitization and saved gripper analysis. See the machine index for primary modules.

This handoff adds166 sanitized owner Python source references (983,976bytes), original/sanitized SHA manifest, fail-closed exporter/four tests, comprehensive report/index, top-level navigation and byte-identical integration of nine existing Hopper maintenance receipts. Two historical trailing-space lines were normalized in the public references; original private hashes remain unchanged. Sources compile, but role-redacted archives deliberately stop before imports and are **not a new qualified runnable deployment**. Local exact helpers remain private and some are destructive/non-idempotent historical tools.

Excluded from Git: credentials, private runmaps/SSH endpoints, private execution bindings, raw trajectories/RGB/videos, tensors/checkpoints, vendored WBC, upstream model repositories, venvs, large backups and machine caches. This satisfies whole-code source continuity without treating whole workspace contents as source code or publicly disclosing private assets. Existing tracked source was not wholesale rewritten.

## 10. Next Owner: Safe Continuation, Unknowns and Review Gate

The new owner should first confirm exact local private bindings and all terminal states, then refresh remote HEAD/issues. Do not rerun init/freeze/install/submit/deletion/closeout helpers just because source is archived. Avoid loading model/optimizer pickles during metadata inspection. Keep original failure evidence and immutable capsules intact.

The pending D1 gripper proposal has NOT been adopted. It proposes zero-integration G0 timing/force checks, four2x2 G1 causal CPU replays, at most two task-independent driver interventions, eight safe Teacher regressions and only then a fresh four-case M512 panel. Old D0/P1FAIL remain; no limit relaxation, clipping, arm geometry/damping retune, WBC change or v3.1 switch. A passed read-only trace audit is not safety requalification.

Original R1 ceilings remain3PBS/8H200hours/3584formal+4isolatedengineering updates/5200queries/96physical starts/1,536,000steps/61,536RGB pairs, Core40CPUhours<=8cores and Hopper16CPUhours, Hopper120GiB/Core32GiB/local1GiB new storage,20GiB personal reserve, at most2sanitized checkpoints/60GiB. A remaining allowance is not permission to bypass failed P1. Pending amendment subcaps are not additive and do not restart the original clock. After cutoff, obtain a new authorized window before scientific work.

Unknown/uncompleted: causal gripper dynamics, D1 engineering effectiveness, live T2/remaining cases, M2048/M4096/DEV/confirmation, R1 full optimizer/RNG neural continuation, independent restore from public sanitized checkpoint, public raw-data authorization, complete portable launcher/profile enforcement, current native memory-normalization cause and future Go2/WBC/hardware transfer. PI0294 original deadline and PI0292 old allowance must not be inherited as a fresh GO.

Recommended next research step is review/adoption of bounded causal safety requalification, **not more collection or speculative main training**. Maintain v3.0 and original M512/fullstate; use saved evidence first, compare Teacher/model under the same explicitly adopted task-independent executor if a new GO permits it. Leave selected/deployment=NONE. This publishing owner stops with the science block preserved; no background scientific task was created for the next agent.

# V5 execution — 2026-11-14

Fixed plan `f61305f9c1f847e460851fd067892160f71ca432`. Actual Core snapshot **2026-11-14T06:15:20+08:00**. V5 remains incomplete. [Machine-readable actuals](ACTUAL_20261114.json) retain per-condition identities, result hashes and costs; site paths, process IDs, raw data and weights remain private. All active sources are immutable and healthy owners are preserved.

## Actual endpoints

| Operation | Actual updates | T / S visits | Saved / physical evidence |
|---|---:|---:|---|
| Original R0 / R1 models | 500 each | 16,000 / 0 each | Fixed TRAIN8 finished: R0 0/8; R1 5/8 successes, 1 valid failure, 2 technical failures |
| Clean-base 2EP SF3K, GPU0 | 1167 | 37,344 / 0 | Complete step500 independently read back; 3K/fresh endpoint RGB/native tasks pending |
| Fresh named N20 FM3K, GPU1 | 818 | 26,176 / 0 | Complete step50087-array state independently read back;3K/fresh endpoint reconstruction pending |
| Fresh expert production | 0 model updates | Not optimization visits | 24/24 new production successes; 25 across production cohorts, plus11 newly sealed actor-benchmark demos |

N20 and 2EP SF use different datasets and are not a matched FM/SF comparison. H200's 2EP FM3K job remains unknown, with its last dated snapshot stated below. No queue, commit, metadata check or CPU test is called training success.

## Fixed R0/R1 native population — all 16 conditions terminal

| Fixed condition | Native integrations | Candidate raw→installed→fully consumed H40 | Result |
|---|---:|---|---|
| R0 TRAIN01 r3 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R0 TRAIN01 r4 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R0 TRAIN07 r3 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R0 TRAIN07 r4 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R0 TRAIN13 r3 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R0 TRAIN13 r4 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R0 TRAIN25 r3 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R0 TRAIN25 r4 | 5,703 | 1→0→0 | Valid policy failure: go2arx5_vla.world_model.task_learning_student_v1.StudentProposalRejected: TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:Original nominal gripper range; no clipping or fallback |
| R1 TRAIN01 r3 | 21,203 | 15→15→15 | Success: NATIVE_STAGE_PASSED |
| R1 TRAIN01 r4 | 21,203 | 15→15→15 | Success: NATIVE_STAGE_PASSED |
| R1 TRAIN07 r3 | 21,203 | 15→15→15 | Success: NATIVE_STAGE_PASSED |
| R1 TRAIN07 r4 | 14,228 | 9→9→8 | Valid policy failure: ValueError: ORIGINAL_FULL_NATIVE_CONTACT_REJECT:['task_right_pad', 'task_cube_geom'] |
| R1 TRAIN13 r3 | 20,128 | 15→15→14 | Technical failure: TimeoutError: AUTO1_FINITE_ENDPOINT_OR_OPERATOR_STOP |
| R1 TRAIN13 r4 | 0 | 0→0→0 | Technical failure: ValueError: FUSION_PAUSED_QUERY_BRIDGE_EXACT_REGISTERED_SCOPE_LEASE_STATE |
| R1 TRAIN25 r3 | 21,203 | 15→15→15 | Success: NATIVE_STAGE_PASSED |
| R1 TRAIN25 r4 | 21,203 | 15→15→15 | Success: NATIVE_STAGE_PASSED |

R1 is **5/8 requested, 5/6 technically valid**; its one valid contact failure and two technical failures remain in the requested denominator. R0 remains0/8 with8 valid nominal-gripper rejections. Evaluation adds0 updates or expert demos. There were15 real native worlds plus one worker-only zero-integration condition, **185,995 terminal integrations**, and **107 raw queries →99 installed candidate H40s →97 completely consumed H40s**. Exact complete-prefix solver/gripper confirmation covers3,880 commands. The two failed prefixes separately retain21 and16 recorded partial atoms; neither is a completed H40. Initialization cycle−1 is excluded. Five successful conditions separately contain20 confirmed terminal-hold atoms each,100 total; those are not extra model queries or candidate H40s.

All conditions are TRAIN diagnostics. R1 uses its registered endpoint-basis compiler: these are composed-policy grasp/lift/hold successes, not proof of direct raw-curve execution or independent generalization. The original score and native safety remain fixed. First success:20.618mm lift /1.7s continuous valid hold; all621 samples safe,4 supported feet and current standing contract passed. The retained older short-task2mm/.5° diagnostic is false (maximum BASE displacement6.803mm); no threshold was changed. Exact first-result hashes remain in the actuals file.

First failure mechanisms and cold repairs are preserved. TRAIN07 r4 hit the original right-pad/cube contact gate. TRAIN13 r3 reached20,128/21,203 integrations at the original1,200s wall guard; no retry. The next preparation hit a recipe wall check at0 workers/physics (0.6348s). The following TRAIN13 r4 worker hit a separate campaign wall check before simulation or a model query; it also stays failed. The repair applies the already declared3,600s allowance at both entries, independent of tmpfs transport, and verifies real native0–3 /neural8–11 affinity. Only the never-started TRAIN25 r3/r4 suffix then ran, and both succeeded. Original failures, physical ticks, contact gates, score, WBC/legs/gains/cost/solver and2ms dynamics were unchanged.

Terminal pair CPU totals **11391.88s**; summed neural-worker wall **11670.35s** includes loading/waiting and is not measured GPU utilization. First successful pair's15 actual generations took6.51s; peak allocator7,475,250,688 bytes. Per-condition result hashes and CPU/wall remain in the actuals file.

## Fresh complete demonstrations and actual throughput

The three-world production cohort is **CLOSED0,24/24 sealed successes**:8 TRAIN layouts ×restarts5/6/7, one cube, PICK_HOLD. Every episode has21,703 native integrations,640 actually consumed commands,321 synchronized body/wrist frame pairs,61 full H40 windows and a real2s successful tail. Actual CPU slots6–7,12–13,14–15 are separate; cameras persist, the writer is bounded/asynchronous, and expert collection loads no SLA. Group wall **9399.03s**, including initialization and archive; total worker CPU **15221.32s**. This is **9.192 complete demonstrations/hour** under concurrent training/evaluation, not an isolated1/2/4-actor result.

| Production condition | Worker wall s | Worker total CPU s | Sealed shard SHA256 |
|---|---:|---:|---|
| train01-r5-p10 | 1095.61 | 569.11 | `6b1517e6a2988518b561e4ec139db2be1c96c0ef72ad7fdd9e2efd1090ca386c` |
| train04-r5-p10 | 1158.13 | 699.80 | `b5473315bcc80079b5edead5323a0e34e3d82f3d72e0ded94103f6721f7029a9` |
| train07-r5-p10 | 1122.92 | 695.40 | `9a6158e0e6dd88202bca1b971b12b8196a9b079dbc4fb74052acd414d33e3bd0` |
| train10-r5-p10 | 1222.23 | 589.04 | `b979e32d5a108b0ac3a120b445828fa42735762b6a0f9eb850d02e91c4a38fb6` |
| train13-r5-p10 | 1283.69 | 715.52 | `5cfa20d4ad657d0e85e2e173d446648a22c56e338bc2ef50d15a85acd4dcb6b4` |
| train16-r5-p10 | 1259.83 | 719.84 | `74eb680f8a96030dfbf93875fd6cc11f21c52a1e144c1c3d3ecfebf91dd0ae28` |
| train21-r5-p10 | 1407.97 | 596.62 | `48a9bc903c03bc79e96fabc988447efcbd58b8923b562487e6e1d936c0962584` |
| train25-r5-p10 | 1470.06 | 719.40 | `b7be045fc2618f23166249cc2896b290f01f912eee9236fc70447ea662ee903d` |
| train01-r6-p10 | 1450.21 | 717.01 | `132e872e9e2e1ba8bb2c320ad3dcca35fe456d97270486c1e31c41a3752df38f` |
| train04-r6-p10 | 718.54 | 465.79 | `a6d3e1d6860b9694b21a60b2eda93919bc2f04bc24df112998c18d0bcd8f63d7` |
| train07-r6-p10 | 778.73 | 597.72 | `d47a02983c455e22d555690ab9ed023e4c85bb98b683b6450d236201b6b4a0cd` |
| train10-r6-p10 | 715.45 | 595.58 | `5a76bc83278fd02a55342c4be1e4b386bef0742f010894ae010458cf6ee6a837` |
| train13-r6-p10 | 516.57 | 456.91 | `c87a4a3a80d2cdd71ec6ca9e539707568113a4516955da161ea80b58f6476c29` |
| train16-r6-p10 | 622.53 | 592.81 | `d1e63f09254b630bac59ab46f4359da9a879c3041cfda73ec001621c29fc9f63` |
| train21-r6-p10 | 622.75 | 594.39 | `d71c07a6919eb067b5bd2107937829af655191b1004fab43b1dc739012fc45a3` |
| train25-r6-p10 | 975.44 | 554.83 | `588ab8f8fd703785680b68018500bacae720e4e62370e34bfdf5182ad7a8eb38` |
| train01-r7-p10 | 1077.71 | 693.88 | `a5801025be80430776d502602e80650876646b480fe3705a0c6beb859357bab2` |
| train04-r7-p10 | 1066.14 | 685.91 | `26514727d07f339b25484a141b02bd68eadccacc860a49194724f1563e8cb0de` |
| train07-r7-p10 | 1069.11 | 576.39 | `e39e77a290f565d586d5047536927fb7c797f2453e00c251402113308d4b8a27` |
| train10-r7-p10 | 1154.10 | 710.60 | `4ee5668525127ebcbbcfbe6b07895c28d04e8cc4c099e929aaa65861211483f5` |
| train13-r7-p10 | 1156.77 | 703.85 | `e3675596b1f0ff8021be728d8a531cda615d85ae63fc4f69e66ce5421210f130` |
| train16-r7-p10 | 837.75 | 565.65 | `83b16bfcb921e7ba9ba160d6b088519bd55ef1befc817a5ba09d39f11148aed1` |
| train21-r7-p10 | 927.63 | 703.16 | `572019e0263b3210e6afc213a50afaa338521f7cf3f3facc3472228555552550` |
| train25-r7-p10 | 931.11 | 702.10 | `4a136b3fc76605733968d0ae40bf2f1d3f16221784ecbdfb34fbc934556204e0` |

Across new production there are **25 complete demonstrations /542,575 terminal integrations /16,000 consumed commands /8,025 dual-camera frame pairs /1,525 complete H40 windows**. All-attempt accounting includes27 native starts:25 successes and2 zero-tick technical failures;0 live production workers. Two earlier prephysics preparation failures remain separate. The original partial48-request actor experiment is CLOSED1 with46 unstarted; its one success and paused-startup failure are never relabelled as a completed benchmark. Original GLFW zero-tick and failed worker costs remain recorded. Old engineering16 and original2EP are excluded. Formal D100_new/D300_new and PICK_PLACE remain0; one-cube PICK_HOLD coverage is insufficient for those claims.

The new fixed1/2/4-worker benchmark uses the same8 layouts ×2 fresh restarts8/9 at each count (48 requests), P10, unchanged Teacher and recording,600s paused readiness and3,600s total allowance. It preserves both GPU jobs and records concurrent load; setup/initialization/archive and all attempts enter each rate. It began only after actual production CLOSED0. Its first preparation mistakenly used an old compiler receipt instead of the preparation input: CLOSED1,0 physical requests/models/updates,120.54s retained. A cold corrected run verifies the actual preparation specification and original model bytes. All8 fresh static scene checks passed with0 integrations. Actual1-worker execution uses CPU4–5; at this stable snapshot12 workers are registered,11 are terminal,11 complete demonstrations are sealed,0 closed workers await coordinator sealing and1 are live. A closed but unsealed worker is neither live nor a sealed demonstration. Across production and the benchmark this makes36 fresh complete demonstrations; benchmark demos do not enter the immutable N20 snapshot. Matched throughput is still pending; no8/16-worker expansion has started.

## First fit on the fixed fresh N20 snapshot

Dataset **T_PICK_HOLD_TRAIN8_P10_N20_V5**, manifest SHA256 `ec1e69592a7825ff2e7cc008ff46d1478cd8fe9a0451c17b6a89834878332cc1`, is the first20 declared production assignments; no success-based substitution. The five remaining production demos and11 actor-benchmark demos are outside this immutable training snapshot. Admission verified each actual CLOSED0 source, command consumption, RGB/clock/frame custody, five-stage coverage and real2s tail. Admission cost31.377s; independent all-window readback12.568s:20 episodes,1,220 supervised H40 windows,6,420 frame pairs and12,800 consumed commands. Unique window stages: PREGRASP47, DESCEND384, CLOSE80, LIFT135, HOLD574. Phase/source/privileged truth are not student inputs; normalizer uses TRAIN only. No old2EP or engineering16 episode is relabelled.

GPU1 really loaded clean SLA_base and committed **818 Adam updates /26,176 T visits /0 S visits**, B32=micro8×4/highest. The formal environment path uses `sla-pt`. Actual peak allocator **10,985,271,040 bytes**; latest update6.510s, host wait1.070s. Initial observed updates were about7–10s each; host preparation/wait is material. Parameters, Adam and gradients stay on device; logical input-payload, metric D2H and checkpoint transfer counts are separate and their active-trainer final totals are pending. No hotpatch, batch reduction or replacement occurred. The endpoint remains fixed3K/96K T/0 S with full87-array Adam/RNG states, saves500/1500/3000 and fresh ordinary-RGB16 reconstruction; step500 is now completely saved and read back;3K and fresh-process endpoint reconstruction remain pending.


**New-data complete step500 /16,000 visits:** SHA256 `5e70c7885848f521693e107d9923f10c9d7191848502e65956d974f817ea14e3`. An independent CPU readback verified all87 saved arrays /625,851,664 file bytes, their hashes/shapes/dtypes/finite values,86 arrays changed from initialization, and unchanged initial bindings/frozen-fingerprint binding. Audit0.5092 wall/0.4000 CPU seconds on2–3,0 extra parent loads/updates/visits/physics. This is saved-state verification; the active trainer's terminal frozen-parameter recheck is still pending. Sixteen ordinary-RGB generations exist at500, manifest `efba9951213bdf9c854cdce34b7e0e549b03d5ae48cc510b845f204fbe943438`. **No fresh-process500 reconstruction, complete3K or new-data native success is claimed.** Checkpoint file I/O/readback bytes are distinct from measured bus traffic; active-trainer H2D/metric/checkpoint final counters remain pending.

The separately typed N20 native bridge is implemented on `811d9efd0c7614c3ddf57ac6d6c8c37ed5a4166b`, with its TRAIN8 inputs committed before this model's outputs. It requires the actual20-episode/1,220-window audit, complete3K87-array state and fresh ordinary-RGB16 proof. Intermediate500 cannot enter. It reuses the existing raw/P10 consumer: full decoded H40 preview gate,60 queries ×10 executed atoms plus40 actual final-hold atoms. A real archived-scene custody mismatch was fixed by first checking the closed original producer and unchanged model bytes, then its original builder digest plus full static/robot/2ms provenance. All4 actual TRAIN scenes passed this metadata readback with0 model/physics. Other callers keep the original current-file identity check. **N20 native requests remain0; TRAIN8 fitting does not replace independent DEV48.**

## Matched spatial arm and complete saved evidence

The reused122-window spatial seam is CLOSED0:2 alpha0 updates /64 T visits,2 alpha.5 updates /64 T visits and2 independent FM-reference Adam operations /64 visits; **6 actual optimizer operations /192 total target exposures**,0 S/physics. All64 common microgradient pairs and both Adam/parameter pairs match exactly; positive spatial gradients reach all5 deployed VLM LoRA-B leaves. Training loads no teacher. Four complete states /420 arrays passed saved-file hash/shape/dtype/finite readback without extra updates. Qualification wall1,780.75s is owner wall; feature/mask H2D383,872,512 bytes per arm (767,745,024 total), hot feature D2H0; peak allocator10,713,120,000 /9,616,449,536 bytes. Exact state hashes remain in the actuals.

The distinct GPU0 SF candidate uses source `a00497eefd343f4f393b6b144ad2a7598b873dca`, clean base, original2EP122 windows, alpha.5, B32=micro8×4/highest, fixed3K/96K T/0 S. All87 common initial arrays match clean FM, initial complete SHA `de63b72c3df92021d4c89c83860390bb42a5239fdd8db7e4c5bbedbcedbf9585`. It is now at1167 actual updates. **Step500 is completely saved**, SHA256 `2aba39fb3aefbda36129eb2e8227cd64bfab0b4fef21e2d62d2d40d87de87b28`:105 arrays /726,615,568 array-file bytes independently read back,104 changed, unchanged frozen fingerprint/bindings. Audit0.6144 wall/.5126 CPU seconds,0 loads/added updates/physics. Sixteen ordinary-RGB generations are saved there, but fresh-process500 reconstruction and3K completion are not claimed.

The existing source `47add1915449277b961e9028a473b52fbbdf13d8` cold follower waits without a GPU owner. Only actual original CLOSED75, terminated process and complete105-array prefix allow full Adam/RNG/projector/cursor restore at the same absolute schedule to **3000 cumulative**, never another fresh3K. Actual restore remains pending. Healthy source/parameters/affinity are not changed in place.


## Independent new DEV48 and actual finite endpoint followup

New inputs are frozen in [raw3k_pick_hold_dev48_inputs_v5.json](../../../experiments/sla_go2_posttrain/raw3k_pick_hold_dev48_inputs_v5.json):8 new layouts ×6 restarts,60 fixed query seeds each. These are local one-cube PICK_HOLD development conditions, separate from old DEV/TEST, the later two-task/five-object evaluation and C24. They were fixed before any raw3K DEV output. A finite Core CPU0–1 preparation is **CLOSED0** after112.14s. Every layout passed5,703 original saved-startup FK/contact checks and unchanged robot/2ms contract; actual compiled initial geometries match declaration and are disjoint from the supplied sealed old TRAIN/DEV/TEST catalogs, including2 distinct old TEST initial model inputs. No TEST scores/images/actions/trajectories, model load, optimization or physical integration occurred. Disjointness proof SHA `364f66921d2fa8fba11c785b7fe98c75a745313af94e58125ff4784e7d7361b2`. A second metadata readback verified all8 closed-producer/model byte identities. This is static preparation, not a passed student DEV condition.

The existing native executor now carries the new split through its spec, validates the full48 fixed rows and actual scene proof, and permits at most8 contiguous conditions per finite group. Successive groups require actual CLOSED0, vanished coordinator, terminated native/neural partners and unchanged prefix outcomes/policy/seeds; valid failures remain in the prefix. Existing registered owners, Journal, original safety, full-H40 preview and P10 consumption are reused. Read-only capacity polling is also available for this raw3K role. A predispatch bug was fixed before new endpoint physics: capacity previously read the old base's GPU metadata; it now uses the candidate's actual reconstructed card/allocator/owner.

The concrete finite N20 followup on source `75e703f5c412ddc725e9145e12c27b02c3390e6e` is **verified live**, polling60s without a GPU owner/model or new optimizer updates. Deadline **2026-11-14T20:39:25+08:00**. It waits for the original N20 candidate's actual CLOSED0 and process exit, full3K/96K and fresh ordinary-RGB16; intermediate500 cannot trigger evaluation. It then reads back all87 endpoint arrays, seals the typed admission and dispatches fixed TRAIN8, followed by the separate DEV48 in6 contiguous groups. If all8 TRAIN attempts are rejected at the original entry before any post-start integration, it retains them and leaves all48 DEV unstarted for interface diagnosis. Technical/safety stops halt new dispatch; no worker, query or failed condition is automatically retried. Model/physics admission still occurs through the original runtime. Current new-data student physical requests remain0; the queue is not task success.

## Reusable book source seeds — command seam checked, not yet SLA-admitted

Actual source storage contains21 previously qualified complete BOOK episodes plus6 expert suffixes. This turn rechecked the immutable full-episode numeric files and original independent task/safety/release/withdraw audits:13,440 consumed commands,13,461 real observations and12,621 original full H40 anchors; a future2Hz view would have1,281 anchors. Source requests are full mechanical fractions with44mm travel. The original source Teacher already requests2mm closed /42mm open; conversion to native nominal g=(source_g−2/44)/(42/44−2/44) reproduces each desired physical width exactly, maximum error **0m**, without altering measured state grip or any old labels. Cost0.0577 wall/0.0249 CPU seconds,0 new source episodes/model loads/updates/physics. Source manifest `1dcb555894a3400b0ec1a032c60a039e2b40ac38705f339ff2d327b10a1c3c9d`. This reuses command data and does not repeat official GR00T/Processor qualification.

Mount-frame/EEF calibration and lossless RGB shards remain necessary before SLA book-data admission/500-step first fit. These are fixed-arm source-port demonstrations, **not floating Go2/WBC target qualification**, and add0 public RoboTwin S_cmd episodes. Six correction suffixes are not full skill seeds. No book training is yet queued; this evidence makes coordinate/image conversion the next skill step rather than recollecting existing full seeds or waiting for grab/placeD300.

## CPU input throughput, future cold runs only

The original N20 reader replayed the same first128 actual sample indices twice at512MiB and3GiB frame-cache limits on CPU2–3. All512 observations, including both images/state/actions/text, are byte-identical. Cold/repeated preparation is20.5098/9.8186s at512MiB and15.3904/0.1293s at3GiB;44 shards/2,066,419,216 bytes remain cached in the latter. Total46.6804 wall/45.7393 CPU seconds,peak process RSS2,160,488KiB,0 GPU owners/model loads/updates/physics. The first attempt stopped before examples at a conservative40GiB headroom check; the bounded read-only3GiB replay then used24GiB headroom. Kernel page-cache state and concurrent learners were uncontrolled: this measures host input preparation, **not end-to-end training speedup**.

Optional bounded cache configuration, memory headroom and immutable continuation-profile checks are implemented for future cold runs. Default512MiB and existing active profiles remain unchanged. This code is not deployed to the live learners or their cold follower. Existing prefix restores must use their original source; a new source cannot silently replace an adapter identity. No N20 repeat training is launched for this host benchmark.

## Fallback, retained negatives and next evidence

H200 job716146 remains unknown after existing access rejection; no cancel/restart/duplicate or bypass route. Last dated2026-11-13 23:35 SGT:513 updates /16,416 T visits and complete500 SHA `ba52032ec238ad2cb53f9bef7b4af3a840f862fcbee477e597bffefeb19e5d45`. It is not a current3K success. A40 isolated `sla-pt` preparation is CLOSED0,201 parent files /12,471,050,953 verified bytes, plus139 frozen spatial-input files /294,236,160 bytes with0 repeated teacher forwards. Latest scheduler read **2026-11-14T06:04:21+08:00** still returns PBS connection refused:0 allocations/submissions/model loads/updates; no training on login.

Original S1 stays500/16,000, TRAIN4 0/4 and old DEV15/16 terminal with0 successes and one unstarted condition. Original S2 candidate stays0 updates, separate from the new SF candidate. Five default RoboTwin archives total250 episodes;20 were already audited. Clock/desired-command/TCP/current-gripper gaps keep S_cmd0. No repeat of the camera/SFT2000/GR00T qualification or old audit is claimed.

Next evidence: continue N20 past its verified500 save toward complete3K/fresh ordinary RGB; continue both healthy learners through finite endpoints and any permitted same-budget cold restore; settle fixed1/2/4-worker complete-demo rates; admit complete3K/fresh-RGB artifacts before raw native TRAIN8. Independent DEV48/C24, PICK_PLACE/five-object coverage, formalD100/D300, corrections, command-source generation, book/door and shared/moving skills remain unfinished. CPU regression checks validate code and accounting, not training/task success. WBC/legs/gains/cost/solver,2ms/native safety, historical failures and old DEV/TEST stay frozen.

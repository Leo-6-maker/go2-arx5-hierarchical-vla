# V5 execution — 2026-11-14

Fixed plan `f61305f9c1f847e460851fd067892160f71ca432`; execution continues from the reviewed runtime, in an independent implementation worktree. Dated Core snapshot: **2026-11-14T04:01:39+08:00**. This is an incomplete V5 result. [Machine-readable actuals](ACTUAL_20261114.json) include per-condition hashes and costs. Raw data, model arrays, site paths and process IDs remain private.

**Actual progress:** Core SF has547 committed updates /17,504 target visits and a fully saved, independently read-back step500. There are16 complete fresh expert demonstrations across8 layouts. R1 retains3 task successes,1 valid policy failure and2 technical failures out of8 requested conditions; only its never-started suffix may continue. These are TRAIN results, not independent DEV/TEST generalization.

## Spatial updates, complete states and RGB

The Core GPU0 spatial seam is CLOSED0. Alpha0 and alpha0.5 each have2 B32 arm updates /64 target visits. The independent FM reference adds2 real Adam updates /64 visits: **6 optimizer operations /192 total target exposures (128 arm +64 reference),0 source visits,0 physical requests**. All64 common microgradient pairs and both parameter/Adam update pairs are bit-exact. Positive spatial gradients reach all5 deployed VLM LoRA-B leaves; teacher weights are not loaded into training. The initial105-array states match exactly. Four complete states /420 arrays passed independent post-save hash/shape/dtype/finite readback without new updates.

| Qualification arm | Step0 complete SHA256 | Step2 complete SHA256 |
|---|---|---|
| alpha0 | `040b4971f17217e6db835dfe1d1ff24ba29a9f3696252ccfeedad7cc012fd4fa` | `febe48e10ceadfd744d1620e0d840e369862013f9c049db845c0b0cecb0819d4` |
| alpha0.5 | `6c78cd3d538d56e87e4e1faaddb579ef732a2653d8eee020e1cc45075e48cacd` | `f62c274bf6baf5ee74233f27fdfffa2e77ad1463ed2bb0cb66f1df58f4d31301` |

Qualification wall1,780.75s is allocated owner wall, not GPU utilization. Peak allocator10,713,120,000 /9,616,449,536 bytes. Feature/mask staging was383,872,512 bytes per arm,767,745,024 total, once per bank; feature hot-path D2H0. Parameters, Adam, gradients and projector remained GPU resident. Checkpoint/audit transfers are separate costs.

The distinct clean-base matched **SF3K** candidate uses source `a00497eefd343f4f393b6b144ad2a7598b873dca`, original122-window2EP data, B32=micro8×4, highest precision and alpha0.5. It targets3000 total updates /96,000 T visits /0 S visits; it does not warm-start from the FM endpoint or the two-update seam. All87 common initial arrays match clean FM; common-initial complete SHA `de63b72c3df92021d4c89c83860390bb42a5239fdd8db7e4c5bbedbcedbf9585`. Actual micro8 peak allocator12,018,348,288 bytes. The healthy trainer is unchanged.

**Step500 /16,000 visits is actually saved**, SHA256 `2aba39fb3aefbda36129eb2e8227cd64bfab0b4fef21e2d62d2d40d87de87b28`. Independent CPU readback verified all105 files /726,615,568 saved array-file bytes;104 arrays differ from initialization, every shape/dtype/hash/finite check passes and all initial bindings remain unchanged. Frozen fingerprint remains `507aa0a700c2e8cacfb398c76e2147da4a82cbf05bc2e7b4ec85d1562c091b06`. Audit cost0.6144 wall seconds /0.5126 CPU seconds,0 parent loads,0 added updates/visits/physics. Sixteen ordinary-RGB generations are saved at step500; **fresh-process step500 reconstruction is not claimed**. This is a complete intermediate state, not completed3K or dense native success.

The real cold follower on source `47add1915449277b961e9028a473b52fbbdf13d8` remains waiting without a GPU owner or new updates/model loads. Only actual predecessor CLOSED75, terminated process and complete105-array prefix admit one restore with Adam/RNG/projector, unchanged absolute schedule/visit cursor and **3000 cumulative updates**. CLOSED0 complete3K skips further training; other failures are preserved. Partial discarded visits and repeated cold-load/transfer costs remain visible. Actual cold restore is still pending.

## Fixed native endpoint and failures

The unchanged fixed population is R0 then R1:4 TRAIN layouts ×2 restarts ×2 methods =16 requested conditions. Both original models retain500 updates /16,000 visits. Evaluation adds0 optimizer updates and0 expert demonstrations. It is separate from old S1 TRAIN4/DEV16.

| Fixed condition | Terminal integrations | Raw→installed→fully consumed H40 | Actual outcome |
|---|---:|---|---|
| R0 TRAIN01 r3 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R0 TRAIN01 r4 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R0 TRAIN07 r3 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R0 TRAIN07 r4 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R0 TRAIN13 r3 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R0 TRAIN13 r4 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R0 TRAIN25 r3 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R0 TRAIN25 r4 | 5,703 | 1→0→0 | Valid original nominal-gripper rejection |
| R1 TRAIN01 r3 | 21,203 | 15→15→15 | NATIVE_STAGE_PASSED; task success |
| R1 TRAIN01 r4 | 21,203 | 15→15→15 | NATIVE_STAGE_PASSED; task success |
| R1 TRAIN07 r3 | 21,203 | 15→15→15 | NATIVE_STAGE_PASSED; task success |
| R1 TRAIN07 r4 | 14,228 | 9→9→8 + partial21 | Valid original right-pad/cube contact rejection |
| R1 TRAIN13 r3 | 20,128 | 15→15→14 + partial16 | Technical:1,200s wall endpoint at20,128 integrations |
| R1 TRAIN13 r4 | 0 | 0→0→0 | Technical:campaign wall gate before simulator/model query |
| R1 train25-r3 | Live snapshot 1325 | Not terminal | Fixed unstarted suffix now active |
| Remaining unstarted | 0 | 0→0→0 | 1 conditions, never replayed |

There are **14 terminal condition attempts**, including one worker-only0-integration condition; **13 terminal native worlds actually started**, plus1 currently active condition(s). Fixed R1 success is **3/6 terminal,3/8 requested**, with1 valid contact failure and2 technical failures. R0 remains0/8 successes with8 valid original nominal-gripper failures. Terminal native integrations total143,589. Terminal delivery is **77 raw queries →69 endpoint-compiled installed H40s →67 fully consumed H40s**, with2,680 command confirmations from complete prefixes. Two incomplete cycles record21 and16 partial atoms; they are not called complete H40s or silently discarded. Initialization cycle minus1 is excluded. Live-query counts are not added to these terminal totals. No clipping, teacher fallback, valid-failure retry, original hard safety event or old DEV/TEST rerun occurred.

The first R1 success reached20.618mm actual lift and1.7s continuous valid grasp/lift/hold. All621 scored samples were safe,4 feet supported, current grasp standing bounds passed and native trace checks/clean shutdown passed. The retained older short-task pose diagnostic is **false** (maximum BASE displacement6.803mm). No threshold changed. Native result `d0eaf24767623d814508341832755ee609c1feaa0ea6b716dd1cb12fde714cdd`, neural result `1de9355a53dfaa56f9cbdeb03328a3151e384a139bbbdc0511b324af5a77eff8`. First-pair native/neural wall945.95/885.24s overlap; total pair CPU1,147.86s. Its15 actual generations took6.51s, peak allocator7,475,250,688 bytes. Across terminal conditions pair CPU is8910.87s and summed neural-worker wall9692.87s; waiting/loading wall is not measured GPU utilization. Exact per-condition costs/hashes remain in the actuals file. R1 uses the registered endpoint basis; this is composed-policy success, not direct raw-curve execution.

TRAIN07 restart4 hit the original right-pad/cube contact gate and stays a valid policy failure. TRAIN13 restart3 reached20,128/21,203 integrations at the original1200s wall allowance; its technical failure is retained. A cold three-row successor's first preparation was refused before any worker/model/physics by the native recipe wall gate (0.6348s group wall). The next version passed that gate but its TRAIN13 restart4 worker was refused by a second campaign wall gate:0 integrations,0 SLA queries, native world not started. The paired GPU lease error is the downstream symptom; the actual native worker records `AUTO1_REGISTERED_SCOPE_HOST_LEASE_OR_TIME_INVALID`. That condition remains the second technical failure. Neither failed TRAIN13 condition is retried.

The repair applies the declared3600s allowance at both original entry layers, independently of the separate tmpfs transport. Exact physical ticks,2ms dynamics, gains/cost/solver, WBC/legs, original contact/safety and scores are unchanged. Native0–3 /GPU8–11 affinity is actually verified. Existing closed RGB qualifications are reused. The cold successor retains all14 terminal rows and permits **only TRAIN25 restarts3/4**. Read-only capacity polling is60s before any pair effect; a busy healthy GPU owner is preserved. Native recipe, original predecessor feedback and actual campaign-scope metadata preflight passed before dispatch; registered pair checks still run normally. A registration/dispatch is not a new native success.

## Fresh expert production and first-fit queue

The original48-request1/2/4-actor experiment is CLOSED1:2 starts,1 sealed success,1 paused cold-CppAD startup failure at0 integrations and46 unstarted. The success has21,703 integrations /640 consumed commands /321 dual-camera frames /61 complete H40s, shard SHA `6a2894ec1e76db8578b88c4699ece9155ea3d66c2a1476cb786135102d270209`; wall2,544.19s /CPU734.23s. Its second worker used286.79 wall /7.93 CPU seconds; group wall3,003.36s. The earlier separate GLFW0-tick world used160.37 wall /3.57 CPU seconds. All remain in all-attempt costs. Failed receipt-as-spec setup and the parallel successor's rejected5400s wall request had0 physics. No complete1/2/4 throughput comparison or8/16 expansion is claimed.

The distinct production population is24 requests:8 TRAIN layouts ×fresh restarts5/6/7, three independent worlds on source `25ba570b52673de66c9b7df13b64b0a46fef5af8`. At the snapshot it has **15 sealed successes,3 live,6 unstarted**. Actual CPU slots6–7,12–13,14–15 and distinct ROS/world owners are verified. Each complete episode has21,703 integrations /640 commands /321 dual-camera frames /61 full H40 windows. Master polling60s; no blind worker retry. Cold startup uses the unchanged3600s total and600s paused readiness, requiring original markers and feedback. Frozen `recompileLibraries true`, solver and physics are unchanged.

| Newly sealed production condition | Worker wall s | Total CPU s | Shard manifest SHA256 |
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

Worker walls overlap. Collection under concurrent learning/evaluation is not an isolated actor-count benchmark; all-attempt setup/archive throughput is due at finite-group closeout. Across new operations there are **16 fresh complete demonstrations /347,248 terminal integrations /10,240 consumed commands /5,136 dual-camera frames /976 H40 windows**, covering8 layouts, one cube and PICK_HOLD only. All-attempt accounting is21 native starts,18 terminal (16 success +2 zero-tick technical failures),3 live, plus2 prephysics preparation failures. Old engineering16 and original2EP are excluded. D100_new/D300_new and PICK_PLACE admission remain0.

The separately typed **T_PICK_HOLD_TRAIN8_P10_N20_V5** reader, immutable snapshot admission and existing dense training-loop routing are implemented on source `cdc6ec75400ec1d2b508b4349166cda4a395da4c`. Selection is the first20 declared production assignments, without success-based substitution. Every selected episode must already have its own CLOSED0 worker, source command/RGB/clock custody, complete five-stage coverage and actual two-second tail. A named TRAIN-only normalizer is fitted before oversampling; the student receives only legal RGB/state/language. The strict old two-episode reader is unchanged in scope.

Its **actual finite background process is live**, with15 of20 selected episodes sealed. One real source episode passed the new61-window derivation/roundtrip path (5.3121 wall /0.7783 CPU seconds); this inspection admits no training dataset. The queue holds no GPU owner and has0 model loads/updates/physical requests while waiting. After all20 are sealed and read back, it uses the original GPU1 owner and actual capacity at a cold boundary. Target recipe remains clean SLA_base,3K/B32=micro8×4/highest,96K T /0 S visits, full Adam/RNG state, finite save/exit and ordinary-RGB reconstruction. Input device payload bytes, metrics D2H, checkpoints and owner/queue wall are separate costs. CPU preparation uses4–5; learner8–11. The pending first fit is not training success or formal D100.

## Dense bridge, fallback and retained evidence

The separate dense3K raw/P10 role is implemented with fixed TRAIN8 inputs: full decoded H40 passes the whole-preview gate;60 real queries execute10 atoms each, followed by40 actual hold atoms at endpoint21,703. Raw curve interiors, single sigmoid, actual body/wrist RGB, original ACK/solver/gripper consumption and safety are preserved. Cancelled suffixes are not called consumed. Admission requires actual complete3K,87 FM or105 SF arrays, exact fresh ordinary-RGB16 reconstruction and its actual card/owner. **Dense native requests remain0**; intermediate500 and Core4 cannot enter.

H200 job716146 was not cancelled/restarted/duplicated. Existing direct reads closed and one authenticated Vanda jump explicitly rejected HPC access. Current state remains unknown; last dated2026-11-13 23:35 SGT is513 updates /16,416 visits, complete500 SHA `ba52032ec238ad2cb53f9bef7b4af3a840f862fcbee477e597bffefeb19e5d45`. No3K completion is claimed.

A40's isolated **sla-pt** preparation is CLOSED0:201-file12,471,050,953-byte SLA_base verified,49,649,542 source/upstream/dense-input bytes staged, plus139 frozen spatial-input files /294,236,160 bytes with0 repeated teacher forwards. Preparation wall1,638.12s. Site metadata relocation still precedes any GPU run. Latest02:58 SGT Vanda PBS read again returned connection refused; A40 allocations/submissions/model loads/updates remain0 and no training runs on login. H200's unknown job is never duplicated.

Original S1 remains500/16,000, TRAIN4 0/4, DEV15/16 terminal with0 successes (last condition unstarted). Original S2 candidate updates remain0. Five default RoboTwin archives have250 episodes,20 previously audited; clock/desired-command/TCP/current-gripper gaps keep S_cmd=0. No repeated20-episode audit or camera/SFT2000/GR00T qualification is claimed.

Next actual evidence: finish the remaining R1 conditions without rerunning failures; seal/audit the fixed N20 cohort and obtain its first real gradient/save; continue the healthy SF candidate through its finite window and same-budget cold restore; receive complete3K/fresh-RGB artifacts before dense native TRAIN8. Independent DEV/C24, PICK_PLACE, formal D100/D300, correction data and source command admission remain unfinished. WBC/legs/gains/costs/solver/2ms safety and historical negatives stay frozen. Relevant N20 and native-entry CPU tests passed; they validate software/accounting, not training or task success.

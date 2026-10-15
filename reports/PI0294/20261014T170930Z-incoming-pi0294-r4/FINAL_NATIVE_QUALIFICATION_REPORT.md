# PI0294 R4A — actual native hold verdict and dependency stop

Status: **PARTIAL — VALID_NATIVE_HOLD_QUALITY_FAIL**. Updated 2026-10-15, Singapore time.

The full Go2 + ARX5 is the adopted primary domain; fixed-base ARX5 remains a baseline. Human adoption and original R4 clock/budget are unchanged. Scope release: `0ee6211fbdfd8b7cb6449e5a4d814a68ca29c0b9`; original new-work cutoff 12:09:30 SGT and closeout 13:09:30 SGT. This report closes an actual scientific dependency failure before those cutoffs; unused resources do not authorize quality retries.

## Actual control result

After the source-bound receiver fix, the native free-root robot completed **10.014 s of constant-world EEF hold** after a five-simulation-second native warmup. It used the original native plant, twelve actuated legs, six arm joints, frozen leg policy, original MPC costs/gains/limits and original strict safety observer. It did **not** yet include the proposed active-contact gripper/task/camera overlay. This is a native control qualification, not a manipulation episode or a full equipped-executor PASS.

| Measurement | Recorded result | Original gate | Verdict |
|---|---:|---:|---|
| Actual MuJoCo TCP position RMS | 0.012535354 m | <=0.015 m | PASS |
| Actual MuJoCo TCP orientation RMS | 0.069650907 rad | <=0.05 rad | **FAIL** |
| Native measured-FK position RMS | 0.012525667 m | <=0.015 m | PASS |
| Native measured-FK orientation RMS | 0.069599038 rad | <=0.05 rad | **FAIL** |
| Source-clock residual | 0.020838534 s | <=0.025 s | PASS |
| Accepted / rejected world-reference packets | 33 / 0 | correct installation | PASS |
| Installed constant reference error | 0 m / 0 rad | unchanged requested geometry | PASS |
| Original native safety stops / illegal contacts / native warnings | 0 / 0 / 0 | zero | PASS |

All installed references retain the original observation source anchor. Physics and native FK independently show the orientation failure. No gain, cost, joint state, threshold, seed or target correction was used.

## What changed and what was checked

1. Reused the delivered private package and prior legal home + MPC-arm startup. An isolated incremental clock overlay was compiled and linked; existing transitive libraries were resolved without installation or changing protected binaries.
2. The actual native parser CPU reproduction proved that zero-lead world binding replaced action[0] with the later measured pose. The new receiver preserves the requested first world point, EEF and optional BASE endpoints, and source time. Actual parser CPU checks cover source epoch, stale/future packets, sequence, cancel, world frames and 41 native endpoints. These CPU cases are not runtime reset/TTL qualification.
3. Two launch pre-dispatch errors (DDS domain range; existing bundled Xvfb path) were retained with zero physics. The first physical attempt stopped at 5.468 s when cross-topic delivery put the measured EE packet 6 ms ahead of the manager's MPC observation. The reconstructed rejected wire has the identical original SHA256. A bounded wait for the matching observation fixed delivery ordering without changing the source anchor or accepting a future timestamp. The replacement then accepted all33 packets.
4. The wrapper originally labeled its final pause check `PAUSE_CLOCK_ADVANCED`. Saved original records show physical time stayed exactly **15.186 s** before and after pause; queued EE feedback changed 15.174 to15.180 s, both before that frozen physical time. The original RESULT/error is preserved. This is a read-only adjudication of delayed feedback, not a rerun or a claim of exact whole-history resume. Full MPC/leg-history resume and runtime lifecycle negatives remain NOT_QUALIFIED.

Entry deviation: the native physical wrapper sourced the original native ROS environment and validated the sealed R4 host/UID/source/authority/lease/output binding, but did not invoke the deployed `infrastructure/enter.sh` profile wrapper. This is recorded in `ENTRY_DEVIATION.json`; no retrospective entry/environment PASS is claimed. Original core/build/config identities and the negative physical observation are retained. It does not permit repeating the valid quality failure.

Build source: `29a670812b37f62e0d20708e085bbfb7c2f539c4`; actual final physical receiver/harness: `9ecd1fe75cb727a9d45364381511a696aacf5409`; independent CPU scorer: `ce04b7b3fe721de3247d08b90a69d5ccd87eae36`. Binary/core/array identities are in the accompanying receipts. New code has an actual-delivered-parser CPU check and a numerical scorer sanity check; no broad test suite or extra model query was run.

## Saved-array mechanism evidence and limits

The last two seconds still have **0.070513483 rad** orientation RMS. Mean rotation bias has norm **0.069309699 rad**, with jitter around that mean **0.006885810 rad**. Thus the failure is persistent, rather than only a startup transient.

Joint3 and joint4 requested-minus-actual mean errors are **0.035365544 / 0.035124304 rad**. The installed requested orientation is unchanged; physics and native measured FK agree on the achieved error. These observations support a downstream tracking-bias hypothesis, but **do not prove a unique controller, gravity-compensation or mechanical cause**. All18 recorded joints retain positive margins within the scored task interval.

BASE preference was omitted, not a zero-velocity soft hold. The frozen WBC chose nonzero body commands and the free root actually moved; that movement is not attributed to a learned BASE head. The policy did not issue any high-level velocity command. No translation/yaw qualification or BASE-head benefit is claimed.

The useful next decision would concern a narrowly specified controller-tracking diagnosis using this same evidence. It cannot be silently implemented as a gain/cost change or a fresh success-seeking rollout under this run.

## Dependency stop and retained independent results

- No further native quality attempt, equipped-plant physics, fullbody Teacher/pilot, v4.0-mob13 collection, FB13 canary/training or fullbody model evaluation was launched after the valid hold FAIL.
- Fullbody pilot/data/FB13 are **NOT_RUN / NOT_PRODUCED / NOT_TRAINED**, not failed learned models. Runtime reset/resume, whole-stack shadow cloning and multiple fullbody workers remain unqualified.
- Prior fixed-base pilot: four actual successes retained. Formal collection: ten actual worker successes; nine independently admitted, one original contact-force reconstruction audit FAIL (1.182544996e-5 N exceeds1e-5 N) retained without rerun or tolerance change. Thirty-eight never-started cases remain NOT_RUN_SCOPE_REDIRECTED. These are not fullbody data.
- The original v3.1/original-statistics codec rejection across73322 windows remains unchanged; no refit, pruning or quality retry.
- R3 M2048 remains the original complete2048 checkpoint and verified public archive. The fixed16 model panel is **0 observed /16 planned, all NOT_RUN**. Job666401 first received a GPU at05:42:08 SGT, after the original03:16:40 new-work cutoff, and admission correctly rejected it before a model load/query. That is resource/window incompletion, not a learned-policy quality FAIL. A separate terminal reconciliation releases the old reservation and charges12 s conservatively from native allocation start/end (PBS reports3 s process walltime). No R3 training or evaluation was restarted.

## Evidence, resources and disposition

The actual native raw evidence and build material remain private. **92 files /102522271 bytes** were copied to private shared storage, individually SHA256-read back; tree `65d995cd2b549ba4ab1c0b5594731ed41dd83d1201c615c813803184b61604ac`. No raw/video/WBC public upload and no source deletion. Public delivery contains self-authored integration code and small receipts only.

Parent R4 totals at closure: **20 charged raw starts;234331 conservatively charged main integrations;8974 RGB pairs;Core CPU11955.912888 s;Hopper CPU0.67567 s**. R4 PBS/GPU/model queries/formal updates/engineering updates are all zero. Native child process CPU is recorded and additionally bounded by the inherited four-core wall interval to cover potentially unreaped descendants. Exact final integration count remains UNOBSERVED where only sampled traces survive; the conservative charge is not represented as exact. Quota headroom UNKNOWN.

No active native processes or reservations remain. Original failures, source versions and budgets are retained. **selected=NONE; deployment=NONE.** Execution completion, native quality and learned task success are separate; this is a real negative control result and a dependency stop, not overall R4A completion.

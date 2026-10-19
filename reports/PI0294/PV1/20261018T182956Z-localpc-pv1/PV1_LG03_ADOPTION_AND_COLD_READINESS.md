# PV1-LG03 adoption and cold readiness

Status: **FAST_SCIENCE_GOAL_ADOPTED; B0 incomplete; training remains blocked.**

The live state supersedes the snapshot in the LG03 handoff. Job `675967.hopper-m-02` is terminal and settled. B0 has **12/48** independently audited valid conditions: 8 successes and 4 valid task failures, evenly split as 6 valid conditions per task. No optimizer update has run. `selected/deployment` remain `NONE`.

## Engineering closure

`DEV/F1/004/T1/N294101` remains an engineering-incomplete attempt and is not counted as a task result. Its 22nd physical command was applied and ACKed, while the file-RPC server did not expose the already prefetched next request to `select(2)` until after the local 20-second watchdog. The server later emitted the exact expected 770-byte response; its reconstructed SHA-256 equals the abandoned stdout SHA-256. No replay occurred and no native safety fault was observed.

The repair replaces buffered `stdin.readline()` behind fd readiness polling with one raw-fd newline frame reader. It preserves prefetched frames, joins split frames until one monotonic deadline, and retains the original EOF, size and timeout contracts.

## A/B cold readiness

Both primary arms are **PREPARED_NOT_SUBMITTED**. They bind the same CORR@2048 parent and frozen statistics, separate frozen data manifests, the matched absolute-cursor sampler, identical fresh optimizers, 4096 updates, effective batch 32, and 12 fixed archived TRAIN probes at step 1024. Full checkpoint recovery files are present. Preparation loaded no model, issued no query, submitted no PBS job and performed no update.

Native training admission remains blocked solely at this layer by the formal B0 gate: 12 valid conditions are present and 48 are required. Task success rate is not an additional B0 admission gate.

## Provisional runway

The conservative scenario uses the prior observed maximum condition cost with 25% headroom, explicit cold-job overhead and accounting pads, and four training stages (`0→1024` and `1024→4096` for each arm).

| Scope | Total H200 including 4.085 h used | PBS total | Current reading |
|---|---:|---:|---|
| B0 remainder + both A/B endpoints + both DEV48 | 64.60 h | 14 | Fits the 96 h cap while preserving the final 20 h reserve |
| Above + LOCKBOX80 with no engineering replacements | 90.42 h | 18 | Fits the numeric cap, but not if the last observed 5.15 h queue delay repeats |
| Above + the registered 20-replacement engineering upper | 96.61 h | 18 | Exceeds the H200 cap by 0.61 h; not admissible now |

Queue delay is unknown and is not filled with zero. The latest 5.15-hour wait is an observation, not a statistical bound. Consequently, the matched A/B science loop is the priority; LOCKBOX admission must be recomputed from actual B0, training and DEV costs after a unique candidate qualifies.

Project `CFP04-CF-007` was read back as active with the current account as a member. Another user-owned H200 job was observed and was not modified. There is currently no PV1 H200 job.

## Verification and next action

The focused RPC/control tests pass. The local PI0294 suite passes 882 tests; the only local execution failure is the absent optional `torch` package, and the checkpoint-delivery collector is unavailable locally because `safetensors` is absent. GitHub CI remains the authoritative dependency-complete check.

The next legal action is: publish this repair, require green CI, then create at most one new cold B0 allocation only after fresh owner, lease, project, ledger, single-H200, evidence and resource gates pass. WBC, plant, gripper PD, S1, scoring and statistics remain frozen.

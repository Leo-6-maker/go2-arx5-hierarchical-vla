# PI0277 R3 — Latest-result adoption and one conditional generated-gripper bound repair

Date: 2026-10-07 SGT. Same authority: **PI0277_R3_SINGLE_TASK_FOCUS_AND_FAST_PHYSICAL_FEEDBACK_GO**.
This addendum is part of the R3 canonical release. It takes precedence over the pre-result wording in `PI0277_R3_STAGE_RESOLVED_FOCUS_AND_LOCAL_FEEDBACK.md` and `R3_REVIEW_AND_EXECUTION_LIMITS.json` ONLY for (a) latest known state and (b) the single explicitly bounded objective amendment below. It grants no extra lineage, updates, episodes or extended deadline.

## 1. Results arrived during publication; use them now

The concurrent owner commit included in `c056338b783b453e415893a339ea88db3b7efc49` contains `MILESTONE4096_RETURN.json`, observed2026-10-07 01:11:53 SGT, and `GRIP4096_ENVELOPE_AUDIT.json`.

Actual GRIP4096 TRAIN4:0 first/persistent stack,0 on-top,2 sustained lifts. Three valid gripper-envelope failures and one full640-control physical run. All145 saved outputs and native predicates were CPU-replayed; do not redo this completed screen or replay because the earlier R3 draft said PENDING.

| seed | actual outcome | commands executed | first sustained-lift diagnostic/event information |
|---|---|---:|---|
|10000|valid gripper envelope FAIL|32|no grasp/lift|
|10001|valid gripper envelope FAIL|336|grasp and lift; first2cm crossing9.21s|
|10002|full physical, no stack|640|grasp and lift; first2cm crossing22.28s|
|10003|valid gripper envelope FAIL|0|no executed motion|

All four recorded IK-hold counts are0. This does not mean their future unexecuted motion is feasible. The2048 IK observations remain valid history, but **IK is not the universal/latest sole explanation**.

Three refusal points came exclusively from the UNEXECUTED H40 suffix:
-10000 at1.6s: indices32..36; whole minimum-0.109375; planned first8 commands0.994140625..1.01171875.
-10001 at16.8s: index39; whole maximum1.1015625; planned first8 commands-0.01171875..0.01171875.
-10003 at0s: indices35..39; whole minimum-0.140625; planned first8 commands0.947265625..0.984375.
The actual fixed admission envelope remains[-0.1,1.1] over ALL40 commands. Those refused chunks were not installed. These are genuine policy-output failures, not license to ignore the suffix, clamp the full output, change the evaluator or relabel old results. Some values are only slightly beyond threshold, but changing thresholds after seeing them would change the experiment. Source10002 completed without refusal and still never stacked; removing refusals alone is not proof the task is solved.

The full task result satisfies R2's registered FOCUS trigger. The owner has already prepared and submitted **651332.hopper-m-02**, observed01:08:35 SGT; the01:11 result described it as Q waiting for GPU, canary not yet run. Adopt that job and its actual newer state, never qsub a duplicate. This publication is not an instruction to repeat GRIP4096, start another FOCUS, or cancel a healthy FOCUS. Record any concurrent later results before acting.

## 2. Immediate execution: existing FOCUS first, not another training design detour

Proceed with the already sealed16-episode FOCUS, existing GRIP objective, FAST_CACHE2, local recovery and R2/R3 probes. Do not change a queued/active source seal silently or rebuild the dataset. FOCUS is a targeted fitting attempt, not an assurance that suffix overshoot or task progression will disappear.

Alongside it, add a CPU-only saved-output summary per snapshot: counts of prefix-only/suffix-only/both/neither gripper-envelope violations; min/max and violating indices; OPEN/CLOSE/transition-label conditional errors; actual executed control count and task events before termination. All full-H40 failures remain in the score denominator.

## 3. One bounded GRIP_BOUND continuation is now authorized, only if the failure recurs

Trigger: the earliest fully recorded FOCUS TRAIN4 screen at512/1024/2048 reports at least2/4 valid whole-H40 gripper-envelope failures, each with no out-of-envelope value in its first8 planned commands. Use the fixed conditions and recorded outputs; do not add screening seeds or queries. If a qualifying candidate already exists, prioritize it rather than changing its weights. If FOCUS4096 is already complete, this rule grants NO further updates.

If triggered, the owner may apply **once** a generated-gripper box penalty for the REMAINING updates of the same FOCUS lineage, at a saved optimizer boundary. This is a declared R3 objective phase change named `FOCUS_GRIP_BOUND_R3`, not the original sampler-only causal experiment and not a new fresh-start lineage. Preserve weights, optimizer moments, scheduler/global step, RNG and remaining visit order. Do not reset the cosine schedule or get another4096 steps.

Let g[k] be the existing differentiably decoded final S4 command fraction, with the original valid channel9 mask m[k]. Define:

`L_box = sum_k m[k] * ((relu(-g[k])/0.05)^2 + (relu(g[k]-1)/0.05)^2) / sum_k m[k]`.

The new total is **`L_old + 0.1*L_gripper_generated + 0.1*L_box`**.

This uses a physical command interval[0,1], safely INSIDE the unchanged[-0.1,1.1] admission envelope. It directly trains against overshoot anywhere in H40 with uniform valid-point weights. Existing pose/native/generated-gripper terms and temporal reductions are unchanged. No threshold sweep, no sigmoid/new head, no output clipping/binarization, no extra S4 generation and no simulator differentiation. Evaluation remains exactly the same unclipped raw-output admission followed by the old declared actuator transport. The0.05 scale and0.1 coefficient are prospective fixed engineering choices, not a literature-optimal claim.

First verify CPU zero/in-range/out-of-range/mask/finite-difference cases, affine mapping and finite values. Reuse the existing S4 graph in FP32; add the new scalar once. On the first actual continuing optimizer update, record active violations, box contribution and finite gradient reach. An in-range sample legitimately has zero box gradient. This health observation counts as a normal formal update; no discarded new training sweep or extra model query is granted. Previously qualified loader/S4 and local-resume mechanics are reused, but all checkpoints and inference snapshots must record the new objective phase and exact source hashes. New loss code must be applied through a safe checkpoint/resume or an explicitly source-bound update-boundary mechanism, never through editing files under an active immutable worker.

If no trigger occurs, keep the original FOCUS objective; do not add the penalty just because it is available. If the triggered implementation is not ready safely, continue the existing valid FOCUS and disclose the amendment NOT_EXECUTED, rather than use unvalidated code or lose useful training. No new candidate screen points beyond the original512/1024/2048/4096, no third phase/loss search, no counter reset. Both old and post-change snapshots stay visible in the selection trail. There is no claim of isolated loss effect without a matched control.

## 4. Remaining task bottleneck and decision discipline

Track task progression separately from output range: secure grasp -> actual lift -> carry toward the target -> native on-top -> release -> persistent stack. Seed10002's late first lift despite a long grasp is evidence that task progression is still slow/ineffective in at least one full run; it is not proof of a particular internal phase mechanism.

The saved-prefix audit from the main R3 protocol remains useful: does the model request progress inside the8 executed points, or repeatedly only in the future that is replaced? Also inspect grasp centering, command/actual opening and target/reference installation. Do not change replan cadence, horizon, source IK, gate or teacher from these correlations alone. No hidden target state or teacher phase may enter training inputs or online execution.

The main R3 thresholds and remaining authorization apply unchanged: fixed TRAIN4>=3/4 -> DEV8>=6/8 -> freeze one candidate -> CONFIRM16>=12/16; original task predicate/horizon; up to8 snapshots and2 DEV candidates total; one fresh confirmation only;07:00 training boundary and08:00 actual handoff. Local training/recovery/evaluation first; HF, full matched CONTROL and multiscene work remain deferred. selected/deployment remain NONE. A successful source-simulation pilot is not a Go2 or hardware result.

## 5. Machine-readable delta and budget

- primary final status: VALID_FAIL_TRAIN0_OF4; no repeated primary evaluation required.
- sole FOCUS job:651332, adopt actual status; no duplicate.
- no additional formal updates: shared PI0277 cap8192, remaining FOCUS at most4096 less actual newer work.
- no additional qualification cap, quality query cap, physical starts, teacher or TRAIN data allowance.
- one new permitted objective phase ONLY on the registered recurrence trigger and within the remaining existing FOCUS steps.
- with primary final screen now completed, remaining nominal work is at most4*4 TRAIN +2*8 DEV +16 CONFIRM =48 learner episodes;48*80+4*16=3904 quality queries,153600 nominal physics steps. Subtract all actual prior work; these are not fresh entitlements.
- all old3/4 envelope failures, the full-run failure and old IK cases are retained.

Reviewer actions: inspected concurrent terminal/envelope/queue receipts; no SSH, native execution or model calls. The initial R3 review correctly described its earlier snapshot, but this addendum supplies the newer state and explicitly narrows/amends its no-new-loss clause. The canonical must link BOTH protocol and addendum so the executor cannot miss the conditional authority or mistake it for blanket permission.
# PI autonomy blackboard

This issue is the persistent governance surface for long-horizon Corelab/Hopper execution in the Go2 + ARX5 hierarchical VLA project.

It does **not** replace stage-specific PR evidence. Stage PRs contain implementation, protocols, reports, hashes, and execution records. This issue defines who may do what next and which gates may be crossed autonomously.

## 0. Scientific objective

Maintain and test the architecture:

```text
Vision + Language + proprioception
        -> GR00T
        -> task-space motion intent / EEF trajectory
        -> general trajectory-tracking WBC
        -> Go2 + ARX5
```

Current claim boundary remains conservative:

- dense H40 relative EEF is the primary baseline;
- Stage16 qualified native WBC tracking for smooth native references;
- Stage20S integrated the full GR00T -> provider/SafetyGate -> WBC chain but exposed an aggressive/oscillatory model trajectory profile;
- Stage23 ruled out the audited action transform/mask/order/evaluator as the primary defect and produced a bounded negative O1@200 result;
- Stage24 established that BASE category ID10 (`NEW_EMBODIMENT`) is initializer/unused-like and that static inactive/active coupling is not ID10-anomalous;
- Stage25 is determining how much O1@200 actually changed ID10 and shared diffusion parameters before selecting the next causal Hopper experiment.

Do not promote these observations into a general claim that GR00T cannot learn Go2+ARX5, that dense EEF is fundamentally invalid, or that any single hypothesized mechanism is causal unless a later controlled experiment closes that claim.

---

# 1. Authority hierarchy

Before every new action, each agent must resolve instructions in this order:

1. **latest explicit GPT/PI gate comment in this issue**;
2. current active stage PR and its latest GPT/PI directive;
3. immutable stage reports / seals / ADRs;
4. older handoffs and historical comments.

A newer PI gate overrides an older authorization. Silence is not authorization.

If two instructions conflict, choose the more restrictive interpretation and report the conflict.

---

# 2. Roles

## GPT / PI

Owns:

- scientific question and hypothesis selection;
- claim boundary;
- representation/ABI decisions;
- promotion / HOLD / STOP / BLOCKED decisions;
- authorization of new model execution, H200 experiments, WBC/hardware work, validation/holdout reads;
- choosing between competing causal experiments.

## Corelab agent

Owns:

- source audit;
- data/checkpoint/config identity;
- simulator/WBC integration implementation;
- provenance and artifact sealing;
- experiment harness/protocol implementation;
- CPU/static diagnostics;
- bounded engineering fixes that preserve a frozen scientific protocol;
- stage reports, ADRs, handoffs, tests, branch/PR hygiene.

Corelab is the **implementation/provenance owner**, not the scientific decision maker.

## Hopper agent

Owns:

- execution of already-authorized CPU/PBS/H200 experiments;
- exact input verification;
- scheduler/runtime operation;
- logs/counters/checkpoint/output sealing;
- read-only result extraction and machine-readable execution returns.

Hopper is the **execution owner**, not the protocol designer. If the frozen harness/protocol is wrong or incomplete, Hopper fails closed and returns to Corelab/PI; it does not silently patch scientific semantics.

---

# 3. Gate classes

## LOCAL gate — agent may cross autonomously

An agent may continue without a new PI message only when all of the following hold:

- the scientific question is unchanged;
- the exact dataset split and locked-data boundary are unchanged;
- the model/action/state ABI is unchanged;
- no new model forward/inference/training class is introduced;
- no new GPU/H200 class is introduced;
- no metric/pass threshold is changed after seeing results;
- no checkpoint/source is substituted;
- no claim is promoted;
- the work is a mechanical/provenance/test/runtime correction inside the already-authorized scope.

Examples:

- fix a path parser while retaining the same identity rules;
- add fail-closed hash/shape/dtype checks;
- repair logging or report serialization;
- retry a transient scheduler allocation under the exact same frozen experiment;
- recompute hashes or scalar summaries from already-authorized sealed artifacts;
- add tests that preserve the frozen scientific contract.

## PI gate — mandatory STOP

The agent must stop and request GPT/PI review before any of the following:

- selecting between scientific hypotheses/experiments;
- H1 vs H2 vs H3 vs H4 selection;
- first H200/GPU model execution for a new experiment class;
- increasing experiment exposure/budget beyond the explicitly authorized ceiling;
- modifying loss, optimizer semantics, sampler, horizon, representation, normalization, action/state ABI, timing semantics, model architecture, category mapping, trainable-module policy, or WBC interface;
- reading validation `e`, locked holdout `f`, or any new evaluation split;
- WBC rollout after a model-side diagnosis unless explicitly authorized;
- ROS/ZMQ/P1M/physical hardware;
- replacing or re-downloading an unverified checkpoint/source;
- changing a frozen PASS threshold after seeing results;
- unexpected frozen-module changes, provenance ambiguity, or a result that changes the causal interpretation;
- representation pivot away from the current dense relative-EEF baseline;
- scientific claim promotion.

---

# 4. Fail-closed rule

When an identity/provenance/gate assertion fails:

```text
STOP
-> preserve evidence
-> do not weaken the check
-> do not substitute another artifact
-> do not repair history
-> do not continue to tensor/model execution
-> report BLOCKED with the exact failed predicate
```

Credential, SSH-key, password, permission, VPN, access-control, or security-setting changes are never an autonomous recovery path.

---

# 5. Agent-to-agent handoff protocol

Agents communicate through GitHub only.

An agent may cite another agent's completed GitHub evidence, but **one agent cannot create new scientific authorization for the other**.

For an automatic handoff to be valid, the latest PI gate must already contain the downstream authorization token. Otherwise the upstream agent finishes and stops.

Every stage return must contain:

```text
ROLE
STAGE
GIT_SHA / EXECUTION_CODE_SHA
ACTIVE_PR
EXECUTION_CONTEXT
INPUT_IDENTITIES
AUTHORIZED_SCOPE
ACTUAL_ACTIONS
FORBIDDEN_ACTIONS_CONFIRMED_FALSE
RESULT: PASS / STOP / BLOCKED
BLOCKER
EVIDENCE_HASHES
CLAIM_BOUNDARY
NEXT_GATE_REQUEST
```

Do not report only prose such as "done" or "works".

---

# 6. Git/PR discipline

- Keep scientific stages in Draft PRs until PI closes the stage.
- Do not merge stage PRs autonomously.
- Do not rewrite immutable historical evidence.
- Commit code/protocol/report changes; do not commit large model/checkpoint binaries.
- Large binaries remain on the authorized execution host and are represented on GitHub by hashes, manifests, scalar summaries, and provenance.
- A new stage branch may be created autonomously only when the latest PI gate explicitly names the next stage or authorizes `NEXT_STAGE_BRANCH_ALLOWED=true`.
- Otherwise continue the current stage PR or stop.

---

# 7. Data boundary

Default unless a later PI gate explicitly changes it:

```text
train / authorized tiny fixtures: allowed as named by the current protocol
validation e: forbidden
holdout f: forbidden
physical hardware logs outside an authorized hardware stage: forbidden
```

A metric computed on training fixtures is a learnability/diagnostic result, not a generalization claim.

---

# 8. Current frozen state

S1 Corelab supplemental-history binding at PR #10 HEAD
`bd81e1ad796de1bb133d915ab38881384a0bb96e`
is accepted for the Stage25 provenance contract.

Current gate state:

```text
ACTIVE_STAGE: Stage25 BASE -> O1@200 parameter-delta gate
ACTIVE_PR: #10
CORELAB: SUPPORT / HOLD unless Hopper exposes an implementation/provenance blocker
HOPPER_CPU_STATIC_DELTA: AUTHORIZED
HOPPER_H200: HOLD
H1: HOLD
H2: HOLD
H3: HOLD
H4: HOLD
VALIDATION_E: LOCKED
HOLDOUT_F: LOCKED
WBC: LOCKED
HARDWARE: LOCKED
NEXT_SCIENCE_GATE: GPT_PI_REVIEW_AFTER_STAGE25_PARAMETER_DELTA
```

The Stage25 CPU/static execution authorization is only for the exact frozen BASE and O1@200 pair and the corrected PR #10 harness. It does not authorize model construction, forward, inference, training, GPU, or H200.

---

# 9. Long-horizon operating principle

The goal of autonomy is to remove unnecessary PI micromanagement of **mechanical execution**, not to delegate scientific hypothesis selection.

Agents should independently diagnose and repair local engineering/provenance failures inside a frozen contract, keep strong evidence, and make progress until they hit a PI gate. At a PI gate they must stop with a decision-ready evidence package rather than guessing the next science experiment.

Separate comments below provide the exact long-horizon prompts for Corelab and Hopper.
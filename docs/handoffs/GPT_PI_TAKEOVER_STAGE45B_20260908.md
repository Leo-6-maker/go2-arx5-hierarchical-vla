# GPT / PI takeover handoff — Stage45B FULL24 fresh-base adaptation

Date: 2026-09-08 (Asia/Singapore)

This document is a **handoff to a new GPT/PI review window**. It is not a new scientific result and it does not itself authorize any additional execution beyond the latest explicit authority in GitHub Issue #11.

The purpose is to make the new GPT/PI independently reconstruct and audit the project state before it issues any new task, gate, training instruction, rollout instruction, or claim.

---

## 0. Mandatory takeover rule

The new GPT/PI MUST NOT begin by asking an agent to continue training, submit an H200 job, run WBC, generate new data, merge Stage45B, or change the protocol.

Its first task is **read-only scientific/provenance review**.

Required order:

1. Read GitHub Issue #11 body.
2. Read the latest explicit `PI_GATE_STATE_*` comment in Issue #11. At handoff time the latest known gate is `PI_GATE_STATE_0073`, comment `5584578126`.
3. Read the latest Stage45B Corelab return in Issue #11, comment `5585110534`.
4. Read this handoff in full.
5. Read the current Stage45B execution artifacts on the exact frozen branch/HEAD:
   - `manifests/STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1.json`
   - `docs/stages/STAGE45B_H200_HANDOFF.md`
   - `scripts/stage45b_full24_fresh_base_train.py`
   - `hpc/pbs/stage45b_full24_fresh_base.pbs`
6. Read repository-level scientific summaries:
   - `docs/RESEARCH_TIMELINE.md`
   - `docs/ACCEPTED_BASELINES.md`
   - `docs/REPOSITORY_GUIDE.md`
7. Independently verify Git branch/SHA lineage and whether a newer Issue #11 gate or agent return supersedes anything in this document.
8. Only after the above, produce an explicit takeover audit (template in section 15).
9. Only after that audit is internally consistent may the new GPT/PI issue a new PI gate or reaffirm an existing authorized action.

**Authority rule:** Issue #11 latest explicit PI gate > current agent return/evidence > branch manifests/reports > this handoff > old chat summaries.

If any two sources conflict, do not silently choose the convenient interpretation. Identify the conflict, preserve evidence, and fail closed until resolved.

---

## 1. Project north star

Architecture:

```text
Vision + Language + proprioception
        -> GR00T N1.7
        -> StructuredActionV1 / task-space EEF trajectory
        -> SafetyGate
        -> reference-only receding-horizon handoff
        -> general trajectory-tracking WBC
        -> Go2 + ARX5
        -> fresh observation back to GR00T
```

Current research objective is not merely to make the stack run. It is to establish a scientifically interpretable hierarchy where GR00T provides task-conditioned EEF motion intent and the WBC provides general trajectory tracking, with clear attribution when behavior fails.

Current work is still simulation/development evidence. Do not promote task success, general robustness, real-time performance, or hardware readiness without a later explicit gate and evidence.

---

## 2. Repository and governance state

Repository:

`Leo-6-maker/go2-arx5-hierarchical-vla`

Canonical governance blackboard:

GitHub Issue #11.

Repository consolidation is complete. Historical stacked PRs were archived/closed without deleting evidence branches.

Current `main` at handoff time:

```text
c54a10024c9ab7243c6cd3d2d0acab8cd399e4cd
Merge Stage45A FULL24 adaptation protocol
```

Parents:

```text
bfa14bf1b74eae908c0922575a4db671c903b9b8
49a0f873130aa36582743576f2e782f6305953fa
```

Current active Stage45B execution branch:

```text
stage45b/full24-fresh-base-execution
HEAD: d21f801fc99c457792f83e49bbde4c6db44536d2
```

This branch HEAD is **frozen execution provenance** for the current B1 package. Do not move it merely to add documentation or cleanup. This handoff is intentionally committed on a separate branch:

```text
handoff/gpt-pi-stage45b-20260908
```

Stage45B execution branch must not be merged automatically after experiment completion.

---

## 3. Current latest execution authority: PI_GATE_STATE_0073

At handoff time, Issue #11 comment `5584578126` is the latest known explicit gate.

Its central scientific question is:

> Does replacing the single-window TINY1 training coverage with the frozen 24-window Stage22 TRAIN fixture improve target-directed StructuredAction behavior under an otherwise matched fresh-base training recipe?

Primary matched-step comparison preregistered by PI:

```text
historical TINY1 STRUCTURED@1000
vs
new FRESH_BASE_FULL24 checkpoint-1000
```

Reason: both should originate from the exact same BASE family and use the same Stage38-style recipe through 1000 optimizer steps, while training-data coverage changes from 1 window to 24 windows.

`FULL24 checkpoint-2000` remains the preregistered fixed final endpoint / persistence characterization. It is not allowed to erase or replace the matched-step @1000 comparison, and `checkpoint-1000` may not be chosen downstream merely because it looks better.

No post-hoc binary task-success threshold is allowed.

---

## 4. Why Stage45 exists: accepted scientific evidence ladder

### Stage40 — exact closed-loop integration

Accepted narrow claim:

The exact `GR00T -> StructuredActionV1 -> SafetyGate -> native WBC` stack demonstrated a sealed synchronous feedback loop on the authorized TINY1 development fixture, with fresh post-WBC observations and no handoff reset/teleport/history clear.

This established integration, not task success/generalization.

### Stage41 — long-horizon engineering

10 reference-only handoffs were executed stably with deterministic same-seed repeatability. No reset/teleport/history clear/termination was observed. This established bounded engineering stability on the development fixture, not robustness.

### Stage42 — feedback/model causality

Counterfactual probes showed detectable model output dependence on proprioception, body vision, and wrist vision. Historical endpoint semantics and gripper soft-limit diagnostics were also resolved. This did not establish equal modality importance or task grounding.

### Stage43 — controlled initial-condition cohort

Five fixed pre-episode base translations were tested:

```text
C0 nominal
C1 +0.05 m X
C2 -0.05 m X
C3 +0.05 m Y
C4 -0.05 m Y
```

All five closed loops were lifecycle-stable; WBC trajectory tracking remained roughly ~2 cm RMS. However actual EEF target progress was negative in all five conditions.

Representative final target distances were roughly 0.40–0.48 m, much larger than WBC tracking error. This was the major evidence that WBC was not the leading failure source.

### Stage44R — model/data attribution repair

Accepted diagnosis:

`STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED`

Key evidence:

- R1: expert labels were correct and target-directed: 24/24 positive target progress, 24/24 positive endpoint cosine.
- R2: Stage43 negative behavior was already present at the decoded GR00T reference layer:
  - decoded GR00T: 20/20 negative
  - installed reference: 20/20 negative
  - achieved EEF: 20/20 negative
  - no binding mismatch and no WBC tracking mismatch was supported.
- R3 primary exact training cell `red_target_train_a/expert-0000`:
  - predicted progress `+0.113261 m`
  - target-vector cosine `0.997154`
  - endpoint error `0.009431 m`
- Across the 24 archived Stage44 outputs, 18/24 were target-progress positive.

Interpretation accepted narrowly:

The frozen STRUCTURED@1000 checkpoint can fit its exact single training cell, but broader target/replanning-state coverage is insufficient. This supports expanding training coverage before blaming the training representation, binding layer, or WBC.

---

## 5. Data provenance: TINY1 vs FULL24

### TINY1 historical diagnostic

Exactly one training window:

```text
red_target_train_a / expert-0000
40 frames
1 of 24 available Stage22 TRAIN windows
```

It was used intentionally as a learnability/representation diagnostic, not as a general policy dataset.

### FULL24_EXISTING_TRAIN

Existing Stage22 TRAIN-only materialization; no new simulator data are needed for Stage45B.

Frozen identity:

```text
FULL24 canonical tree SHA256:
2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789

regular files: 85
total bytes: 919661
materialized episodes/windows: 24
frames: 960 = 24 x H40
```

Composition:

```text
4 TRAIN targets x 6 replanning snapshots = 24 windows
```

TRAIN targets:

```text
red_target_train_a [0.30, +0.10, 0.58]
red_target_train_b [0.30, -0.10, 0.58]
red_target_train_c [0.30, +0.05, 0.64]
red_target_train_d [0.30, -0.05, 0.63]
```

Validation and holdout remain locked:

```text
red_target_holdout_e = Validation-E, LOCKED
red_target_holdout_f = Holdout-F, LOCKED
```

Target XYZ is evaluation/provenance metadata only and is not a model input tensor.

Language remains:

`move the end effector to the red target`

Model-facing inputs remain body RGB, wrist RGB, StateV1, and language.

---

## 6. Frozen model/action/runtime contract

Do not silently change any of the following:

```text
GR00T source:
51d4c89f72fda44cbf77285c6a8114b52676b8a1

exact BASE model tree:
8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4

historical STRUCTURED@1000 tree:
30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137

Stage22 stats:
fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23

Stage22 relative stats:
8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1

Cosmos revision:
9ce19a195e423419c349abfc86fd07178b230561

Cosmos materialized tree:
1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9

modality config:
e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031
```

Action/runtime contract:

```text
StructuredActionV1
K16 / T16 / H40
universal D132 -> physical D10
physical D10 = EEF 9D + gripper 1D
frame = NATIVE_WBC_RELATIVE_REFERENCE
TCP = x5_link6_gripper_center
exact D1 StructuredAction decoder
numeric target XYZ excluded from model input
```

Historical live-loop timing that must not be conflated:

```text
H40 action sample period = 0.05 s
handoff target = 0.975 s
executor prefix = round(0.975 / 0.05) = 20 action samples
native WBC dt ~0.02 s on Corelab
native pre-handoff schedule = round(0.975 / native_dt) = 49 native ticks
```

Do not confuse 20 H40 action samples with 49 native WBC ticks.

---

## 7. Frozen Stage45B training intervention

Only authorized intervention:

`B_FRESH_BASE_FULL24`

Starting weights:

```text
EXACT BASE
8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
```

Option A warm-start from historical STRUCTURED@1000 is forbidden under the current gate.

Training recipe:

```yaml
seed: 2222
max_steps: 2000
save_steps: [1000, 2000]
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
optimizer: adamw_torch
learning_rate: 0.0001
weight_decay: 0.00001
lr_scheduler_type: cosine
warmup_steps: 10
warmup_ratio: 0.0
max_grad_norm: 1.0
bf16: true
fp16: false
tf32: true
resume_from_checkpoint: false
save_only_model: false
tune_projector: true
tune_diffusion_model: true
tune_visual: false
tune_llm: false
tune_vlln: false
use_relative_action: true
use_percentiles: true
num_gpus: 1
```

No precision, quantization, offload, image-resolution, model-input, action-representation, normalization-policy, optimizer, seed, or budget change is allowed as a convenience fix.

---

## 8. Stage45B B0 — COMPLETE / PASS

Issue #11 return comment: `5585110534`.

Token:

`CORELAB_STAGE45B_FULL24_STRUCTURED_STATS_CLOSED`

B0 facts:

```text
FULL24 TRAIN labels: 24 exact windows
FULL24 tree: 2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789
statistics policy: MINMAX_TO_NEG1_POS1
selected bounds: q01/q99
use_percentiles: true
inactive universal dimensions excluded
padding contribution excluded
Validation-E/Holdout-F reads: 0
```

B0 StructuredAction statistics artifact SHA256:

```text
49c7139dff09d5795e31205c7e72cb11e5801ce5f176572ebc1ec8d91f310c3d
```

Reported payload SHA:

```text
c31c818121ac01a402fc70e577db225ffb6f5d110649aef9dada94338c8a5032
```

A transport copy is committed under `reports/artifacts/`; decoded JSON must reproduce the B0 artifact SHA exactly.

B0 execution exposure:

```text
GR00T = 0
WBC = 0
reference updates = 0
training/backward/optimizer = 0
GPU/PBS = 0
```

---

## 9. Stage45B B1 — COMPLETE / PASS

Token:

`CORELAB_STAGE45B_H200_ADMISSION_READY`

Frozen execution branch/HEAD:

```text
stage45b/full24-fresh-base-execution
d21f801fc99c457792f83e49bbde4c6db44536d2
```

Frozen files:

```text
manifests/STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1.json
docs/stages/STAGE45B_H200_HANDOFF.md
scripts/stage45b_full24_fresh_base_train.py
hpc/pbs/stage45b_full24_fresh_base.pbs
```

Static checks reported PASS:

```text
JSON validation
Python compile
PBS shell syntax
handoff unit tests
hash self-check
remote branch equality
clean execution preparation
```

No H200 model construction or scientific training had occurred as of this handoff.

---

## 10. Stage45B B2 — NOT YET EXECUTED

Current next scientific execution class under `PI_GATE_STATE_0073` is one H200 fresh-base FULL24 training run, subject to takeover re-audit.

Target successful token:

`HOPPER_STAGE45B_FULL24_FRESH_BASE_TRAINING_COMPLETE`

Required successful endpoint:

```text
optimizer/global step = 2000
checkpoint-1000 retained with full trainer state
checkpoint-2000 retained with full trainer state
```

Runtime evidence must bind job/host/GPU identity, observed global steps, loss/LR logs, peak memory, checkpoint inventories/tree hashes, and optimizer/scheduler/RNG state presence.

Retry rule is strict:

- failure before first optimizer step: only an identical-bundle mechanical replacement may be considered after proving zero optimizer exposure;
- once any optimizer step occurred, failure/preemption/OOM is PI STOP; no resume or silent rerun.

Do not use Vanda A40 for this intervention.

---

## 11. Stage45B B3 — planned offline evaluation after successful training

Issue #11 gate 0073 preregisters TRAIN-only offline evaluation of both new checkpoints:

```text
FRESH_BASE_FULL24@1000
FRESH_BASE_FULL24@2000
```

on exactly 24 archived TRAIN observations.

Maximum official free-running sampler exposure:

```text
24 x 2 = 48 GR00T get_action calls
```

Historical TINY1 STRUCTURED@1000 baseline should be reused from sealed Stage44 A3 outputs rather than resampled.

Primary comparison:

```text
historical TINY1 STRUCTURED@1000
vs
new FULL24 checkpoint-1000
```

Secondary comparison:

```text
FULL24 checkpoint-1000
vs
FULL24 checkpoint-2000
```

Metrics include target progress/sign, target-vector cosine, predicted-vs-expert EEF RMSE/max, endpoint error, path-length ratio, speed ratios, roughness, initial continuity, raw/decoded action hashes, finite/shape contract, and offline SafetyGate characterization.

Claim boundary:

Improvement on these 24 TRAIN windows is training-distribution fit/coverage evidence. It is not unseen-target generalization, closed-loop task success, or robustness.

Operational note for takeover review: the dedicated H200 handoff document tells the H200 operator to stop after B2 and return artifacts before B3. This is compatible with separating training execution from Corelab evaluation, but the new GPT/PI should explicitly review this role boundary before allowing B3. Do not let an H200 operator automatically proceed into B3 merely because gate 0073 describes both stages.

---

## 12. Three-server operating model

The infrastructure strategy has now been clarified.

### Corelab — primary development / control plane

Strengths:

```text
fast local iteration
CPU/static tests
small-GPU GR00T inference/debug
Isaac/WBC/runtime integration
manifest/evidence generation
```

Corelab is the preferred main development server.

### Vanda — artifact mirror / staging / recovery bridge

Role:

```text
Hugging Face artifact mirror
large-file staging
Hopper network bridge if needed
independent replication later
```

Vanda A40 remains scientifically HOLD unless a future gate explicitly authorizes it.

### Hopper — H200 execution / checkpoint origin

Role:

```text
pre-stage immutable artifacts on login/persistent storage
CPU/static/hash preflight before GPU allocation
request H200 only after everything is ready
perform frozen large-model training
write checkpoints to persistent storage
release H200 before non-GPU upload work when possible
```

H200 is not the main development environment. The queue is expensive; imports, paths, hashes, configs, PBS syntax, data availability, and artifact staging should be debugged before requesting GPU resources.

### Network topology

Direct Corelab-to-Hopper and Corelab-to-Vanda connectivity is unavailable/unreliable for project transfer. Vanda and Hopper can communicate directly. All three machines have independently demonstrated direct HTTPS access to Hugging Face.

Therefore:

```text
GitHub = source code + configs + small manifests + provenance
Hugging Face = portable model/data artifact registry
Vanda = optional mirror/relay, not mandatory if Hopper can pull directly
```

---

## 13. Hugging Face infrastructure state

All three servers have now been configured for Hugging Face access under user namespace:

`leoyuer`

Operational rules:

- use per-server fine-grained access tokens;
- never ask the user to paste raw tokens into chat;
- never print token values in logs/manifests/Issue #11;
- token files should be private (mode 600; config directory mode 700);
- large model cache lives on server-appropriate large/persistent storage, not ordinary small home quota;
- HF is transport/storage, not scientific authority;
- every scientific artifact must still be pinned by HF revision/commit where applicable plus deterministic materialized tree/file hashes and GitHub evidence seal.

Server role intent:

```text
Corelab: read/write artifact publishing
Vanda: read/write mirror/staging
Hopper: read/write may be used for project checkpoint publishing, but training jobs should not waste H200 allocation uploading large artifacts when upload can occur after GPU release
```

Do not record credential fragments as provenance. Record only authenticated username, CLI version if relevant, repository/revision, and artifact hashes.

---

## 14. Global locks / forbidden operations at takeover time

Unless a newer Issue #11 gate explicitly changes them, treat the following as locked:

```text
Validation-E: LOCKED
Holdout-F: LOCKED
new simulator/data collection: FORBIDDEN
MULTIINIT120 generation: FORBIDDEN
Stage43 closed-loop rerun: HOLD
WBC execution for Stage45B: HOLD
Vanda A40 scientific execution: HOLD
hardware / ROS / ZMQ / P1M: LOCKED
Option A WARM_START_FULL24: FORBIDDEN
Stage45B execution branch auto-merge: FORBIDDEN
post-hoc success threshold: FORBIDDEN
outcome-dependent checkpoint/recipe/statistics swap: FORBIDDEN
```

Also preserve the general governance rules:

```text
silence is not authorization
fail closed on ambiguity
no autonomous scientific claim promotion
no unrecorded model/checkpoint/data substitution
no branch/history rewrite to make evidence look cleaner
```

---

## 15. Mandatory first response from the new GPT/PI

Before issuing any new task, the new GPT/PI should perform GitHub read-only review and return something equivalent to:

```yaml
GPT_PI_TAKEOVER_AUDIT:
  latest_issue11_gate_read:
  latest_gate_comment_id:
  newer_gate_found: true/false

  repository:
    main_sha:
    stage45b_branch:
    stage45b_head:
    lineage_verified: true/false

  scientific_state:
    stage40_closed_loop_understood: true/false
    stage41_long_horizon_understood: true/false
    stage42_feedback_causality_understood: true/false
    stage43_negative_target_progress_understood: true/false
    stage44r_diagnosis: STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED
    stage45_primary_question_restated:

  data:
    tiny1_scope_restated:
    full24_tree_sha256:
    full24_scope_restated:
    validation_e_locked: true/false
    holdout_f_locked: true/false

  stage45b:
    b0_status:
    b0_stats_sha256:
    b1_status:
    execution_manifest_reviewed: true/false
    runner_reviewed: true/false
    pbs_reviewed: true/false
    h200_handoff_reviewed: true/false
    b2_executed: true/false
    optimizer_exposure_seen: 0/unknown/nonzero

  infrastructure:
    corelab_role:
    vanda_role:
    hopper_role:
    github_role:
    huggingface_role:

  frozen_contract:
    base_tree_verified:
    groot_source_verified:
    cosmos_identity_verified:
    structured_action_contract_verified:
    training_recipe_verified:

  ambiguities_or_blockers:
    - ...

  claim_boundary:
    - ...

  takeover_result: PASS_TO_GOVERN / BLOCKED_PENDING_REVIEW
```

The new GPT/PI should not simply copy this handoff's conclusions. It should verify the live repository and Issue #11 state and identify any drift.

Only if `takeover_result = PASS_TO_GOVERN` should it decide whether to:

1. reaffirm the already-authorized B2 H200 execution after final Hopper preflight;
2. modify/supersede gate 0073 if a new blocker or infrastructure fact changes the execution plan;
3. issue a separate gate after B2 before B3, if it prefers an explicit training-to-evaluation stop boundary.

---

## 16. Specific takeover questions the new GPT/PI must answer

The new GPT should be able to answer, from repository evidence rather than memory alone:

1. Why did Stage43 fail despite good WBC tracking?
2. What exact evidence in Stage44R attributes first negative direction to decoded GR00T rather than binding/WBC?
3. Why does the exact TINY1 source cell support coverage limitation rather than a complete training-fit failure?
4. Where did FULL24 come from, and why is it not new data generation?
5. Why is `TINY1@1000 vs FULL24@1000` the primary matched-step comparison?
6. Why is `FULL24@2000` still retained even if @1000 looks better?
7. Why is improvement on TRAIN24 not yet generalization?
8. Why is H200 needed even though FULL24 is only 960 frames?
9. Why should Hopper be execution-only rather than the main development server?
10. What exact retry rule applies if H200 fails before vs after the first optimizer step?
11. Which splits/resources are still forbidden?
12. Which exact branch/HEAD and B0 stats hash define the current execution package?
13. Why must the handoff not move the frozen Stage45B execution branch tip?
14. What would constitute enough evidence to proceed from Stage45B offline results to a later Stage43 closed-loop comparison?

If the new GPT cannot answer these coherently, it has not completed takeover review and should not issue a new experiment.

---

## 17. Current next decision point

At handoff time, the expected immediate next operation is **not automatically “submit H200”**.

The expected next PI action is:

```text
new GPT/PI takeover audit
        -> verify live Issue #11 + branch/manifests
        -> verify no newer evidence or optimizer exposure
        -> verify Hopper artifacts/environment can be staged without consuming H200 queue time
        -> then either reaffirm one frozen B_FRESH_BASE_FULL24 H200 execution or block with explicit reason
```

If B2 later completes successfully, preserve the exact training artifacts and stop the H200 operator. Then the PI reviews training evidence and decides whether/where B3 train-only offline evaluation should run.

---

## 18. Final claim boundary for takeover

What is currently supported:

- exact multimodal GR00T -> StructuredActionV1 -> SafetyGate -> WBC closed-loop integration on development fixtures;
- long-horizon lifecycle stability and deterministic repeatability at tested scope;
- measurable model dependence on all three tested observation modalities;
- Stage43 stable execution but negative target progress under five controlled initial conditions;
- Stage44R model-side attribution and single-sample coverage limitation diagnosis;
- a frozen FULL24 train-only adaptation protocol, FULL24 statistics closure, and H200 execution package.

What is NOT currently supported:

```text
Stage45B training success (not yet run at handoff)
FULL24 improvement (not yet measured)
unseen-target generalization
Stage43 recovery after FULL24 training
binary task success
general robustness
real-time/asynchronous qualification
physical Go2 + ARX5 readiness
hardware safety claim
```

The new GPT/PI must preserve this boundary until new sealed evidence changes it.

---

## 19. Takeover marker

Recommended takeover marker after the new GPT completes the mandatory read-only audit:

`GPT_PI_STAGE45B_TAKEOVER_AUDIT_COMPLETE`

This marker should mean only that the new PI understands and has independently verified the current project state. It is **not** itself an H200 execution token.

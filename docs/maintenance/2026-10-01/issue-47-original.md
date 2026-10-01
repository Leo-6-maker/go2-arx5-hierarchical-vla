# VANDA AGENT TASK — Reusable train/eval infrastructure freeze

## Objective
Create a stable, reusable **future-facing** training/evaluation infrastructure while preserving the exact historical Stage45B/Stage45C evidence chain.

This task is **codebase convergence**, not new scientific execution.

The key rule is:

> Do not rewrite, replace, reinterpret, or re-run Stage45B/Stage45C evidence-producing paths merely to make the repository cleaner.

Historical execution identity remains frozen. New reusable code must be introduced as a canonical path for **future experiments**, with regression tests showing behavioral equivalence to frozen Stage45 contracts where appropriate.

---

## Current authority / evidence boundary

Treat the following as immutable scientific history:

- Canonical Stage45C scientific execution identity:
  - branch: `stage45c/b2-checkpoint-train24-evaluation`
  - execution HEAD / merge commit: `74e77d83d85ef9b290c8f3642f242fb25aaa87cb`
- Stage45C successful H200 job:
  - `609482.hopper-m-02`
- Stage45C execution gate:
  - `PI_GATE_STATE_0100`
- Stage45C evaluation scope:
  - `BASE x24`
  - `STRUCTURED@1000 x24`
  - `FULL24@1000 x24`
  - `FULL24@2000 x24`
  - total 96 checkpoint-window evaluations / 384 action-head sampler calls
- Stage45B accepted training execution:
  - accepted training HEAD: `8197e4cd591b9c44cf9f43576d3020e9e6ff5f24`
  - successful job: `608087.hopper-m-02`

Do not change the meaning of those historical runs.

PR #46 is reporting/docs-only and is not the Stage45C scientific execution identity.

---

# Required architecture

Refactor toward a three-layer pattern:

```text
experiment manifest
      ↓
generic reusable runner
      ↓
stage-specific frozen wrapper
```

The repository should distinguish clearly between:

1. **generic future infrastructure**
2. **experiment-specific configuration/manifests**
3. **historical frozen execution wrappers / evidence artifacts**

Do not collapse these categories.

---

# Workstream A — Generic evaluation executor

Create a reusable evaluation entry point, preferably along the lines of:

```text
scripts/eval_checkpoint_cohort.py
```

Name may differ if repository conventions strongly suggest another path, but the semantic role must remain clear.

The generic evaluator should be **manifest-driven** and should preserve the following Stage45C-proven invariants:

- explicit cohort / checkpoint configuration
- checkpoint runtime mapping and checkpoint-specific decode statistics
- deterministic same-noise-across-checkpoints support
- fixed sampler contract support
- raw prediction persisted **before decode**
- raw StructuredAction tensor contract `[1,40,132]`
- decoder path:
  - inverse normalization
  - StructuredAction reconstruction
  - K16/T16 structured decode
  - physical trajectory output `[40,10]`
- no `target_xyz` in model input
- target XYZ remains metadata/expert/evaluation only
- no hidden WBC / simulator / hardware side effects
- complete counters / hashes / provenance emission
- fail-closed binding/provenance validation
- offline/local checkpoint construction support
- explicit execution identity fields

The Stage45C four-step action-head sampler contract must be represented as configuration/adapter logic, not silently replaced with a public convenience API.

Historical Stage45C used source-mapped timesteps:

```text
[0, 250, 500, 750]
```

with `dt = 0.25`.

Do not silently alter that historical meaning.

---

# Workstream B — Pure result analyzer

Create a separate reusable **read-only** analyzer, preferably:

```text
scripts/analyze_checkpoint_eval.py
```

This script must not instantiate the model or perform inference.

It should consume existing evaluation artifacts such as:

- output manifest
- per-window rows
- checkpoint aggregate rows
- raw prediction manifest/hashes when needed for provenance checks

It should support at minimum:

## 1. Per-window view
For each checkpoint × window, expose metrics including available fields such as:

- initial target distance
- final target distance
- minimum target distance
- `absolute_progress_m`
- `best_progress_m`
- progress sign
- `endpoint_target_vector_cosine`
- endpoint displacement
- path length
- maximum excursion
- roughness / second-difference metric
- endpoint error where expert reference exists
- gripper RMSE where defined
- gate / validity status if present

## 2. Per-target grouping
Stage45C TRAIN24 is effectively structured as multiple windows per target. The analyzer must support deterministic grouping by target identity **without changing the underlying sealed rows**.

Expected analysis hierarchy:

```text
per-window
   ↓
per-target
   ↓
whole-cohort
```

## 3. Paired checkpoint comparisons
Because Stage45C uses matched windows and same noise across checkpoints, support paired deltas such as:

```text
FULL24@1000 - STRUCTURED@1000
FULL24@2000 - FULL24@1000
```

Primary comparison:

```text
STRUCTURED@1000 vs FULL24@1000
```

Secondary comparison:

```text
FULL24@1000 vs FULL24@2000
```

The analyzer should report distributional evidence, not only means.

Suggested outputs:

- per-window CSV/JSON summary
- per-target summary
- whole-cohort summary
- paired delta table
- compact Markdown report

Do not hard-code a scientific success threshold unless an existing governed contract defines one.

---

# Workstream C — Generic training runner

Refactor toward a reusable training entry point, preferably something like:

```text
scripts/train_adaptation.py
```

The generic training runner should handle infrastructure concerns such as:

- exact base checkpoint identity
- checkpoint tree hashing
- statistics binding
- provenance manifest generation
- offline/local model construction
- trainer-state preservation
- full checkpoint save semantics
- preflight/runtime identity checks
- output sealing
- failure sealing
- explicit resume vs fresh-base semantics

Experiment-specific values must live in manifests/config, not become hidden defaults.

For example, Stage45B FULL24 fresh-base training parameters are historical experiment configuration:

```text
max_steps = 2000
warmup_steps = 10
scheduler = cosine
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 1e-5
seed = 2222
batch_size = 1
grad_accumulation = 1
bf16 = true
fp16 = false
tune_projector = true
tune_diffusion_model = true
tune_visual = false
tune_llm = false
tune_vlln = false
resume = false
save_only_model = false
save checkpoints = 1000, 2000
```

Do not encode this whole recipe as generic runner defaults.

Also preserve the semantic distinction that Stage45B was **fresh BASE 2000-step training** with fixed warmup=10 and cosine schedule. Do not relabel it as an exact O1+1800 resume or exposure-only continuation.

---

# Historical wrappers must remain frozen

Do not delete or rewrite historical execution scripts merely because generic replacements exist.

In particular, Stage45B/Stage45C execution-producing scripts should remain inspectable as historical artifacts.

Preferred pattern:

```text
historical stage wrapper
    -> imports/calls shared future-safe helpers only if this does not rewrite historical evidence

future generic runner
    -> canonical path for new experiments
```

If changing an old historical wrapper would alter its source identity or complicate provenance, leave it untouched and instead create a new canonical implementation.

---

# StructuredAction / EEF interface invariants

Do not change these contracts during code cleanup:

- raw model action tensor: `[1,40,132]`
- horizon `H=40`
- universal action dim `D=132`
- StructuredAction geometry controls `K=16`
- timing controls `T=16`
- physical materialization `[40,10]`
  - XYZ 3
  - rot6d 6
  - gripper 1
- external VLA semantic ownership = EEF task-space intent
- WBC semantic ownership = body/gait/joint realization
- gripper remains parallel channel
- StructuredAction geometry convention and StateV1 rot6d convention remain distinct contract layers
- provider thin waist rejects raw universal tensor downstream
- SafetyGate remains non-corrective
- reference binding remains explicit

No opportunistic representation cleanup is allowed in this task.

---

# Required regression tests

Add tests that prove the generic code does not silently alter governed behavior.

At minimum cover:

1. manifest/provenance validation
2. checkpoint-specific statistics binding
3. target XYZ excluded from model input
4. raw saved before decode
5. universal `[1,40,132]` contract
6. physical `[40,10]` decode contract
7. same-noise-across-checkpoints pairing
8. sampler contract explicitly configured
9. zero inference in analyzer
10. no WBC/simulator/hardware imports/actions in generic checkpoint evaluator
11. historical mapping provenance and current execution authority remain distinct concepts
12. no silent resume semantics in generic training runner

Prefer CPU/synthetic fixtures for regression tests.

Do **not** re-run Stage45C H200 inference for this refactor.

---

# Documentation deliverables

Add/update concise repository documentation explaining:

- which scripts are historical/frozen
- which scripts are canonical reusable future infrastructure
- how experiment manifests bind checkpoints/data/statistics/sampler/evaluation
- how analyzer consumes sealed outputs without inference
- how to reproduce analysis from existing results

Suggested document:

```text
docs/EXPERIMENT_RUNNER_ARCHITECTURE.md
```

or equivalent existing documentation location.

---

# Branch / PR expectations

Work on a new branch from the current canonical Stage45C lineage. Do not mutate the historical Stage45C branch in place unless the repository workflow requires a dedicated descendant branch.

Create a PR when implementation is complete.

The PR description must explicitly state:

- no Stage45B/Stage45C scientific evidence was re-generated
- no H200 inference was run
- no Validation-E/Holdout-F/WBC/simulator/hardware was accessed
- which old scripts remain frozen
- which generic entry points are now canonical for future experiments
- regression test results

Do not merge your own PR unless separately authorized.

---

# STOP / authority boundary

This task does **not** authorize:

- PI_GATE_STATE_0101 or any successor gate
- new H200 evaluation
- new training
- Validation-E
- Holdout-F
- WBC execution
- simulator execution
- hardware execution
- merge of PR #46
- reinterpretation of Stage45C scientific results

When complete, post a concise handoff to the PR and to Issue #11 containing:

- branch
- HEAD
- PR number
- files changed
- tests
- architecture summary
- explicit statement that historical evidence was not re-run
- remaining risks / follow-ups

Then STOP for PI review.
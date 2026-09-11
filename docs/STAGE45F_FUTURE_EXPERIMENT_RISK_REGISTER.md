# Stage45F future experiment risk register

Status: **PREPARATION ONLY — no experiment authorization**  
Authority: Issue #11 `PI_GATE_STATE_0129` / comment `5637525136`  
Base scientific state: accepted Stage45D/D2A HEAD `a843294780c1392369bfc987e468857c63fc13a5`

This document records plausible post-Stage45E branches before their results are known. It is intentionally written before candidate selection so later choices can be checked against a pre-existing decision structure rather than invented after seeing metrics.

## Current principle

The next large training run is justified only after we know **what variable we are trying to scale**. The project should avoid changing data breadth, normalization, objective, scheduler horizon, training steps, sampler, and rollout distribution at the same time.

The default sequence remains:

```text
current N2 / L1 TRAIN24 evidence
        -> seed / mechanism closure
        -> PI candidate decision
        -> recipe freeze
        -> first Validation-E gate
        -> only then broad scaling / data expansion if evidence supports it
```

Validation-E and Holdout-F remain locked during all preparation described here.

---

## Scenario A — one candidate wins TRAIN24 on only one training seed

### Risk
A single training seed can create an apparent intervention win that does not survive optimizer/initialization variance.

### Prepared response
- freeze the candidate recipe first;
- replicate with the canonical missing training seeds rather than changing the recipe;
- preserve every scientifically valid replicate, including worse runs;
- do not select the best retry by metric;
- compare the distribution of paired TRAIN24 effects across seeds.

### Do not
- open Validation-E immediately from a one-seed win;
- change LR/batch/scheduler between replication seeds;
- discard a valid replicate because its result is worse.

---

## Scenario B — candidate improves from 1000 to 2000 and appears under-trained

### Risk
Increasing `max_steps` can silently alter the learning-rate trajectory. With a cosine scheduler tied to `max_steps`, a fresh 4000-step run does **not** have the same first 2000 optimizer updates as a 2000-step run. Resuming a 2000-step checkpoint is also not automatically equivalent to a run whose scheduler was defined for 4000 steps from the start.

### Prepared response
Before any longer run:
- bind dataset, normalization, loss, optimizer, LR, weight decay, trainable modules, precision, batch and seed;
- record whether scheduler horizon is tied to `max_steps` or explicitly fixed;
- decide fresh-vs-resume as an explicit scientific choice;
- if resume is proposed, bind checkpoint + optimizer + scheduler + RNG state;
- record exact sample/window exposure rather than inferring epochs from steps;
- estimate walltime/checkpoint storage from observed telemetry.

### Classification
A longer run may only be called an exposure study when the LR/scheduler/history question has been resolved. Otherwise it is `SCALING_CONFOUNDED_OR_UNPROVEN`.

### Do not
- describe `4000/24` as exact per-window exposure;
- call changed cosine horizon a pure step-count intervention;
- silently move from fresh training to resume training.

---

## Scenario C — per-target effects are heterogeneous / coverage gaps remain

### Risk
Adding more examples can change both state/action support and robust normalization statistics. If q01/q99 are recomputed, the physical gain of normalized error changes at the same time as data coverage.

### Prepared response
For every proposed new TRAIN example:
- preserve unique window/source-episode/target identity;
- preserve independent expert/oracle label provenance;
- exclude Validation-E and Holdout-F identities;
- quantify new XYZ physical support under the **current frozen stats first**;
- report per-axis and per-window normalized out-of-range fractions;
- report target/window counts and coverage dimensions;
- only after this audit may PI choose whether stats stay frozen or are recomputed.

### Classification
- frozen stats + new data: `DATA_ONLY_CANDIDATE_REQUIRES_PI_SUPPORT_JUDGMENT`;
- changed/recomputed stats: `DATA_PLUS_NORMALIZATION`.

No automatic clipping threshold is defined in advance by the prep tooling.

### Do not
- call a new-data + new-q01/q99 run data-only;
- reuse old stats merely to preserve an ablation if clipping/support becomes scientifically unacceptable;
- use duplicated windows as evidence of new diversity.

---

## Scenario D — teacher-forced field remains good but free-running remains poor

### Risk
More static demonstration training may not fix state-distribution shift if the model leaves the teacher bridge during rollout.

### Prepared response
Consider a separately gated rollout-state augmentation / DAgger-style stage:

```text
frozen policy or policy+WBC
        -> visited TRAIN states
independent simulator expert/oracle
        -> desired future EEF labels
```

Required provenance:
- generating policy checkpoint tree;
- recipe/sampler identity;
- WBC identity if WBC generated states;
- visited-state artifact identity;
- independent simulator/oracle implementation and environment identity;
- oracle-label artifact identity;
- fixed state-collection and relabel budget before execution.

### Hard rule
GR00T may generate the visited state but **must not supervise itself**. A GR00T output copied back as a training target is not an accepted DAgger label.

### Do not
- mix rollout-generated states with self-labels;
- collect states from Validation-E/Holdout-F;
- change WBC and policy at the same time without naming the intervention accordingly.

---

## Scenario E — TRAIN24 winner appears robust across seeds

### Risk
TRAIN24 has already been repeatedly inspected and is no longer an unseen selection set.

### Prepared response
- issue one PI selection decision;
- freeze exact checkpoint, recipe, dataset, normalization, loss, sampler, decoder and metric identities;
- bind the TRAIN decision-packet hash;
- create a Validation-E access ledger before any read;
- use a fresh immutable evidence root;
- keep Holdout-F closed.

### Do not
- read Validation-E and then silently mutate the same candidate while describing the later result as a clean first validation;
- open multiple candidates on Validation-E merely because TRAIN24 is close;
- read Holdout-F for tie-breaking.

---

## Scenario F — first Validation-E result is weak

### Risk
Repeatedly tuning to the same validation split gradually converts it into another development set.

### Prepared response
The first result should be preserved with exact candidate identity. A later recipe change must be logged as a new selection generation; PI must explicitly decide whether Validation-E remains usable for another controlled selection cycle or whether a fresh development split is required. Holdout-F remains untouched until a final candidate is frozen.

### Do not
- overwrite the failed Validation result;
- call a tuned second look an unseen validation;
- use Holdout-F as an informal debugging set.

---

## Scenario G — large-scale fine-tuning is finally justified

A broad training campaign should have, at minimum:
- one frozen scientific recipe;
- cross-seed TRAIN evidence;
- explicit validation policy;
- exact dataset tree and stats route;
- exact exposure logging;
- intermediate checkpoint plan;
- observed seconds/step and checkpoint-size telemetry;
- enough scratch capacity for all retained full-state checkpoints;
- no metric-based retry selection;
- a predeclared stop/extension rule.

At that stage 4k/8k or larger budgets may be considered, but the exact numbers remain future PI decisions rather than authorization embedded in this preparation branch.

---

## Current prepared tooling

```text
stage45f_future_experiment_planner.py
  -> neutral scenario matrix

stage45f_scaling_admission.py
  -> scheduler/resume/exposure confound analysis

stage45f_exposure_accounting.py
  -> exact per-window sample exposure from future trace

stage45f_data_expansion_audit.py
  -> new-data support/clipping + DATA_ONLY vs DATA_PLUS_NORMALIZATION classification

stage45f_rollout_relabel_admission.py
  -> independent expert/oracle provenance for DAgger-style relabeling

stage45f_recipe_freeze.py
  -> exact selected-recipe ledger before any Validation-E request

stage45f_compute_budget.py
  -> telemetry-bound walltime/storage estimates only
```

Every tool remains CPU/static or input-JSON-only and returns a PI decision state rather than experiment authorization.

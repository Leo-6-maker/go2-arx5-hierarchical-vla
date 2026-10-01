# CORELAB AGENT TASK — Mirror sealed Stage45C outputs and perform read-only scientific evaluation

## Objective
Use Corelab as the **analysis environment** for the already-completed Stage45C TRAIN24 checkpoint evaluation.

This task must **not** re-run inference. Its purpose is to recover, verify, organize, and scientifically analyze the existing sealed Stage45C outputs.

The primary scientific question is whether expanding adaptation coverage from historical TINY1/STRUCTURED training to FULL24 improved TRAIN24 behavior at matched training exposure, and secondarily whether additional exposure from 1000 to 2000 steps helped.

---

# Canonical production identity

Treat the following as the execution authority for the artifacts being analyzed:

- canonical Stage45C branch:
  - `stage45c/b2-checkpoint-train24-evaluation`
- canonical scientific execution HEAD / merge commit:
  - `74e77d83d85ef9b290c8f3642f242fb25aaa87cb`
- execution gate:
  - `PI_GATE_STATE_0100`
- successful H200 production job:
  - `609482.hopper-m-02`
- node-local production output root:
  - `/scratch/e1192548/stage45c_pi0100_h200_train24_eval_74e77d83d85e_20260910`
- canonical terminal-return directory:
  - `/scratch/e1192548/stage45c_pi0100_terminal_return_609482_20260910_v2`

The reporting/docs PR #46 is not the scientific execution identity.

---

# Expected completed evaluation scope

Existing production output should correspond to exactly:

```text
BASE              × 24 TRAIN24 windows
STRUCTURED@1000    × 24 TRAIN24 windows
FULL24@1000        × 24 TRAIN24 windows
FULL24@2000        × 24 TRAIN24 windows
```

Total:

```text
96 checkpoint-window evaluations
384 source-mapped action-head sampler calls
96 raw predictions
```

Raw prediction shape:

```text
[1,40,132]
float32
```

Production semantics:

- 4 sampler calls per window
- sampler timesteps `[0,250,500,750]`
- `dt=0.25`
- raw prediction persisted before decode
- no training
- no backward
- no optimizer
- no resume
- no checkpoint writes
- no Validation-E
- no Holdout-F
- no WBC
- no simulator
- no hardware
- no network fetch

---

# Required artifact identity checks

Before scientific analysis, verify artifact integrity against the production seals.

Expected seals:

```text
frozen protocol SHA256
0f8c564f534ce3b29a5029cc927f93a03697181b4b1df0aa9e80566af4cd020f

checkpoint mapping SHA256
d2281b451970ef408566ded15851bd252021ccf4de1bf5e59df9b8edf5b4c1d7

execution release SHA256
3750f6d7d01021631f93660d5438e498860b6f7b19e3b30e8d696ee222a60532

runtime binding raw SHA256
28c3e76dadc0d0f3f94498bf21b05a2d782a0244f87aef8319d6d5bb253f846e

runtime binding canonical SHA256
ea93e76b664d865461c4e109aa99ec91150d993c3780111fe9bbb03d3ff8c882

terminal return SHA256
979cf567db33ffe0f8c2f6f362d402cb3757bfe84d3b803a07322b763a5cb331

output manifest SHA256
c3aea2bbd1e81aa79df474be364fd991121d3fdae4e0fa89f1838401311809a4

submission manifest SHA256
7df7a4a9cdfec93ba0eae2861c08f50c4134bf56f6eb750cd6fa0fdafde68212

qsub argv SHA256
5fd9e1f41808d61b01c1815c7303a3a13852fad9e7fad91c997f363ef70b7aca

submission wrapper SHA256
9ce5e7f36eb1467238edea7e712a91ad456274c127e63e54851e5d18a6576b5f

terminal qstat SHA256
9e8cc1b44830b4dc3f39bc79451b2d9770a64c1df5204c2ab928d6aa38209adb
```

Checkpoint tree identities:

```text
BASE
8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4

STRUCTURED@1000
30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137

FULL24@1000
38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c

FULL24@2000
2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5
```

If any seal, count, identity, or inventory differs, STOP scientific interpretation and report the discrepancy first.

---

# Phase 1 — Read-only mirror to Corelab

Create a dedicated Corelab analysis root. Suggested structure:

```text
<corelab_root>/stage45c_pi0100_train24_eval/
    source_sealed/
    verified_mirror/
    analysis/
    reports/
```

Requirements:

1. Keep the Hopper source artifacts untouched.
2. Copy artifacts to Corelab; do not move/delete source files.
3. Preserve timestamps where practical.
4. Generate a file inventory with relative paths, sizes, SHA256.
5. Verify copied files against source hashes.
6. Mark `source_sealed/` or equivalent mirror as read-only after verification if operationally practical.
7. Any derived analysis output must live outside the sealed mirror.

Do not modify existing JSON/JSONL/NPY files in place.

---

# Phase 2 — Reconstruct the existing Stage45C result table

Locate the production result artifacts for all 96 evaluations.

Reconstruct a canonical analysis table with one row per:

```text
checkpoint × TRAIN24 window
```

Expected row count:

```text
96
```

For each row, preserve at minimum where available:

- checkpoint identity
- window identity
- target identity
- source episode/window provenance
- raw prediction file/path identity
- raw prediction SHA256
- decoder/statistics binding identity
- gate/validity state
- trajectory metrics
- expert-comparison metrics where available

Do not regenerate model predictions.

---

# Phase 3 — Scientific analysis hierarchy

Analyze at three levels:

```text
per-window
    ↓
per-target
    ↓
whole TRAIN24 cohort
```

Do not rely only on global averages.

## Per-window
For each of the 24 matched windows, compare the four checkpoints directly.

## Per-target
Group windows deterministically by target identity.

If TRAIN24 is 4 targets × 6 windows, verify that from the artifacts rather than assuming it. If grouping differs, use the actual metadata.

## Whole-cohort
Provide descriptive summary across all 24 windows per checkpoint.

---

# Phase 4 — Metrics to examine

Use the existing metric definitions from the repository. At minimum inspect available values for:

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
- position roughness / second difference
- endpoint error where expert reference exists
- gripper RMSE where defined
- validity / SafetyGate-related state if present

Do not invent missing metrics.

Do not define an arbitrary binary task-success threshold unless a governed repository contract already defines one.

---

# Phase 5 — Primary paired causal comparison

The primary scientific comparison is:

```text
STRUCTURED@1000
      vs
FULL24@1000
```

Reason:

- approximately matched optimization-step exposure
- principal intervention is adaptation data coverage
- same TRAIN24 windows
- same-noise-across-checkpoint Stage45C design enables paired analysis

For every matched window compute paired deltas such as:

```text
Δprogress = FULL24@1000 absolute_progress_m
            - STRUCTURED@1000 absolute_progress_m

Δcosine = FULL24@1000 endpoint_target_vector_cosine
          - STRUCTURED@1000 endpoint_target_vector_cosine

Δfinal_distance = FULL24@1000 final_target_distance
                  - STRUCTURED@1000 final_target_distance
```

Be careful with sign conventions: lower target distance is better, while higher progress/cosine is generally directionally favorable.

Report:

- 24 paired deltas
- positive/negative/tied count
- median paired delta
- mean paired delta
- spread/range/IQR if useful
- per-target paired summaries
- outlier windows
- consistency across targets

Do not reduce the conclusion to a single mean.

---

# Phase 6 — Secondary exposure comparison

Secondary comparison:

```text
FULL24@1000
      vs
FULL24@2000
```

This asks whether additional exposure under the same FULL24 coverage provides further improvement.

Use the same paired methodology.

Do not conflate this with the coverage question.

---

# Phase 7 — BASE reference

Use BASE as a reference condition to quantify whether either adaptation checkpoint differs materially from the unadapted model.

Useful paired references:

```text
STRUCTURED@1000 - BASE
FULL24@1000 - BASE
FULL24@2000 - BASE
```

BASE is contextual evidence, not the primary coverage comparison.

---

# Phase 8 — Failure-mode inspection

For windows that remain negative or regress, inspect existing decoded trajectories and metrics for patterns such as:

- wrong target direction
- insufficient displacement
- overshoot
- high excursion
- unstable/rough trajectory
- apparently correct early motion but poor endpoint
- checkpoint-specific regressions
- target-specific failure concentration

If raw/decoded trajectory arrays are available, produce simple read-only plots for inspection.

Do not feed anything back into the model.

---

# Scientific interpretation boundary

The analysis may support statements about **TRAIN24 checkpoint behavior only**.

It must not automatically claim:

- unseen-target generalization
- Validation-E performance
- Holdout-F performance
- WBC success
- simulator success
- hardware success
- real-robot robustness

The current review question is narrowly:

> Did FULL24 adaptation improve the already-frozen TRAIN24 evaluation relative to STRUCTURED@1000 at matched exposure, and did 2000 steps add further improvement over 1000?

---

# Required deliverables

Create derived analysis artifacts under the Corelab analysis directory, preferably including:

```text
artifact_inventory_sha256.txt
mirror_verification.json
stage45c_96_rows.csv
stage45c_per_target_summary.csv
stage45c_whole_cohort_summary.csv
stage45c_paired_deltas.csv
stage45c_failure_windows.csv
STAGE45C_TRAIN24_SCIENTIFIC_REVIEW.md
```

File names may differ, but equivalent information must exist.

The Markdown scientific review should include:

1. artifact provenance / verification result
2. exact checkpoint identities
3. dataset/window inventory
4. per-window findings
5. per-target findings
6. whole-cohort findings
7. primary coverage comparison
8. secondary exposure comparison
9. BASE reference
10. failure modes / anomalies
11. claim boundary
12. recommendation for the next PI decision

---

# GitHub return protocol

Do **not** upload large raw prediction files or checkpoint artifacts to GitHub.

Return only compact evidence / summaries to GitHub.

When analysis is complete, post a detailed handoff to Issue #11 containing:

- Corelab analysis root path
- source Hopper root path
- mirror verification status
- exact row counts
- hashes/inventory seal for the derived analysis bundle
- concise metric tables
- primary paired comparison conclusion
- secondary paired comparison conclusion
- per-target consistency/inconsistency
- important failure windows
- exact scientific claim boundary
- recommendation: e.g. hold / proceed toward a new PI gate / require further TRAIN24 diagnosis

If convenient, commit only small text/CSV/Markdown analysis summaries on a dedicated reporting branch and open a draft PR. Do not commit model outputs or large binary artifacts.

Do not merge without separate PI authorization.

---

# STOP / authority boundary

This task authorizes only:

- read-only artifact copy
- hash/inventory verification
- read-only parsing
- deterministic derived metrics/summary computation from existing outputs
- plots/tables/reports derived from existing Stage45C artifacts

This task does **not** authorize:

- any model inference
- any H200 job
- any retraining
- any checkpoint modification
- Validation-E access
- Holdout-F access
- WBC execution
- simulator execution
- hardware execution
- PI_GATE_STATE_0101 or successor gate issuance
- PR #46 merge

If required sealed production artifacts cannot be located or verified, STOP and report the missing evidence rather than substituting a fresh run.
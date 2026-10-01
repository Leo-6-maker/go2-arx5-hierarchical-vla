## Objective

Stage45C is scientifically complete on the sealed TRAIN24 cohort. The next question is **not** to rerun the four checkpoints or immediately open Validation-E. The objective of Stage45D is to determine why broader FULL24 coverage does not improve the fixed-1000-step comparison while FULL24@2000 shows a modest recovery, and to separate optimization/exposure, data-normalization, and sampler/free-running explanations before any new training or generalization claim.

Controlling PI authorization is issued separately in Issue #11. This issue is the execution workspace for Corelab Stage45D D0 only.

## Current accepted evidence

Stage45C Track S completed with deterministic reconstruction of all 96 sealed raw predictions and exact historical CPU decode. Four 24-window aggregates reproduced the H200 sealed aggregates with max numerical difference <1e-12. Current claim boundary:

- FULL24@1000 vs STRUCTURED@1000: **does not support a coverage-improvement claim under the fixed 1000-optimizer-step budget**.
- FULL24@2000 vs FULL24@1000: **directionally supports a modest additional-exposure benefit on sealed TRAIN24 only**.
- This does **not** yet establish unseen-state generalization, convergence, or that more steps are the correct intervention.

## Stage45D D0 scientific questions

1. Was FULL24@1000 materially under-exposed / under-optimized relative to STRUCTURED@1000?
2. Is the 1000→2000 improvement broad across windows/targets, or concentrated in a small subset?
3. Did broader FULL24 data widen or distort the action/statistics distribution in a way that makes the same optimization budget harder?
4. Are remaining failures better explained by learned local-field quality or by free-running sampler/integration behavior?
5. What is the smallest next H200 diagnostic that can distinguish those hypotheses without opening Validation-E or starting new training?

## D0-A — Exposure audit

Using only already-sealed training manifests, trainer state, dataset manifests and provenance, reconstruct for STRUCTURED@1000, FULL24@1000 and FULL24@2000:

- unique training example/window count;
- effective batch size and gradient accumulation;
- optimizer steps;
- samples/examples consumed if recoverable exactly;
- estimated passes/repetitions per unique example;
- save/checkpoint schedule;
- exact scheduler horizon and warmup semantics;
- any shuffling/sampling semantics relevant to exposure.

Do not infer unsupported exact counts. Clearly separate exact recovered quantities from derived estimates.

Required output: `stage45d_exposure_audit.csv` + machine-readable JSON summary.

## D0-B — Optimization-state audit

Read-only inspect existing Stage45B/FULL24 training evidence, including trainer state/log history where available. Report at least:

- training loss trajectory around early, ~1000 and ~2000 steps;
- learning-rate trajectory and exact LR at/near the two checkpoints;
- gradient norm or related optimizer diagnostics only if they were historically recorded;
- whether the curve is still improving, flattening, unstable, or cannot be diagnosed from available evidence;
- whether checkpoint-2000 appears plausibly under-converged, without claiming convergence from loss alone.

Important: the 2000-step cosine schedule is defined on a 2000-step horizon. Do not propose `resume 2000→4000` as a clean continuation. A future fresh 4000-step run also changes the LR trajectory before step 2000 and therefore would be a new optimization protocol.

Required output: `stage45d_optimization_audit.json` and a concise plot/table if derivable from already-existing logs only.

## D0-C — Dataset / normalization / action-distribution audit

Using existing sealed Structured and FULL24 artifacts only, compare distributions relevant to learnability:

- EEF XYZ action distribution;
- rotation representation distribution;
- gripper distribution;
- StructuredAction geometry/timing latent distributions where recoverable;
- q01/q99 or equivalent normalization bounds;
- normalized magnitude and near-boundary/saturation frequency;
- per-dimension variance/spread;
- target/category balance using existing target IDs only;
- evidence of broader FULL24 data increasing heterogeneity or scale.

Do not modify statistics, regenerate historical stats, or substitute one checkpoint's statistics for another. All comparisons must preserve exact artifact identity.

Required output: `stage45d_data_stats_audit.json` plus compact CSV tables for per-dimension/per-target summaries.

## D0-D — 24-window paired failure taxonomy

Use the already reconstructed 96-row Stage45C evidence. Do not rerun model inference or decode.

For each of the 24 matched windows, compute and tabulate checkpoint-to-checkpoint behavior for at least:

- absolute/final progress;
- final target distance;
- best/minimum target distance;
- endpoint target-vector cosine;
- path length;
- roughness;
- endpoint error / expert comparison metrics where already available.

Create a deterministic **analysis taxonomy** (not a success threshold) that identifies patterns such as:

- persistent-good;
- persistent-negative;
- recovered-at-2000;
- regressed-at-2000;
- coverage-helped-at-1000;
- coverage-hurt-at-1000.

If labels overlap, define precedence explicitly. These are descriptive analysis labels only and must not be promoted to task-success labels.

Then summarize by existing `target_id` and whole cohort. Determine whether the FULL24@2000 gain is broad or target/window-concentrated.

Required outputs:

- `stage45d_window_taxonomy.csv`
- `stage45d_per_target_diagnosis.csv`
- `stage45d_failure_mode_summary.json`

## D0-E — Hypothesis ranking

Based only on D0-A through D0-D, rank support for the following hypotheses:

- H_OPT: FULL24@1000 is under-optimized / under-exposed;
- H_DATA: broader FULL24 distribution / normalization makes learning harder;
- H_SAMPLER: learned local field may be adequate but free-running source-mapped Euler rollout accumulates error;
- H_POLICY: local action field itself remains poor even near expert trajectories;
- H_HETEROGENEITY: gains/failures are concentrated in specific targets or motion regimes.

Use `SUPPORTED / PLAUSIBLE / WEAK / NOT_SUPPORTED / NOT_TESTED` with direct evidence references. Do not choose a new training recipe from this evidence alone.

## D0-F — Freeze, but do not execute, the next H200 diagnostic

Prepare a Stage45D D1 **teacher-forced / local-flow diagnostic protocol** using existing checkpoints only. It must remain non-executed in this Corelab gate.

Preferred candidate checkpoints:

- STRUCTURED@1000
- FULL24@1000
- FULL24@2000

Use the existing TRAIN24 windows and frozen action/state semantics. The protocol should compare model-predicted local flow/velocity against the oracle flow target at fixed, explicit flow times/noise, with no adaptive extension.

Required protocol decisions to freeze for PI review:

- exact checkpoint identities;
- exact window subset or full TRAIN24 set and rationale;
- exact noise construction/identity;
- exact flow timesteps;
- exact teacher-forced state construction;
- metrics: local-flow RMSE, cosine, geometry/timing/gripper decomposition, per-window/per-target summaries;
- exact H200 action-head call budget;
- raw-before-derived sealing;
- fail-closed identities and output artifact tree;
- explicit statement that target XYZ remains evaluation/supervision metadata and not model input.

The D1 protocol must distinguish these cases:

1. teacher-forced good + free-running bad → sampler/integration diagnosis;
2. FULL24@1000 teacher-forced poor but FULL24@2000 better → optimization/exposure diagnosis;
3. failures concentrated in targets/dimensions → data/normalization diagnosis;
4. teacher-forced broadly good and free-running stable → candidate to discuss Validation-E.

Required output: `STAGE45D_D1_TEACHER_FORCED_DIAGNOSTIC_PROTOCOL_DRAFT.json` plus Markdown review note. **Do not submit PBS/qsub or run GPU/model forward under D0.**

## Claim boundary and forbidden actions

This issue does not authorize:

- new model construction or forward;
- GPU/H200/PBS/qsub;
- new training or resume;
- Validation-E or Holdout-F reads;
- WBC, simulator or hardware;
- changing sampler/timing/action/state/normalization semantics;
- rewriting Stage45B/C historical evidence;
- merging PR #49.

Corelab may perform CPU/filesystem-only read-only analysis, add deterministic analysis scripts/tests if needed, and commit/push those support changes on a PI-approved Stage45D branch only if the Issue #11 gate permits it.

## Final return

Return to this issue and Issue #11 with:

```yaml
TOKEN: STAGE45D_D0_DIAGNOSTIC_REVIEW_COMPLETE
EXACT_GIT_HEAD:
EXPOSURE_AUDIT:
OPTIMIZATION_AUDIT:
DATA_STATS_AUDIT:
WINDOW_TAXONOMY:
PER_TARGET_DIAGNOSIS:
HYPOTHESIS_RANKING:
D1_PROTOCOL_DRAFT:
D1_ESTIMATED_H200_CALLS:
CLAIM_BOUNDARY:
MODEL_CONSTRUCTION: false
MODEL_FORWARD: false
GPU_H200: false
PBS_QSUB: false
TRAINING: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
WBC_EXECUTED: false
SIMULATOR_EXECUTED: false
HARDWARE_EXECUTED: false
NEXT_PI_GATE_REQUEST:
```

Then STOP for PI review.
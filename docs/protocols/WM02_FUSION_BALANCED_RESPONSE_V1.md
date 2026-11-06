# FUSION C10 local response identification

The first real four-arm group showed no old-WM gain and provided only three +X
responses and one +Y response. It cannot identify HOLD or −X behavior. This
successor records balanced physical responses before fitting a new small WM;
it does not repeat the completed PILOT2 optimizer matrix or select task passes.

`python -m scripts.wm02.run_fusion_identification --input <registered-input>`
is implemented. It requires a fresh finite FUSION registry entry/lease and an
immutable source capsule. The original owner, scope deadline, frozen C10 native
executor, 2 ms physics and safety remain authoritative. This file starts no job.

- Four new TRAIN and two new DEV physical parents; whole-parent split is fixed
  before integration. Initial root XY is uniform ±6 mm and object XY relative
  to that root uniform ±1 mm, seed 366201, original HOME_ROOT_OBJECT_XY_V3 reset.
- Each scene first receives a single-world HOLD qualification. Then four
  original actions HOLD/+X1mm/−X1mm/+Y1mm have two independent restarts each,
  one on each qualified CPU/ROS slot. Action order is shuffled with seed 366201.
  TRAIN precedes DEV. No TEST data, four-world execution or GPU physics.
- 6 qualifications +48 physical response episodes, 4,103 integrations each;
  221,562 ticks is the planned count, never an observed success count.
- This is B0 direct, CALIBRATION_INTENT, not GR00T, an expert, or an exact
  counterfactual fork. Original prehold/burn/decision3703/ramp8/H16/H40/50ms
  timing and original task threshold are unchanged. All four candidate
  forecasts are saved before installation; actual installed/solver references
  and achieved response are recorded separately.
- A pair is eligible only when both native runs complete with response evidence.
  On native failure, retain both members, all costs and planned NOT_RUN members,
  and stop this finite cohort. Native PASS with task failure remains data.
  There is no task-score retry or success-based exclusion.
- Export only 8 past atoms/current state and 16 actually observed future atoms.
  Future states are targets, never decision inputs or expert action labels.
  Only the measured decision source-age entry is valid; others are masked.

Fit new response models and static action means on TRAIN only. Action-blind
controls retain history but remove the future candidate from both the neural
features and any ARX base. Match samples, seeds, updates and numeric mode;
save actual checkpoints, restore optimizer/sampler/RNG and export a NumPy
runtime before any frozen native comparison. DEV can evaluate the predeclared
endpoint but cannot supply labels, normalization or a practical-gain threshold.
The original four-window pilot and all historical C08/C10 results retain their
separate identities. Prediction improvement does not establish control benefit.

## Matched small-model endpoint

`python -m scripts.wm02.train_fusion_response --spec <fresh-spec> --output <fresh-fit>`
is implemented for the complete 48-window export. It keeps the original frozen
ARX and fits a 32-hidden-unit tanh TCP-only residual, zero initialized at its
output. All 16 future TCP XYZ states are targets; qpos, contact and rotation are
not learned or claimed as consistent full-state predictions. Physical task
failures remain targets. Own-arm TRAIN normalization, FP32, TF32 off, AdamW
lr0.0003/decay0.0001, millimeter MSE plus0.01 residual penalty; no clipping.

Two arms ACTION_CONDITIONED/NO_FUTURE_ACTION, seeds366301/366302/366303, batch32,
2,000 updates each, same exact global-shuffle sampler. The blind arm removes
the candidate from both its feature vector and its frozen ARX base; all past
observations/references remain. Checkpoints1999/2000 include optimizer, RNG,
normalizer and full sampler state; replay the final update and compare all of
them exactly. Six restoration probe calls are reported separately from12,000
planned main updates. These are planned endpoints, not observed counts.

The primary seed366301 is fixed before DEV and never picked by DEV scores.
Report TRAIN/DEV horizons1/4/8/16 against frozen ARX, current position, constant
velocity and the same new-TRAIN mean action response. Action-minus-HOLD errors
use independent restarts with measured prefix mismatch, not exact forks. Export
all six NumPy models; require <=0.2micrometer Torch/NumPy difference. A later
native worker verifies the actual CLOSED0 training result, restored primary
checkpoint/portable identity and TRAIN static-model identity before use. Native
inference uses NumPy only and the unchanged qualified four-action compiler.

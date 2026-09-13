# PI0159 conditioning-mechanism result

Status: **COMPLETED** (`CORELAB_PI0159_CONDITIONING_MECHANISM_FOR_PI_REVIEW`).

This packet is a CPU-only reconstruction of the completed PI0159 retry5
attempt.  The original model run produced all 12 raw bundles (four
conditions × three local-flow times), then failed only in derived aggregation
with an `IndexError` caused by the batch dimension.  No model query was
repeated; the reconstruction consumed the sealed raw bundles in place.

## Bound execution

| item | value |
| --- | --- |
| branch / reconstruction HEAD | `stage45f/corelab-local-initial-state-pilot` / `857286fcfc7ef89ce9245de1ac5671fecb7fbddf` |
| original model-attempt HEAD | `df8997b2d8bf32efa918d001e8ee5064ed75e2e0` |
| N2 checkpoint tree | `09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152` |
| N2 stats file / payload | `0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb` / `508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572` |
| fixed x0 array / file | `5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd` / `83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922` |
| original raw root | `/var/cache/users/liuyu-nfs/stage45f_pi0159_conditioning_mechanism_20260913_retry5` (node-local) |
| raw admission / attempt-lock SHA | `0882a4b2e45db537e27797677d0187d253ad7ef91d396c08b30f604ca32e2d7e` / `4f4ecefbaf62bd57c531fd56d697cd6b1b5ad87e52df1eea2016a7915cc47c59` |

The source checkpoint, stats, noise, TRAIN/C0 inputs, Cosmos snapshot and
GR00T source remain SHA-bound to the PI0159 admission.  The decoder uses the
Stage45C runtime-V2 bridge with the N2 stats payload; the standardized TRAIN
StateV1 anchor is the primary cross-condition tail view.  Actual-anchor
decodes are retained as mechanics only.  Target XYZ is not a model input and
TRAIN world-frame task metrics remain unverified.

## Observed effects

| condition | active-field delta RMSE vs TRAIN | active-field delta max | tail XYZ delta RMSE |
| --- | ---: | ---: | ---: |
| `TRAIN_REFERENCE` | 0 | 0 | 0 |
| `STATE_C0_VISION_TRAIN` | 0.0168699 / 0.0176052 / 0.0225737 | 0.125 / 0.125 / 0.195313 | 0.0089753 / 0.0032802 / 0.0006546 m |
| `VISION_C0_STATE_TRAIN` | 0.1048504 / 0.1184866 / 0.1578279 | 0.433594 / 0.492188 / 0.773438 | 0.0238641 / 0.0174099 / 0.0084827 m |
| `C0_FULL` | 0.0996889 / 0.1136050 / 0.1511670 | 0.417969 / 0.474609 / 0.740234 | 0.0213695 / 0.0165112 / 0.0081870 m |

Values in each row are ordered by flow buckets `t067`, `t369`, `t783`.
Both single-modality substitutions changed the active teacher-forced local
field: `STATE_AND_VISION_REACH_LOCAL_FIELD`.  The C0 StateV1 path also shows
frozen-processor clipping in EEF dimensions `[0,2,3,4,5,6,7,8]` (8 values;
preclip range `[-2.8933713, 5.0641648]`), so the state effect is not evidence
that every raw StateV1 scalar is used without saturation.  The C0 vision path
with TRAIN StateV1 has no observed state clipping.

## Counters and boundaries

Original attempt: one model construction, four conditioning passes, twelve
action-head calls, sixteen model-side calls, four query starts, and one
post-forward CPU exception.  Reconstruction model calls: **0**.  No retry,
public/free-running sampler, training, backward, optimizer step, ICL, new
data, WBC, simulator, hardware, PBS, Validation-E, or Holdout-F access.

## Evidence files

- `pi0159_conditioning_mechanism.json` — complete machine-readable report
- `conditions.csv` — 12 condition/time rows
- `condition_differences.json` — active-field and decoded-tail differences
- `processed_state_audit.json` — state clipping/information-loss audit
- `raw_manifest.json` — SHA-bound raw bundle inventory
- `anchor_reconciliation.json` — label-only anchor precision record
- `trajectory_plot.svg` — fixed TRAIN-anchor tail overlay
- `evidence_hashes.json` — output and raw-row hashes
- `one_page_conclusion.md` — bounded conclusion and one recommendation

The only recommendation is to authorize one fixed-budget independent-expert /
visited-state local-field supervision diagnostic before any ICL or training
decision; it is not started by this packet.

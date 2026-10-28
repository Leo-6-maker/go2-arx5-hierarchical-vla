# PI0268 — actual codec conditioning

## Finding

The unchanged OLD codec is reversible but poorly scaled on actual PI0267 recovery TRAIN. Its X/Y pose changes activate near-constant rotation components. This is measured on all 24 frozen TRAIN families / 10,752 raw conditions, not inferred from a toy or fitted from EVAL. The evidence warrants a normalization-only comparison; it does not establish a unique cause of R's failures.

| Boundary (10 active representation coordinates) | abs p95 | abs p99 | fraction > 1000 | nonfinite |
| --- | --- | --- | --- | --- |
| state | 29041.533 | 119928.680 | 5.7143% | 0 |
| relative action + gripper | 18292.863 | 110095.109 | 7.8308% | 0 |

These population fractions include all seven variants, not the actual training sampling weights. Padded dimensions are excluded. The ten active coordinates are a 9-coordinate EEF representation plus gripper, not ten independent physical DoFs.

## Binding and transformations

Actual `build/set_statistics` input SHA256: `4439e24348bc43f5c93d380c9ae3c25b4057dbd8111e1e9bf9563b868c53bc73`. It matches the frozen R1 design and actual Hopper dataset statistics. Both checkpoint archival statistics have different serialized bytes but equal parsed numeric structure. A filename or formatting mismatch is not the finding.

Raw StateV1 and absolute labels are FP32. Actual stock pose conversion constructs orthonormal rotation rows, forms relative SE(3), then returns FP64 relative values. The frozen affine map is `2*(x-q01)/max(q99-q01,1e-6)-1`, without clipping. Packed state/action are FP32. Relative action EEF uses the 40×9 relative-action table; absolute-action EEF statistics are unused on this production route. Gripper retains its absolute map. State/action are padded to 132; the 122 extra coordinates remain zero and the corresponding action mask is zero.

Four actual RGB/state/language/label processor calls reproduce their consumed condition identities and raw-to-packed arrays exactly. Collator outputs: state FP32 `[1,1,132]`, action FP32 `[1,40,132]`, mask FP64 `[1,40,132]`, pixels FP32, tokens/embodiment INT64. See [PROCESSOR_BOUNDARY.json](PROCESSOR_BOUNDARY.json). No learned forward was executed. Model `prepare_input` casts floating tensors to the model dtype; the sealed native path uses BF16, while saved training full-S4 rollout tensors are FP32. Source equality between Hopper and Corelab was directly verified: `6bd66ef768f1484ac5a7889c40358a599f1862b1d554d47d357e113f2f96501c`. The S4 log hashes document its real stored dtype; this report does not conflate CPU casts with a model precision experiment.

State EEF forward gains, coordinates 0–8: `21.1145, 10.9642, 36.3641, 101.6841, 5.1001, 484316.4841, 5.1001, 101.6841, 2000000`. Relative-action feature maxima across horizons: `612.9576, 645.2610, 1293.0199, 152514.6506, 252.0907, 484316.4841, 252.0907, 156322.6795, 2000000`. State rotation coordinate8 reaches the 1e-6 floor; relative coordinate8 does so at every horizon. Constant gripper also has a large gain, but its observed constancy does not support blaming amplified gripper motion.

For example, TRAIN M1/c0/first quartile has state abs-p95 about119929 for X tilt and29042 for Y tilt; nominal and Z tilt stay around unit scale. At horizon39, X relative rotation coordinate8 is about117502. These are actual project measurements. Full task×command×variant×quartile, feature×horizon rows remain in [CODEC_TRAIN_STRATA.csv.gz](CODEC_TRAIN_STRATA.csv.gz), including benign dimensions and all failed cases. [CODEC_FEATURE_BOUNDS.csv](CODEC_FEATURE_BOUNDS.csv) records q01/q99/min/max/effective width/floor and both gains. [ABSOLUTE_ACTION_UNUSED_EEF.csv](ABSOLUTE_ACTION_UNUSED_EEF.csv) explicitly labels the unused absolute EEF route.

## Nonlinear conversion, arithmetic and sensitivity

TRAIN Gram–Schmidt first-vector and orthogonalized second-vector norms never fall below0.9999999404 for state/action, or0.9999999999999997 for relative action. Maximum observed row orthogonality errors are below3.4e-8 (FP32 inputs) and5.3e-17 (relative FP64); components lie within[-1,1] up to2.3e-16 roundoff. There is no measured near-zero TRAIN denominator to explain the scale blow-up. See [NUMERIC_SUPPLEMENT.json](NUMERIC_SUPPLEMENT.json) and [GRAM_AND_CAST.json.gz](GRAM_AND_CAST.json.gz).

| Arithmetic operation | maximum XYZ error (m) | maximum SO3 error (rad) |
| --- | --- | --- |
| action FP32 encode/decode | 6.290314707129982e-09 | 3.012645749538664e-08 |
| action BF16 CPU cast then decode | 0.00015030989245820968 | 0.00047507509000915165 |
| state BF16 CPU cast then affine inverse | 0.00038809504644289676 | 0.0006990605710952471 |

These are arithmetic roundtrips of stored numeric tensors, **not a model precision ablation**. A small inverse error does not make a 1e5 learner input or target well-conditioned.

Fixed function probes use the first metadata-ordered TRAIN item in each task/command/variant cell, exactly1e-7 and1e-6 metres/radians on each axis (336 probes). State normalized maximum changes: translation3.70e-6 /3.64e-5; world-left rotation0.203125 /2.0. Original absolute future labels are held fixed to isolate anchor-conversion sensitivity; no probes become training data or rendered observations. All values, including relative-target changes, are in [FIXED_CODEC_SENSITIVITY.csv](FIXED_CODEC_SENSITIVITY.csv). The nearly constant toy fixture is separately labelled constructed.

## Limits

Per-module neural activations are UNOBSERVED. No live normalizer, target, decoder, clipping rule or epsilon changed. Retrospective EVAL ranges are isolated in the trace tables and were not used to fit bounds. The proposed theoretical rotation bounds depend on the verified representation and do not apply to XYZ or arbitrary deltas.

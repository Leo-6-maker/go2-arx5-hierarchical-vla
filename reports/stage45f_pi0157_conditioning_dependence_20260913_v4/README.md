# PI0157 conditioning-dependence execution

Status: **COMPLETED**. The executable report classifies observed fixed-policy
conditioning sensitivity as **SUPPORTED**. Mechanistic interpretation remains
bounded: the State-C swap is visibly different at the processed input, but
the archived C0 state clips eight EEF dimensions under the frozen BASE
normalizer, so this is not evidence that every StateV1 scalar is used without
saturation.

## Bound identities

| item | value |
| --- | --- |
| branch / execution HEAD | `stage45f/corelab-local-initial-state-pilot` / `d19f9d9c2b896a3eaf7d8f40cdefc0ebf39183bf` |
| PI0157 starting HEAD | `f4fdb01e5071b530d6997ec2d0a17666cdeacddb` |
| N2 seed2223@2000 tree | `09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152` |
| N2 stats file / payload | `0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb` / `508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572` |
| fixed x0 array / file | `5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd` / `83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922` |
| host / device | `corelab-4080s` / RTX 4080 SUPER GPU0, `GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad` |
| runtime | Python 3.12.3; Torch 2.9.0+cu128; CUDA 12.8; bf16 |

The processor registration module was the SHA-bound Stage10
`artifacts/go2_arx5_modality.py` (`e61f9429...`); the dataset
`meta/modality.json` was treated as metadata, not as the registration module.
The immutable BASE processor metadata hashes were
`85c1b469...` (`processor_config.json`) and `c97b1b07...`
(`statistics.json`).

## Conditions and counts

`N1_NATIVE_a` was byte-identical to the accepted TRAIN/T input and reused its
sealed raw. New queries were `N2_NATIVE_b`, `N3_NATIVE_c`, `N4_NATIVE_d`,
`X1=(V_T,S_C)`, `X2=(V_C,S_T)`, `Z1=(128,S_T)`, and `Z2=(128,S_C)`.
The final attempt consumed 7 queries, 7 conditioning passes, 28 action-head
calls, and 35 model-side calls; one model was constructed and no exception or
retry occurred. Including the previously closed PI0156 diagnostic, the
recorded cumulative total is 12 queries, 12 conditioning, 44 head calls, and
56 model-side calls. Training, ICL, WBC, simulator, hardware execution, PBS,
Validation-E, and Holdout-F remained zero/false.

Key active-output deltas versus T (float32 raw, 190 active scalars):

| condition | max abs | RMSE | processed-state round-trip |
| --- | ---: | ---: | --- |
| `X1=(V_T,S_C)` | 0.078125 | 0.0143498 | clipping observed (8 EEF dims) |
| `X2=(V_C,S_T)` | 0.557617 | 0.111698 | no scalar loss |
| `Z1=(128,S_T)` | 0.337891 | 0.0615647 | no scalar loss |
| `Z2=(128,S_C)` | 0.311523 | 0.0557020 | clipping observed (8 EEF dims) |

All four native snapshots have the same StateV1 SHA
`6336195b...`; their native image pairs differ. Relative expert comparisons
are retained in `expert_comparison.csv`; TRAIN world-target metrics are
explicitly `UNVERIFIED`. C0 retains the accepted measured-EEF bridge and
the previously observed prefix/tail split, with final progress
`-0.0971843191 m` and best progress `+0.0797300409 m`.

## Evidence and recovery history

The raw `.npy` predictions remain node-local under
`/var/cache/users/liuyu-nfs/stage45f_pi0157_conditioning_dependence_20260913_v4`;
this repository view contains only small derived evidence. The exact report
and artifact hashes are in `evidence_hashes.json` (report SHA
`12c4a816a10a8939da203ce9bcab3822a55330b9a441d5826f30ac7ffb112d8d`). Three
pre-forward admission corrections were retained outside the scientific
count: native target metadata binding, registration-module path correction,
and BASE-processor separation from N2 weights. They entered no model
construction or forward; the final admission was the only successful model
attempt.

## One recommendation

Authorize one fixed-budget teacher-forced/local-field diagnostic that records
StateV1 and paired-image conditioning separately from decoded p0/relative
actions. Do not start training, ICL, WBC, or a new closed loop from this packet.

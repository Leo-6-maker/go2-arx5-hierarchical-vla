# PI0164 controlled-supervision pilot

Status: **BLOCKED / PARTIAL**  
Return token: `CORELAB_PI0164_CONTROLLED_SUPERVISION_PILOT_FOR_PI_REVIEW`

Authority: Issue #11 comment `5653126447` (`PI_GATE_STATE_0164`) and PR #69
comment `5653131645`.  The execution branch is
`stage45f/corelab-local-initial-state-pilot`, current HEAD
`c51cdbbe0e86ea9669775383977f13c5e4ab3619`.

## Scope and fixed identities

The requested pilot was a fresh 256-update continuation from the exact N2
seed-2223@2000 checkpoint.  E was FULL24-only; S was the same deterministic
sequence with 128 odd slots replaced by one fixed q0.  No scientific recipe,
statistics, checkpoint, noise, precision, module set, or seed was changed.

The preferred q0 passed the pre-execution qualification and was bound as the
PI0163 C1 q0 source.  It is the independent `STAGE22_SAME_SCENE_EXPERT`
chunk (`36deefb15ec152aad579ab0f1ec6f8a730446fd3eca0bccab833d794feab1c25`),
with strictly increasing timestamps, `eef_local == StateV1[:3]`, finite
`[40,10]` expert action, and target XYZ excluded from model input.  The q0
artifacts are immutable: NPZ
`e0376e138bb1610a557164d3afffa3da1d7d0446a3b28406374360bbe7c441ad`, JSON
`a4d53a962af98b074b8a9f4a88a2b238e15254db362b2be06b2134d454c67e06`.

| identity | value |
| --- | --- |
| N2 checkpoint tree | `09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152` |
| N2 stats file / payload | `0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb` / `508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572` |
| FULL24 tree | `2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789` |
| BASE tree | `8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4` |
| GR00T source | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| Cosmos revision / tree | `9ce19a195e423419c349abfc86fd07178b230561` / `1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9` |
| modality SHA256 | `e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031` |
| fixed evaluation x0 file / array | `83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922` / `5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd` |

## Admission and attempts

The node-local admission was valid on `corelab-4080s`, GPU0 RTX 4080 SUPER
(`GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad`), Python 3.12.3, Torch
2.9.0+cu128, bf16.  A separate CPU VideoDecoder preflight passed.  The
admission files record the exact identities above and the fixed E/S recipe.

| attempt | output root | admission SHA256 | result SHA256 | construction | forward | optimizer / updates | result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `/var/cache/users/liuyu-nfs/stage45f_pi0164_controlled_supervision_20260913` | `fedceb4c6913a9a859d3a2d1030f5c3613813491d49a265fcda616b773b91084` | `1b83b3b43e9622e616c7986a00b3c2ed40d07567a9f7d09f7ed07adfcc48a7e6` | 1 | 0 | 0 / 0 | BLOCKED: torchcodec native library load |
| recovery 1 | `/var/cache/users/liuyu-nfs/stage45f_pi0164_controlled_supervision_20260913_recovery1` | `2b6727baaedfd9597964a52fc0e914516c731f35ba23c7dcdcbdb80594080925` | `a32cca65674b445c880d8b9c5e5d14bff3d759c7cef8876ec5a4550e2a03a437` | 1 | 0 | 0 / 0 | BLOCKED: torchcodec native library load |
| recovery 2 | `/var/cache/users/liuyu-nfs/stage45f_pi0164_controlled_supervision_20260913_recovery2` | `8b68de43d2d47a0a97c0a11749296dc0e3bf9abc377edce3578b67dab97fef0f` | `4369318998cccaea8e4da1fdd1af8ca8dbd89a7d67b9cad653a3a4d803387144` | 1 | 1 | 0 / 0 | BLOCKED: CUDA OOM during AdamW state allocation |

The first two failures occurred before a model forward; the final attempt
passed the external FFmpeg library setup, constructed E, and reached one
real training forward/backward.  It then failed while AdamW initialized its
state: GPU0 had only 216.69 MiB free while the process needed another 18 MiB.
The two mechanical recoveries are consumed.  No checkpoint at 64/128/192/256
was written, S never started, and no retry or step extension was performed.

## Metrics and boundary

E/S physical metrics, old-task degradation, and the requested final FR
evaluation are **UNAVAILABLE** because neither trained arm produced a valid
checkpoint.  The accepted pre-pilot N2 seed-2223@2000 TRAIN24 reference remains
the only comparable baseline; its sealed 24-window aggregate reports mean
endpoint error `0.11383744038825593 m`, mean XYZ position RMSE
`0.054939310451958455 m`, mean final target distance `0.1138374403882559 m`,
mean absolute progress `0.018374058249132918 m`, mean path length
`1.87489861653302 m`, mean roughness `0.04440989141082435 m`, and 14 positive /
10 negative progress windows.  Aggregate SHA256 is
`0000a907f66fa28266a7b9180499bbd4cbc9b09483899e110247a35a55d87d74`.
These values are reused sealed evidence, not a new PI0164 model result.

Actual PI0164 totals: model constructions **3**, training forwards **1**,
optimizer steps **0**, updates **0**, E **attempted but blocked**, S **not
started**, FR **0**.  Validation-E, Holdout-F, ICL, DAgger, WBC, simulator,
hardware control, PBS/H200/Vanda execution, and data generation were all
zero/false.  No user or other GPU process was killed or reprioritized; the
failed process exited and GPU0 returned to the pre-existing shared occupancy.

## Conclusion and one recommendation

This pilot is a genuine resource-blocked partial, not a scientific negative:
the frozen recipe reached one real forward but no optimizer update, so no
claim about Q supervision, E/S learning, or task improvement is supported.

Recommend one new PI gate for the identical frozen E/S pilot on a clean,
sufficiently free 32-GB-or-larger device (or an explicitly authorized H200),
without changing precision, batch, modules, loss, data, seed, or step budget.
Do not automatically retry under the current gate.

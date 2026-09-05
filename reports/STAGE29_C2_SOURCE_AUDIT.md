# Stage29 C2 pinned-upstream sampler source audit

Date: 2026-09-05 (Asia/Singapore)

```text
PI gate: PI_GATE_STATE_0009 (Issue #11 comment 5548308223)
Stage28 accepted base: 116ada218e02c819f4a2d4bedac7fc3f88e09c58
GR00T source: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
Result: PASS_UNAMBIGUOUS_OFFICIAL_SAMPLER
Model forward: false
PBS submission: false
```

## Canonical call path

The public local inference API is `BasePolicy.get_action`, implemented by
`Gr00tPolicy._get_action`.  It processes one observation, collates it, and
calls `Gr00tN1d7.get_action`.  The model prepares the backbone/action inputs,
runs the backbone, and delegates to the sole N1.7 sampler implementation,
`Gr00tN1d7ActionHead.get_action` / `get_action_with_features`.

The action-head method is therefore not a competing scientific sampler.  It
is the sampler used by the public policy path.  Deployment-specific TRT
helpers are not selected because Stage29 is bound to the stock PyTorch model
and checkpoint path used by Stage20S/22/23/26/28.

## Frozen sampler semantics

- Algorithm: explicit Euler integration, `actions += dt * pred_velocity`.
- Integration count: `num_inference_timesteps = 4`; `dt = 0.25`.
- Time grid: continuous `0, 0.25, 0.5, 0.75`; bucket IDs
  `0, 250, 500, 750` with 1000 buckets.
- Initial state: exactly one `torch.randn([B,40,132])` on the vision-language
  feature device and dtype.  With the accepted runtime this is CUDA bfloat16.
- RNG ownership: the global PyTorch generator for the selected CUDA device;
  the upstream API has no external-noise or generator argument.
- RTC/inpainting: inactive because inference conditioning contains no
  `action` key.  No RTC options are supplied.
- Guidance/sampler clipping/postprocessing: none in the sampler loop.
- Output: normalized universal `[B,40,132]` action tensor.

The processor then takes the modality-declared active H40/10D prefix,
unnormalizes with the frozen TINY1 statistics, and converts relative actions
to absolute physical actions using the last state-history frame as the
reference.  Min/max unnormalization clips normalized values to `[-1,1]` in
the pinned processor utility; this is existing official decode semantics, not
a Stage29 intervention.  The returned physical groups are EEF 9D plus the
gripper scalar, combined as `[B,40,10]`.

## Paired-noise instrumentation

The stock API cannot accept an external initial tensor.  Stage29 therefore
uses the narrow instrumentation authorized by PI_GATE_STATE_0009:

1. restore the same Python, NumPy, CPU-Torch, and CUDA-Torch seed before each
   BASE/H2 member of a pair;
2. temporarily wrap only Python-level `torch.randn` during the official
   `model.get_action` call;
3. require exactly one call and require its result to be `[1,40,132]` on the
   model device/dtype;
4. return that tensor unchanged to the upstream sampler while recording its
   raw bytes, float32 canonical hash, RNG-state hashes, shape, dtype, and
   device;
5. fail closed unless the BASE and H2 raw initial-noise hashes match for the
   same seed.

This instrumentation neither supplies replacement noise nor changes the
integration loop.

## Source identities

| Path | SHA256 | Relevant lines at pinned SHA |
|---|---|---|
| `gr00t/policy/policy.py` | `b53326aee0c26ea34201d70510ad92608a96929d7f39039f578c3d865446de23` | 80-105 |
| `gr00t/policy/gr00t_policy.py` | `ad1183a3ce1ebed6cace4a9cb8c5a123e823a0760ebf46f3694639a5ad1521ec` | 380-432 |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | `6bd66ef768f1484ac5a7889c40358a599f1862b1d554d47d357e113f2f96501c` | 325-475, 603-614 |
| `gr00t/configs/model/gr00t_n1d7.py` | `379ff708496a73b25b12f4656850284a16fde0e205828e2a0c0be0af61463744` | 70-110 |
| `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` | `77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f` | 382-443 |
| `gr00t/data/state_action/state_action_processor.py` | `0151dd0bb6cb727fd401bf78121efc532e7184f2575e6ef25294085f629ac145` | 421-526 |
| `gr00t/data/utils.py` | `fa765b240295d9884ab62326c8a21465ae21e6f10133a4b68ac321e8101e4ab7` | 116-152 |
| `gr00t/eval/open_loop_eval.py` | `e15706fac4cd8caff3229a1538eb0288b487919b28c0e26ecf30c32d0f61245f` | 169-194, 300-311 |

No second semantically plausible stock PyTorch sampler path was found.


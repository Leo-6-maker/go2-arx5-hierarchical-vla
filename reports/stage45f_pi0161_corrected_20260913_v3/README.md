# PI0161 — corrected PI0160 evidence

**Return token:** `CORELAB_PI0161_FRAME_SYNC_CLOSURE_FOR_PI_REVIEW`  
**Status:** `PARTIAL` — CPU correction complete; new B/C execution is blocked by the historical accounting gate.

Authority is Issue #11 comment `5652160514` / PR #69 comment `5652164343`. The
source experiment directories remain immutable. This report is a new, derived
CPU-only packet produced from the sealed PI0160 v7 expert and v3 model raw.

## Corrected expert world-target metrics

The upstream `_step_rows` `desired_pos` and `achieved_pos` are already
world-frame. The old wrapper populated `*_pos_world` by adding the same anchor a
second time. The table below uses the original fields and the measured q0 EEF;
tracking residuals therefore remain unchanged while the target metrics and
achieved path are corrected.

| condition | initial error (m) | final error (m) | progress (m) | normalized | reference path (m) | achieved path (m) | tracking RMS (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 nominal | 0.212094730 | 0.012816297 | +0.199278433 | 0.939573 | 1.042417056 | 1.152901868 | 0.044868181 |
| C2 X−5 cm | 0.240724855 | 0.012864182 | +0.227860674 | 0.946561 | 1.087167348 | 1.171520430 | 0.045612580 |
| C3 Y+5 cm | 0.174480120 | 0.012869219 | +0.161610901 | 0.926242 | 1.027812816 | 1.151575269 | 0.044857983 |
| C4 Y−5 cm | 0.254017791 | 0.012816927 | +0.241200864 | 0.949543 | 1.065995339 | 1.177186265 | 0.044755649 |

All four complete episodes have positive world-target progress and final error
about 1.27–1.29 cm. This corrects the old report's approximately 0.719 m
endpoint; it does not alter the reference trajectory or claim learned-policy
success.

## Corrected local-field baseline

For every one of the 15 sealed teacher rows, the field quantities were rebuilt
from the saved arrays:

```text
v_target = x1 - x0
model_field_error = v_hat - v_target
zero_field_error = -(x1 - x0)
model_reconstruction_error = x1_hat - x1
zero_reconstruction_error = x_t - x1
```

The flow identity `x1_hat-x1=(1-t)(v_hat-v_target)` holds with maximum
absolute residual below `2.5e-7` in every row. Mean corrected zero-field
active RMSE is **1.068703485** and mean XYZ RMSE is **1.104887782** at each of
`t=0.067762418`, `0.369669436`, and `0.783771974`; the former `x_t-x1` values
are not used as a field baseline. The full 15-row table and raw hashes are in
the node-local CSV below.

## Clipping, codec, and C1 persistence

- The old `abs(normalized) >= 1` values are reported as boundary-saturation
  counts: C0/C1/C2/C3/C4 = **7/7/8/5/9**. Pre-clip values were not sealed, so
  true clipping counts are `UNKNOWN`; BASE-state clipping is not inferred.
- All five sealed label codec round-trip flags remain true, with strict
  rotation error recorded as zero. No physical action or source raw was
  modified.
- C1's `failure.json` records `rows_saved=49`, `snapshots_saved=2`,
  `wbc_forward_count=51`, and `reference_update_count=1`; there is **no**
  `rows.json`. The independently persisted evidence is only `q0/q1` plus
  their body/wrist files and `failure.json`. The 49 is an in-memory marker, not
  49 independently persisted rows; the missing telemetry was not reconstructed.

## Accounting gate and execution boundary

Known native simulator constructions from v3/v4/v6/v7 are **14**, already over
PI0160's cumulative cap of 7. v1 and v5 have no persisted admission, so they
remain unresolved rather than being silently counted as zero. Known native WBC
ticks total 2,797 and known reference updates total 35. The PI0161 new
simulator/model phase was therefore not admitted.

```yaml
new_simulator_instances: 0
new_native_wbc_ticks: 0
new_gr00t_forward: 0
new_model_side_calls: 0
training: 0
backward: 0
optimizer_step: 0
gpu_model_execution: 0
PBS: false
H200: false
hardware: false
Validation-E: false
Holdout-F: false
ICL: false
DAgger: false
merge: false
```

The single next recommendation is to resolve the v1/v5 provenance and
over-cap accounting with PI, then authorize the exact two-condition post-tick
synchronized expert capture. The old unsynchronized q0s should not be promoted
to training supervision.

## Evidence and reproducibility

Reference source roots (node-local, immutable):

```text
expert: /var/cache/users/liuyu-nfs/stage45f_pi0160_matched_reset_expert_20260913_v7
model:  /var/cache/users/liuyu-nfs/stage45f_pi0160_matched_reset_model_20260913_v3
```

The correction source is `scripts/stage45f_pi0161_correct_pi0160.py` at
commit `c690c56`; it is CPU-only and has no top-level Torch/GR00T/IsaacGym
import. Its self-test covers the oracle flow endpoint, zero-field sign, and
single-anchor rule.

Node-local derived packet:

```text
/var/cache/users/liuyu-nfs/stage45f_pi0161_corrected_20260913_v3
```

| file | SHA256 |
|---|---|
| `expert_corrected_metrics.csv` | `e9df8379bdec5fc3e606f95d153a6f5f078a37a75ac943c890f8d8aa182f4259` |
| `model_corrected_field_metrics.csv` | `86ba2ba8160566ca96fcb7d50562a72a1df9cd4e2efb5a2263b1e85fc875b94f` |
| `c1_persistence_audit.json` | `af6c14ba05501352dba6c5091311122fb18be52d09f53671f76c56840e389177a` |
| `clipping_codec_audit.json` | `a11e27a8e9631e66ad077508d931c8dad98360bbe69c48bcc2ffe6ea0813c974` |
| `attempt_ledger.json` | `ca7e4beb76d406092ad7452431cd275bc710a270a31d8c5d1434134a51ff7732` |
| `static_audit.json` | `73b53733e8223d518ace2d92e5ac67f2a0350a3a45dc58949c8fa6c9d384d1ed` |
| `pi0160_corrected_result.json` | `5e236d0f0cfa98739e6a6a7c2c20591d4305344e186885a254094ee2b61c079c` |
| `evidence_hashes.json` | `421969fea7a0494804a23b16d10eade90d4d6abcfe6561092d8b57646b6b0169` |

Source bundle checks in the machine-readable result reproduce the accepted
PI0160 hashes, including expert result
`56e6035efe486cda720b286a63f5879b3d85f14766253a232f4842144e37b6a8`, model
result `fd79a35299f94cf1fa5bbe25b88938585752b7bd7cd210859ad951040ad64257`,
and the 20 individual raw prediction hashes.

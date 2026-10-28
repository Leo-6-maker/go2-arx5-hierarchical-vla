# Native encoding and source-contract correction evidence

The released protocol §2 and machine limits declare XYZ <= 1e-5 m, SO3 <= 1e-4 rad and gripper <= 1e-5, with no forbidden clipping or label manipulation. They do not declare zero float64 diagnostic preclip count. Original PI0285 `codec_check.py` records that count (line 79) and checks physical errors (line 80). PI0286 `recovery.py` line 186 added `count==0`. Calling this added condition a frozen original gate in prior reports was incorrect. The original 15/32 receipt is retained as the outcome of that implementation, not a source-protocol scientific verdict.

## Twelve actual saved late windows

Only the previously recorded 12 late anchor-0 windows were encoded/decoded. Every actual encoded packet is float32, shape 40×132, with active channels 0..9 in [-1,1] and zero out-of-range values. Rotation channels 3 and 7 are exactly 1.0. All 12 actual active packets are elementwise identical to the float64 audit preclip intermediate directly cast to float32.

| Actual operation | Dtype | Actual clip effect |
|---|---|---|
| Native `apply_action`, eef_9d | float64 | Clips 80 intermediate values/window; maximum change 9.605649609056854e-11 |
| Native `apply_action`, gripper | float32 | Zero changed values |
| `Codec.encode_actions` final packet | float32 | Zero out-of-range values; identical to direct preclip→FP32 cast |
| Native inverse `unnormalize_values_minmax`, eef/gripper input | float32 | Zero changed values |
| `unapply_action` output | float64 | Original denormalization and relative→absolute conversion |
| `Codec.decode_actions` return | float32 | Original cast |

Actual roundtrip maxima across these 12 windows: XYZ 2.499000650857397e-8 m; SO3 4.719113568238058e-8 rad; gripper 0. All finite/rotation-valid and within the released physical thresholds. Native clipping does occur in the float64 forward intermediate; its final FP32 packet effect is zero for these windows. This report distinguishes those two observations and does not redefine forbidden clipping.

## Exact native source

- `PI0280/official/gr00t/data/state_action/state_action_processor.py`: `apply_action` starts line 333, existing `clip_outliers` branch lines 414–415; `unapply_action` starts line 421.
- `PI0280/official/gr00t/data/utils.py`: `normalize_values_minmax` starts line 72; inverse starts line 116 and calls `np.clip(normalized_values,-1,1)` at line 151.
- `PI0283/code/codec.py`: `encode_actions` packs into a float32 40×132 array; `decode_actions` casts incoming packets and returned commands to float32. Full exact function bodies, lines and file hashes are in the JSON.

The audit computed from target/state float32 enters relative rotation and min/max processing in float64. Frozen effective statistics are float64. Channels 3/7 exceed +1 by 8.499867476530198e-13 / 9.605649609056854e-11 before the original native forward clip; both would independently round to exactly FP32 1.0.

All selected input .npy hashes match sealed rows; inputs, statistics/config/embodiment, bank and code-seal receipt have unchanged before/after hashes. Native source files remain hash-identical. No raw episode-slot arrays opened. Runtime 43.920690 s, hidden CUDA/four CPU threads; 12 encode/decode calls, zero model/teacher/physics calls. No labels, admission status or statistics were written by this probe.

The owner has authorized correcting the implementation-added gate under the original release contract, preserving original receipts. Versioned bank and fixed-sampler results must be reported separately; these 12-window facts alone are not automatic training admission.

Original JSON: `ENCODED_LATE12_NATIVE_CLIP_EVIDENCE.json`, SHA256 `d19f3c2647af2e46705baf64a9b262e1f02cce445e4ac9df41f029a254adbf51`.

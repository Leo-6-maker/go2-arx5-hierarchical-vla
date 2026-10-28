# Bounded preclip explanation: 15 sealed failed windows

**C1 remains CORRECTION_NOT_ADMITTED, 15/32 valid.** The exact 15 failed H40 windows were examined using saved PI0286/data action/state .npy files and the frozen native `action_audit` formula. All 15 recomputed counts match their sealed CODEC records. This addendum resolves the previously unobserved preclip channel/excess details; it does not change admission.

## Twelve 12 s windows

Layouts 285108, 285109, 285111, 285112, 285115, 285116, 285117, 285119, 285120, 285121, 285122 and 285123 have the same two affected channels at suffix anchor 0. Every window has 40 upper-bound occurrences in each channel, totaling 80. All other channels have zero preclip out-of-range values.

| Native channel (zero based) | Relative value | Frozen TRAIN range | Native excess above TRAIN maximum | Preclip normalized value | Excess above +1 |
|---|---:|---|---:|---:|---:|
| 3: relative_rot6d_0 | 1.0 | [0.7125991777307872, 0.9999999999998779] | 1.2212453270876722e-13 | 1.00000000000085 | 8.499867476530198e-13 |
| 7: relative_rot6d_4 | 1.0 | [0.712172269890337, 0.9999999999861762] | 1.3823830968817674e-11 | 1.0000000000960565 | 9.605649609056854e-11 |

These are measured small upper-bound excesses of exact relative identity entries over the frozen TRAIN maxima. Their stored physical roundtrip errors pass XYZ/SO3/gripper thresholds. The frozen zero-preclip rule still rejects these branches; neither a tolerance nor a changed maximum is applied.

## 285118 at 3 s: three windows

Only channel 8 (`relative_rot6d_5`) exceeds its TRAIN range [-0.0782696334891012, 0.08174743139803593]. All excursions are below the lower bound. The three counts sum to 98 occurrences across overlapping H40 windows; this is not a count of distinct controls.

| Suffix anchor / original tick | Count | Native range | Maximum native excess below TRAIN minimum | Preclip normalized range | Maximum excess below -1 |
|---|---:|---|---:|---|---:|
| 0 / 80 | 32 | [-0.08895765305836713, -0.009135780137939194] | 0.010688019569265933 | [-1.13358599692857, -0.13591899214095282] | 0.13358599692856998 |
| 1 / 81 | 33 | [-0.08564411870479349, -0.015181521722833042] | 0.007374485215692297 | [-1.092171234622928, -0.2114827026634276] | 0.09217123462292798 |
| 2 / 82 | 33 | [-0.07862896204304895, -0.017608905729577125] | 0.00035932855394775476 | [-1.0044911279206528, -0.2418217669183077] | 0.00449112792065276 |

The original saved roundtrip audit also fails SO3 in these three windows (maximum 0.010654909013566369 rad, threshold 1e-4). No new roundtrip decode was run here. XYZ and absolute gripper have zero preclip excursions in all 15 examined windows.

## Formula, types and provenance

The actual native relative transform `_convert_to_relative_action(..., EEF, XYZ_ROT6D)` and actual `normalize_values_minmax` are reused. The normalizer applies `2*(value-min)/(max-min)-1` to nonconstant channels and zero to ranges considered constant by its existing `np.isclose` mask. No clipping is introduced in this explanatory intermediate.

Saved target/state are float32; converted relative values and the concatenated preclip audit intermediate are float64. The full JSON records all ten channel ranges, bounds, counts, normalized/native excesses and sealed physical error fields for each window, plus native function source hashes.

All selected .npy file hashes match the sealed row identities. Input files, frozen statistics/config/embodiment, bank and healthy code seal have identical before/after hashes. The frozen statistics SHA is `c3ec9c6b797bb735e3ac38d611b409262de04bbb965cb68b7426df749ba7fb81`.

Runtime: 57.091377 s, hidden CUDA and four CPU threads. Model, teacher, physics and full-codec calls: zero. Raw episode slot files opened: zero. Only 15 saved failed H40 windows were examined. Gates, labels, statistics and admission remain unchanged; learning causality remains UNPROVEN.

Detailed receipt: `PRECLIP_FAILED15_DETAILS.json`, SHA256 `525c5fbd2babe9312e837e4a0ef0d9ef8c63e265219595640094e792b0258330`. The isolated tool is stored on Core under `PI0286/correction/tools/preclip_failed15_cpu.py`; no healthy sealed runtime file was edited.

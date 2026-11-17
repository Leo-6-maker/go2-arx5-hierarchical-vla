# E1 saved-input qualification and boundary comparison

Actual offline result SHA256 `ff309443e89b50c36bd5917c6457dcc43ae03b037e73adfb41b2170c79fdcad9`, source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`. All three ordinary-RGB16 package restorations were bit-exact. No new optimization, native request or dynamics integration was performed by this offline stage or the subsequent CPU-only report.

| Arm | TRAIN96 H40 | CN38 H40 | ROLL96 H40 | Original rejected3 repaired | CLOSE omissions / late events |
|---|---:|---:|---:|---:|---:|
| B0 | 96/96 | 180/180 | 93/96 | 0/3 | 0 / 0 |
| B1 | 96/96 | 180/180 | 93/96 | 1/3 | 0 / 0 |
| B2 | 96/96 | 180/180 | 96/96 | 3/3 | 0 / 0 |

CLOSE denominators are20 annotated TRAIN events and36 CN38 events per arm, each checked against the unchanged100ms standard. ROLL96 has no matched expert truth or terminal-intent labels; its EEF deviation is only from the saved B0 prediction. Those states are diagnostics, not correction labels.

| ROLL96 measure | B0 | B1 | B2 |
|---|---:|---:|---:|
| Start acceleration max(m/s²) | 0.294609 | 0.277715 | 0.237435 |
| Tail acceleration max(m/s²) | 0.254252 | 0.243251 | 0.203571 |
| Interior acceleration max(m/s²) | 0.170047 | 0.144640 | 0.202460 |
| Start margin minimum to original.25 | -0.044609 | -0.027715 | 0.012565 |
| Tail margin minimum to original.25 | -0.004252 | 0.006749 | 0.046429 |
| Nominal H40 progress median(m) | 0.014155 | 0.014333 | 0.014333 |

B2 removed the three original saved-state envelope failures and passed all96 rolling inputs. B1 repaired one of the original three but still failed two and introduced a different failed rolling input, leaving93/96. This supports a boundary-loss effect on the fixed saved inputs; it does not establish a closed-loop task gain. B0/B1 remain in all eight screen conditions despite failing the offline promotion gate.

B2 meets the fixed offline gates; it is not yet a promotion candidate until the actual eight-condition native screen meets its task, family, STOP and technical gates. Full per-bank head/tail/interior margins, seam speed, EEF errors, H40 progress and named failed inputs are in [the machine-readable report](E1_OFFLINE_ACTUAL.json). No gate or threshold was changed.

# PI0266 actual diagnostics

selected/deployment: NONE. Teacher-panel GR00T queries / optimizer updates: 0 / 0.
Complete task outcomes and valid local-window counts are separate. No failed scenario is replaced. Historical PI0265 remains PARTIAL/FAIL.

| Case | Task success | Safety | Valid/proposed windows | Final XYZ m | Final SO3 rad |
|---|---|---|---|---|---|
| D-pi0265-family-000-M1-c0-CLOCK_DRIVEN | True | False | 0/0 | 0.0058709397379444585 | 0.06764315706615434 |
| D-pi0265-family-000-M1-c0-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.04410556698756398 | 0.1601888257925623 |
| D-pi0265-family-000-M1-c1-CLOCK_DRIVEN | True | False | 0/0 | 0.005872699260813973 | 0.07701974341205976 |
| D-pi0265-family-000-M1-c1-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.1370906127625287 | 0.18817083010242225 |
| D-pi0265-family-000-M3-c0-CLOCK_DRIVEN | False | False | 0/0 | 0.00441655974440466 | 0.06042309933647724 |
| D-pi0265-family-000-M3-c0-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.07607540031881664 | 0.14763839266118506 |
| D-pi0265-family-000-M3-c1-CLOCK_DRIVEN | False | False | 0/0 | 0.0034365770253515873 | 0.10283819525512623 |
| D-pi0265-family-000-M3-c1-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.043225406516214086 | 0.14803490784240408 |
| D-pi0265-family-001-M1-c0-CLOCK_DRIVEN | True | False | 0/0 | 0.0039319757201868085 | 0.07355067888998883 |
| D-pi0265-family-001-M1-c0-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.044252002836337546 | 0.14284717654725826 |
| D-pi0265-family-001-M1-c1-CLOCK_DRIVEN | True | False | 0/0 | 0.006424825156507909 | 0.06567900717229654 |
| D-pi0265-family-001-M1-c1-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.09921586384161071 | 0.20742417170452399 |
| D-pi0265-family-001-M3-c0-CLOCK_DRIVEN | False | False | 0/0 | 0.003340406644922888 | 0.056870931928076476 |
| D-pi0265-family-001-M3-c0-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.06634329353253113 | 0.17454647079983004 |
| D-pi0265-family-001-M3-c1-CLOCK_DRIVEN | False | False | 0/0 | 0.004362272515017228 | 0.07288676596218609 |
| D-pi0265-family-001-M3-c1-OBSERVABLE_CORRECTION_V1 | False | False | 100/100 | 0.05224189538483018 | 0.1263777009140747 |

The CPU replay uses a distinct full-URDF traversal on actual root/DOFs and the shared prospectively fixed V2 scorer. Tracking is actual versus installed reference on the same native tick, through first success/failure. V1 is retained for each first20s prefix. This teacher report grants no model deployment or full-task controller qualification.


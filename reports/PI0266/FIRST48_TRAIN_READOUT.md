# PI0266 fixed first48 actual TRAIN readout

This is an interim result for the first48/96 prospectively fixed TRAIN episodes. Overall IN_PROGRESS; training admission false; selected/deployment NONE.

3909 proposed windows;3900 CPU-admitted local windows;9 IK_RESIDUAL rejections. All48 parent complete tasks failed; no parent safety failures. Full-task failure does not remove a valid earlier local window. No case replacement or scientific retry.

Independent same-owner full-URDF FK maximum XYZ error8.321128198348852e-9m and SO3 error3.892780634048788e-8rad. Stored RGB/State/proposal hashes, codec reconstruction and exact observable semantic alias checks passed. This does not prove general Markov observability.

| Task | Command | Q0 | Q1 | Q2 | Q3 | Terminal hold |
|---|---|---:|---:|---:|---:|---:|
| M1 | 0 | 228 | 36 | 50 | 817 | 69 |
| M1 | 1 | 1157 | 2 | 6 | 30 | 5 |
| M3 | 0 | 299 | 1 | 0 | 0 | 0 |
| M3 | 1 | 657 | 543 | 0 | 0 | 0 |

Each active task/command quartile requires>=8 distinct valid source windows. M1 command1 Q1/Q2 is2/6; both M3 commands have Q2/Q3 zero, and M3 command0 Q1 is1. These are intermediate shortages, not the final96-case verdict. Twelve TRAIN families are represented; volume/family support does not substitute for phase coverage.

The healthy original collection process continues the remaining fixed48 cases. Final WINDOW_ADMISSION runs only after all96 registered cases finish. No GR00T queries, gradients or optimizer updates were added by this CPU replay.

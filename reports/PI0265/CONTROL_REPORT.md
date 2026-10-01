# PI0265 — actual controller comparison

**Overall: PARTIAL; PI0265_CONTROLLER_ENVELOPE_BLOCKED.**
Historical PI0264 COMPLETE/VALID_FAIL is unchanged. Formal selected/deployment: NONE.

G0 source/CPU checks passed. Checkpoint training-time installed PyTorch3D remains UNKNOWN. LEGACY and ROW6D alone were compared; native quaternion-vector feature, weights, gains, dynamics and gate were preserved.

C0 is diagnostic, not general-controller admission. Source reset22301 and original reference attitude were retained. Actual direct/provider reset/RNG and all installed canonical-table arrays match exactly for each version.

| Case | Task success | Safety fail | Final XYZ m | Final SO3 rad | Tracking RMS / p95 m | Tracking SO3 RMS rad | Arc bins |
|---|---|---|---|---|---|---|---|
| C0-LEGACY-DIRECT_NATIVE | False | False | 0.019929 | 0.374797 | 0.023901 / 0.026097 | 0.945713 | 0 |
| C0-LEGACY-REFERENCE_PROVIDER | False | False | 0.019929 | 0.374797 | 0.023901 / 0.026097 | 0.945713 | 0 |
| C0-ROW6D-DIRECT_NATIVE | True | False | 0.025309 | 0.059171 | 0.022099 / 0.021759 | 0.143822 | 0 |
| C0-ROW6D-REFERENCE_PROVIDER | True | False | 0.025309 | 0.059171 | 0.022099 / 0.021759 | 0.143822 | 0 |
| pi0265-family-000-M1-c0-LEGACY | False | False | 0.006122 | 0.673639 | 0.009101 / 0.018842 | 0.597136 | 0 |
| pi0265-family-000-M1-c1-LEGACY | False | False | 0.007827 | 0.679919 | 0.008959 / 0.013669 | 0.590948 | 0 |
| pi0265-family-000-M3-c0-LEGACY | False | False | 0.012327 | 0.422361 | 0.020808 / 0.034266 | 0.379214 | 0 |
| pi0265-family-000-M3-c1-LEGACY | False | False | 0.018642 | 0.632341 | 0.015425 / 0.023068 | 0.442302 | 0 |
| pi0265-family-001-M1-c0-LEGACY | False | True | 0.198585 | 0.212240 | 0.033660 / 0.033660 | 0.135006 | 0 |
| pi0265-family-001-M1-c1-LEGACY | False | True | 0.170066 | 0.267008 | 0.033660 / 0.033660 | 0.135006 | 0 |
| pi0265-family-001-M3-c0-LEGACY | False | True | 0.150314 | 0.135006 | 0.033660 / 0.033660 | 0.135006 | 0 |
| pi0265-family-001-M3-c1-LEGACY | False | True | 0.162970 | 0.135006 | 0.033660 / 0.033660 | 0.135006 | 0 |
| pi0265-family-000-M1-c0-ROW6D | True | False | 0.003141 | 0.042304 | 0.007653 / 0.018287 | 0.074151 | 0 |
| pi0265-family-000-M1-c1-ROW6D | True | False | 0.004033 | 0.055188 | 0.005954 / 0.009484 | 0.091863 | 0 |
| pi0265-family-000-M3-c0-ROW6D | False | False | 0.003349 | 0.031809 | 0.013520 / 0.022939 | 0.157133 | 0 |
| pi0265-family-000-M3-c1-ROW6D | False | False | 0.004185 | 0.040607 | 0.011591 / 0.018969 | 0.107123 | 12 |
| pi0265-family-001-M1-c0-ROW6D | False | True | 0.198585 | 0.212240 | 0.033660 / 0.033660 | 0.135006 | 0 |
| pi0265-family-001-M1-c1-ROW6D | False | True | 0.170066 | 0.267008 | 0.033660 / 0.033660 | 0.135006 | 0 |
| pi0265-family-001-M3-c0-ROW6D | False | True | 0.150314 | 0.135006 | 0.033660 / 0.033660 | 0.135006 | 0 |
| pi0265-family-001-M3-c1-ROW6D | False | True | 0.162970 | 0.135006 | 0.033660 / 0.033660 | 0.135006 | 0 |

## Registered qualification

{"LEGACY": {"M1": 0, "M3": 0, "all_successful_tracking_pass": true, "successes": 0}, "ROW6D": {"M1": 2, "M3": 0, "all_successful_tracking_pass": true, "successes": 2}}

Every fixed independent case and every failed start is retained. No score-based retry, fourth-DEV rescue, new pose/seed, gains or settling search. A source-bound contact-classification bug initially treated normal foot/ground support as forbidden self/obstacle contact. Seven zero-actor starts were marked ENGINEERING_INVALID with raw receipts unchanged and budget retained; this is not a scientific-failure reclassification.

Raw budget: {"actor_calls": 18000, "control_ticks": 9027, "simulate_calls": 36108, "simulated_seconds": 180.53999596461654, "starts": 27}.

Requested, installed and achieved are distinct world-coordinate columns. Tracking uses the same native tick, not differences of minimum distances. Native termination state comes from pre-autoreset receipts when available; unavailable terminal camera views are not reconstructed.

## Actual safety and task failures

| Case | Stop reason | Actual forbidden collision pairs | Hard-bound joints |
|---|---|---|---|
| pi0265-family-001-M1-c0-LEGACY | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |
| pi0265-family-001-M1-c1-LEGACY | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |
| pi0265-family-001-M3-c0-LEGACY | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |
| pi0265-family-001-M3-c1-LEGACY | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |
| pi0265-family-001-M1-c0-ROW6D | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |
| pi0265-family-001-M1-c1-ROW6D | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |
| pi0265-family-001-M3-c0-ROW6D | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |
| pi0265-family-001-M3-c1-ROW6D | PRE_STEP_SAFETY | [["base", "x5_link8"]] | [] |

Safety stops and complete but unsuccessful tracking episodes are separate outcomes. A reset-dependent forbidden collision shared by commands/representations is not evidence that either6D packing caused it. C0 or a single successful M1 case cannot replace7/8 complete successes,3/4 per behavior and every successful episode's tracking requirements.

Neither version completed the registered dynamic qualification. G2 feedback collection, R/D training and learned-policy closed loop were not started. This is a controller-envelope stop and overall PARTIAL, not an executed feedback-training pilot or evidence of a unique controller root cause.

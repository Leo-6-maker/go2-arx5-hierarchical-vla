# FULL10000 fixed retention observation

PUBLIC revision7bafb99f58f3ed9dea66e85fff8c8b79a7c9785d. 128 first-valid cells /512 native head evaluations. Independent CPU replay PASS, reportSHA256b2707c86e16969e7bbb09a19174a13365cdd90f488c240f379ce47cc45849b16. Valid terminal quality FAIL retained. Full candidate set is unchanged; original CALIB/unique selection still required.

|Group|REJECT|median XYZ m|p95 XYZ m|median SO3 rad|Quality|
|---|---:|---:|---:|---:|---|
|all64/B0|0|0.054376|0.181114|0.036091|FAIL|
|all64/B1|0|0.053657|0.182935|0.033579|FAIL|
|old16/B0|0|0.052339|0.089704|0.039608|FAIL|
|old16/B1|0|0.051982|0.090716|0.036719|FAIL|

Original four thresholds unchanged: zeroREJECT, medianXYZ<=.020m, p95XYZ<=.035m, medianSO3<=.050rad. All four groups fail XYZ median/p95; SO3 passes. Old16 is a subset, not extra queries. No result-driven retries, candidate substitutions or training changes.

Complete original retention scope: reused G0/B064 + new FULL1000/B064 + FULL2500/5000/7500/10000 B0/B1 each128 =640 logical cells /576 new queries /2304 new head evaluations. FULL1000 remains observation only, never a formal candidate or resume source.

All four formal complete Trainer points and actual10000-update integrity have closed. The authorized four fixed L/Nlive/Q side queries now run once on Corelab, PID3996717, followed by separate CPU replay. No physical episode. CALIB remains unread until those four side reports close. selected_checkpoint=NONE; only original selector may freeze the formal winner. No DEV/F, DAgger, hardware, WBC/gate changes, merge, mail or monitor.
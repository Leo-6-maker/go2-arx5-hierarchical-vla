# PI0294 R2 — bounded actual result

**PARTIAL / VALID_SAFETY_FAIL_NO_D1_STOP.** Human adoption of the complete B–E package is valid. Its D1 gate did not pass, so Teacher regression, P1R, M2048/M4096, DEV selection and confirmation remain **NOT_RUN**. Execution completion and scientific qualification are separate. Selected/deployment remain NONE. No GPU/PBS/neural query or optimizer update was used.

## Actual finite results

| Fixed G1 case | Tick / time | Result |
|---|---:|---|
| A: original arm + original finger |1081 /2.162s|JOINT_LIMIT; every saved high-rate row exactly equals the original failure, max difference0|
| B: original arm + all open |1118 /2.236s|ILLEGAL_CONTACT_OR_PENETRATION; right finger still below effective upper at stop|
| C: initial TCP hold + original finger |1200 /2.4s|Bounded complete, no native safety flag; no task-success claim|
| D: initial TCP hold + all open |1200 /2.4s|Bounded complete, no native safety flag; no task-success claim|

G0 used1081 saved pre/post pairs,48 transmitted prefix points and44 started points. Motor arithmetic error0. Fresh zero-integration force reconstruction exposes original bias/passive/constraint coverage gaps: warmstart and original internal forces were not recorded. Recomputed constraint values are not original measured forces. Moving-arm versus held-arm effects are supported; inertia and contact are not individually isolated.

The frozen **R2_VEL_C01** motor intervention adds only dissipative velocity feedback,9.0931054738Ns/m, with the original400N/m P term,8N cap,0.04m/s reference slew,3Ns/m passive damping,2ms physics and safety bounds. Its gain uses the measured mass diagonal bound0.091402kg; this is not an exact coupled effective inertia or full-system stability proof. Geometry, gains of the arm controller, teacher, model, data and statistics remain unchanged.

Five fixed4s stress cases passed: open/closed hold, open→closed, closed→open and high-end command changes. The sixth, original failure prefix with explicit planned end hold, stopped at **tick1078 /2.156s**, before reaching the end hold: right finger **0.0450229822345m**, **22.9822345µm over the effective0.045m upper bound**. No motor saturation was recorded in that case. The candidate is permanently retired for this run; no damping retune or quality retry.

The other predeclared mechanism, bias-only feedforward, was not launched. In the saved final0.16s the right bias is positive0.00890–0.01831N; adding it would reduce the existing inward motor force at these saved states. Its scale is smaller than reconstructed constraint/coupling terms, and it does not compensate arm-acceleration coupling. This does not prove every bias-only trajectory must fail; it provides insufficient affirmative mechanism evidence for a second repair candidate. The authorized range supplies no third mechanism, geometric change or threshold relaxation. This is the protocol's bounded stop, not an authentication/compile/storage stop.

## Evidence and engineering

Original A was reproduced from t=0, never resumed from an incomplete near-failure state. All first-valid results and raw NPZ remain private on the registered CoreLab run. `EVIDENCE_MANIFEST.json` records hashes and byte counts. Six independent CPU saved-array checks verify actual clock, row counts, extrema, motor cap and the terminal high-rate flag. The driver CPU check validates power sign/gain/cap only; it does not certify physics.

Preparation corrected Windows archive line endings before sealing and a saved-message schema reader before zero-step mechanics succeeded. The first G0 parsing failure remains recorded. Neither correction retried a valid scientific failure. All18 original numerical source hashes match; old capsules and old ledgers are preserved. The existing engine diff is one optional motor hook plus version reporting. No WBC, hardware, new training data, broad cleanup or deletion occurred.

New science consumption: **10 physical starts,15677 integrations,0 RGB,0 neural queries,0 new formal updates,0 PBS/GPU**. Science-worker CPU11.139670125s; saved-array audit CPU appears in the resource receipt. Metadata/SSH/source-staging overhead was not fully instrumented and is explicitly UNOBSERVED, not reported as zero. The old parent512 formal+16 engineering updates and old R1 failure accounting remain separate.

No new model or public checkpoint was produced. Existing public M512 READY `c698f54d0fc422ebfdd1e2ce1fc96552b6930c52` and the original complete private Trainer remain protected. New raw/video/WBC publication was not authorized and did not occur. No new Teacher/P1R video exists because their prerequisite failed; short diagnostics did not render RGB.

## Next boundary

Do not rerun R2_VEL_C01, replay P1 for a better outcome or resume training around D1. A next proposal should explicitly choose one evidence-supported intervention or explicitly decouple training from the executor gate; this run authorizes neither change. No executable resume command is supplied because the dependent scientific gate is closed. Review the actual `STRESS_RESULTS.json`, `FORCE_BALANCE_AND_TIMING.json` and `DIAGNOSTIC_MATRIX.json` before a new decision.

Authority receipt: `1df3e8cbae0ebd24143868483d74e6b03a6c48e2`; actual source `43ced701d73f1df73389711f03b5b5ed68c90867`. Publication and shell exit0 are not D1 PASS. Physical stop, tested candidate outcome and unobserved model learning are reported separately.

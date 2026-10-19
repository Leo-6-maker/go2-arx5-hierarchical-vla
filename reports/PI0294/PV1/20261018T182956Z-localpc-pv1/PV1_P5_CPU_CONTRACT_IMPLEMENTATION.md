# PV1 P5 CPU contract implementation

Fixed implementation commit: `4342c9a57ba3dc64b53cad6b3483267849d946aa`

## Result

The waiting-time engineering priority is complete at CPU-contract level. The repository now has a native `P5_FINAL` path, exact LOCKBOX80 registration after a unique DEV-qualified candidate freeze, final-resource accounting, complete-result acceptance, and a sanitized shared-policy package contract.

This is not a LOCKBOX result and does not authorize opening final conditions. No model, training, inference, PBS submission, physics execution, checkpoint deletion, WBC change, S1 change, score change, or normalizer change was performed by this implementation.

## Fail-closed boundaries

- The current B0-complete-before-training gate remains unchanged.
- P5 cannot build its plan until exactly one candidate passes the existing DEV and B0-retention gates and freezes weights, processor, runtime, precision, statistics, action schema, and selection evidence.
- The same checkpoint serves T1 and T2. Teacher takeover and evaluation-time updates remain zero.
- All 80 conditions remain in the denominator. A valid task failure or S1 rejection cannot be retried. A condition receives at most one replacement and only after a separately settled engineering diagnosis.
- Final grades remain `PV1_VALIDATED`, `PV1_BETA`, or `BEST_RESEARCH_CANDIDATE_NOT_QUALIFIED`; omitting the fixed second seed caps the maximum grade at BETA.
- The offline entry defaults to schema validation only and contains no robot or WBC transport.

## Verification

Focused P5 and neighboring regressions: 49 passed, 0 failed. The broader PI0294 run passed 878 tests; the only local omissions were pre-existing tests requiring the locally absent optional packages `torch` and `safetensors`. GitHub CI installs both and is the final complete-environment check.

At 2026-10-19 21:28 SGT, a read-only scheduler check showed original job `675967.hopper-m-02` running with its unchanged identity and `1xH200, 12 CPU, 128 GB, 08:00:00` request. Scheduler state alone is not a scientific receipt; the existing owner must still publish native qualification and actual B0 outcomes.

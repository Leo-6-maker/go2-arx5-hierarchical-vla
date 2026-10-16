# Q2 Offline Recovery Oracle Contract

Implementation: `src/go2arx5_vla/flowdagger/oracle.py`.
Receipt: ORACLE_FIXTURE_RECEIPT.json. Eight tests actually ran.

Request contains observation identity, actual tick, taskT1/T2, measuredEEF10,
fixed source ABI and explicit evidence kind. Privileged object truth and
executed history are separate arguments; no student feature interface accepts
them. A request is either SYNTHETIC_CPU_FIXTURE or SAVED_ACTUAL_STATE, never a
new live rollout entry. Only one complete H40 proposal is returned per call.

Reply contains validity, complete expertH40 or null, active mask or null,
phase/reason, same observation/tick, validity0..975future ticks, oracleversion
and original safety proof reference. Candidate hashes bind the exact state and
unaltered action. A missing gate/S1 certificate returns INVALID, with proposed
trajectory retained as audit-only; it cannot enter the optimizer as a label.
ShortH8, stale-state, invalidgripper, unknownproof and unbound advice are not
padded or silently repaired.

Actual reuse: copied original T1 Teacher.plan on a synthetic saved-state
fixture, no Engine/MuJoCo/observe rollout. The copied Teacher is not mutated.
The proposal has40 reference points, but without original S1 it remains
INVALID_FOR_TRAINING, not recovery PASS. Two mock providers test safe/invalid
replies, restricted/unrecoverable states and missing horizons.

T2's original Teacher derives geometry/history from an engine, and arbitrary
student-state phase recovery is not supplied by its standard reset constructor.
We do not initialize an engine to manufacture that proof. Real T1/T2 saved
student states, valid recovery phase/history and unchanged gate/S1 certificates
are the exact remaining scientific gap. No new physical clone, Teacher
rollout, numerical search or recovery trial has run.

The interface can wrap a future qualified provider, but a mock callback's
certificate is only fixture evidence. A real collector must verify actual
hash-bound safety originals rather than trust a caller-provided PASS string.
Source standard-start S1 qualification remains separate from off-policy
recoverability and from fullbody WBC/target Teacher qualification.

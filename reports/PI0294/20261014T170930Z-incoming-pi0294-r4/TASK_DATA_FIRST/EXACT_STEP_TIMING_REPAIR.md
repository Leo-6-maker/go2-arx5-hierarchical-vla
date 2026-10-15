# Actual-step timing interface repair

The V2 fixed W0/T2 dispatch reached native step2574 (5.148s), with one actual
observation and one submitted reference. The planner accepted zero task references:
its first packet was rejected by the unchanged25ms source-clock bound, and the
new gripper adapter rejected the same stale transaction. No native joint/contact
safety stop occurred. This is an incomplete pre-install engineering attempt, not
a successful episode or an object-task quality verdict. Its raw costs remain.

The free-running simulator advanced during Python Teacher/reference construction.
Refreshing the source timestamp or enlarging its tolerance would conceal the
problem. The replacement is an isolated actual-step pause adapter:

- Integrate original2ms physics up to an explicit quota, then pause the original
  simulation flags at that exact step. Stop the inner catch-up loop too.
- Serialize the existing leg inference against the simulator's existing recursive
  mutex. During a quota pause, publish actual state but do not advance the FSM,
  motion counter, leg history or gait. Record those counters and a live ACK.
- Suspend only the owned external solver/controller while preparing a query;
  resume them for reference installation while physics remains paused. Receive
  the original reference's matching accept/installed ACK before resuming physics.
- Keep the same actual state/source timestamp, original25ms age bound,0.4s TTL,
  H40 and20Hz observations. Waiting is explicitly non-real-time.
- Reuse the unchanged V2 plant/gripper working stroke and original observer,
  controller/solver/policy objects, weights and gains. Vendor trees stay untouched.

CPU source/build/contract receipts are required before dispatch. They are not
physical qualification. The first replaced W0/T2 transaction is eligible only
under an exact source/result predicate proving zero installed task references and
no native safety stop; valid quality/safety failures remain non-repeatable.
The V1 W0/T1 safety failure remains in the original four-card pilot denominator.
All runs spend the existing cumulative ledger and adopted window.

No new hold test, WBC tuning, model query, new data recipe or budget is introduced.
Actual results follow in later source-bound receipts; this file is the engineering
description, not a PASS claim.

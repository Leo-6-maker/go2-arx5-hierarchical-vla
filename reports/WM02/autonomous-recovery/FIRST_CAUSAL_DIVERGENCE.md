# First saved causal divergence · A0

The first saved difference is **step4 / 8ms**, in qpos, qvel and ctrl. This is an observation about original failed runs, not a new target qualification. Both original initial states match. No new target constructor, Policy query or integration was executed.

The authoritative projection is `startup-causality-02`, source `c5a3a67fa3a5b5cfafe93c341b16f0081c0661c6`; inputs and per-tick results are in `T1_T2_0_27_TIMELINE.json`. Full poses/torques remain private.

| Evidence | T1 | T2 |
| --- | --- | --- |
| Control calls before first integration | 435 | 15 |
| First nonzero leg ctrl in saved integration packet | step10 / 20ms | step4 / 8ms |
| Initially repeated leg ctrl | steps10–27 in this window | steps4–23 |
| First original limit / STOP | None in saved5.4s; task was not successful | step24 / 48ms, FL calf23.357mrad beyond limit |
| Later original integrations after first STOP | NOT_TESTED, no STOP observed | 3, steps25–27 |

Conditional arithmetic uses **actual XML actuator/sensor order FR/FL/RR/RL**, mapped to named MuJoCo joints; generalized-state body order is FL/FR/RL/RR. Under fixed first recorded q-target and zero target leg velocity, the first reconstructed torque matches the first applied torque within `6.59e-11` (T1) / `1.02e-9` (T2). The fixed-command prediction differs by85.690N·m at T2 step23, then5.517N·m at24. This rejects the joint-order mistake and demonstrates changing measured-state feedback under that hypothesis. Later actual target/velocity versions and sensor-cache timing were not recorded, so it **does not prove that actual PD stopped or uniquely identify a mutex**.

An earlier offline projection used body-order indices and falsely produced an initial mismatch. That private run01 is preserved and excluded. The new immutable run02 fixes only the projection, not native controller behavior.

| Hypothesis | Distinguishing prediction in a newly admitted trace | Current status |
| --- | --- | --- |
| Policy inference blocks PD | PD callback gaps overlap inference begin/end while physical steps continue | Source-compatible; old sparse logs cannot establish actual overlap |
| Position and velocity queues consume different versions | Distinct publish/consume versions at the same PD call before the first divergent ctrl | UNKNOWN, versions absent in original recordings |
| Policy observation/history versions differ | Different causal input/history hashes before different published outputs | UNKNOWN, original inputs/history absent |
| Initialization mode or warm-up changes first publication | Same initial physical state but different running/FSM transition and wall-to-physics schedule | Different pre-integration call counts observed; exact first Policy publication UNKNOWN |
| PD and integration clocks differ | Actual PD timestamps/feedback generation lag while physical step timestamps advance | Repeated ctrl supports testing; sparse PD output is not a callback census |
| Dynamics/contact cause the difference | Identical applied ctrl and state preceding an earlier state divergence | Cannot explain the observed simultaneous ctrl divergence by itself; intra-step callback timing/contact still unresolved |

The existing R1/stop-latch/trace candidate is still only compiled and loader verified. The selected two outer preload symbol chains were actually checked again (both exit0), frozen23 and candidate-build20 inputs match. Trace ring overflow, real control rate, STOP behavior and2s/5s standing remain **NOT_TESTED**. See `TARGET_SCOPE_REQUEST.json`; it requests a new owner-bound3600-tick maximum and grants nothing.

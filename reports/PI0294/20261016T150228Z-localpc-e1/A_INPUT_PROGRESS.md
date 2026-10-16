# E1 Actual TRAIN Attribution and Fixed Inputs

Authority: c851cc791e1fbb3efb293f7075f0e771518cf5c0. This is an intermediate actual-evidence receipt, not E1 completion or model competence.

All four original E0 TRAIN episodes were deterministically reconstructed from the original initial state and actual ACKed commands. Their recorded high-rate time, qpos, qvel, ctrl, applied wrench and native safety fields match exactly (maximum absolute difference 0). Original XML, initial qpos/qvel, control targets and IK history were checked. Twelve full integration/history snapshots were saved. E0 global process RNG was not recorded; this limitation remains explicit. The frozen deterministic D0 engine does not consume that RNG. No qpos-only approximate restoration is claimed.

Three original TRAIN rejections were reproduced in shadow only:

| Condition | Shadow Offset | Original Illegal Pair | Penetration |
| --- | --- | --- | --- |
| F0/000 T1 at 1.2 s | 127 ticks / 0.254 s | table / right_pad | up to 30.98 um |
| F1/000 T1 at 1.2 s | 156 ticks / 0.312 s | table / right_pad | 8.30 um |
| F1/000 T2 at 12.4 s | 69 ticks / 0.138 s | table / left_pad | 22.01 um |

All triggered the unchanged ARM_TABLE_OR_FLOOR rule. They were not required to exceed the separate 4 mm penetration stop. In all three states, the original same-state expert and XYZ-only expert substitution passed this prefix; SO3-only and gripper-only substitutions did not. These fifteen finite diagnostic hypotheses were never installed into a live trajectory. They support investigating position components, not a unique causal claim, a safe target-bias correction or a gate change.

Hopper actually verified eight successful original v3.0 TRAIN demonstrations and sealed 96 on-expert probes, separate from the twelve saved student observations. Original numeric/RGB hashes, committed absolute EEF10 labels, valid H40 windows, persistent-success receipts and frozen statistics were checked. Original data were not copied, rewritten or refit. The exact historical microbatch 2, accumulation 16, effective batch 32 and fused AdamW precision recipe was recovered and resolved before GPU use.

The complete CPU/native loader check, actual Oracle continuations, real M1 updates and autonomous TRAIN4 results have separate receipts; this document does not substitute for them. No new DEV query, WBC execution/change, v3.1 data, fabricated BASE labels or FlowDAgger inversion is included. E0 results remain closed and unchanged; selected/deployment remain NONE.

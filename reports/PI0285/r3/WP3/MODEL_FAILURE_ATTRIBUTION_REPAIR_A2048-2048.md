# PI0285 R3 WP3 DOSE endpoint: actual TRAIN8 attribution

Status: COMPLETE saved-array CPU attribution of REPAIR_A2048-2048 TRAIN8. This report records 7/8 TRAIN outcomes; it is not a generalization, qualification or causal claim. DEV was neither read nor awaited. Learning causality remains **UNPROVEN**. No model, gradient, optimizer, GPU query, render, physics, teacher or PBS calls were added.

The original TRAIN panel and independent CPU PASS receipt SHA256 bind all eight completed 32s episodes. All **640** saved query input/output hashes were checked. Actual base/wrist RGB and StateV1 equal the pre-command recorded observations exactly; decoded first8 equals recorded commands exactly. Independent TCP/opening reconstruction error is at most 5.96e-8 and gripper actuator polarity error 0. Observation/command control i is physics sample 5*i; achieved state after that same control is sample 5*(i+1). Task times exclude the 20 neutral controls. Intended→installed→same-control-achieved XYZ/SO3 errors and first residual onset are retained per episode. The 1mm/10mrad installation and 1cm/100mrad tracking flags are descriptive thresholds, not quality-gate changes.

Actual gate counts: **637 ACCEPT, 3 WARN, 0 REJECT**. All three warnings are ANGULAR_SPEED_WARN at query 0/task time 0.0s, using TEST_ONLY threshold 2.0rad/s. WARN is preserved as WARN. Full per-query decisions and warning issues remain in JSON.

| Layout | Warning observed angular speed (rad/s) | Actual TRAIN result |
|---:|---:|---|
| 285102 | 2.0790862435 | SUCCESS |
| 285104 | 2.0898752379 | SUCCESS |
| 285107 | 2.0381964057 | SUCCESS |

Seven episodes achieve sustained grasp, sustained lift and horizontal transfer, then pass all 41 required hold samples. Only 285106 fails. There are 81 recorded IK holds, all on 285106, beginning 22.30s. At every hold, installed arm target exactly equals current pre-command achieved joint qpos; installed FK matches pre-command TCP within 1e-5, consistent with declared native IK-None fallback. No runtime input, clock, first8 indexing, polarity or fallback mismatch was found.

| Layout | Result/first missing requirement | Grasp onset/qualification/first drop (s) | First zone (s) | First task hold (s) | Required hold | IK count |
|---:|---|---|---:|---:|---:|---:|
| 285100 | SUCCESS_CONTROL | 2.81/3.11/14.27 | 11.32 | 15.24 | 41/41 | 0 |
| 285101 | SUCCESS_CONTROL | 2.11/2.41/7.42 | 5.14 | 8.94 | 41/41 | 0 |
| 285102 | SUCCESS_CONTROL | 2.21/2.51/7.17 | 6.06 | 8.43 | 41/41 | 0 |
| 285103 | SUCCESS_CONTROL | 3.19/3.49/10.47 | 8.08 | 11.53 | 41/41 | 0 |
| 285104 | SUCCESS_CONTROL | 2.25/2.55/10.52 | 9.13 | 11.32 | 41/41 | 0 |
| 285105 | SUCCESS_CONTROL | 2.35/2.65/5.75 | 11.52 | 14.04 | 41/41 | 0 |
| 285106 | NO_SUSTAINED_GRASP | UNOBSERVED/UNOBSERVED/UNOBSERVED | UNOBSERVED | UNOBSERVED | 0/41 | 81 |
| 285107 | SUCCESS_CONTROL | 3.92/4.22/5.62 | 13.67 | 16.29 | 41/41 | 0 |

The failed 285106 has no sustained grasp or lift. Horizontal displacement first becomes true at 3.19s; that is not evidence of a grasped transfer. It never reaches the full-object blue-zone or full task-hold predicate. All41 required 28..30s samples fail zone alone: table/static/not_grasped/TCP_clear/green_unmoved each pass 41/41. At 28/29/30s the cube is static and ungrasped near [0.032557,+0.629401,0.020000]m while zone center is [0.079723,−0.103667]m. Commands are mostly open-gripper and those three controls have no IK hold. JSON preserves every bad hold sample, the complete predicate-transition sequence, exact late command/installation/achievement/object snapshots and event-local contact forces. This is a late unmet-placement condition; no prior placement success or task-loss timestamp is invented.

No full task-hold loss after first attainment is observed: seven successes retain their 41 samples; the failed episode never attains full task hold. First grasp/lift drops can reflect release or recoverable contact interruptions and are not automatically task failure. For example 285107 briefly exits the zone at 13.70s yet later attains task hold at 16.29s and succeeds. All three warnings occur on successful episodes; the failure has zero warnings. IK begins much later than the failed episode's ungrasped horizontal movement and is absent at the three late snapshots. These observations counter a unique warning/IK causal explanation; they do not establish a different unique learning cause.

Unknowns remain: why the learned action sequence misses sustained grasp/placement on 285106; whether representation, history or contact changes would help; held-out DEV/confirmation performance; population generalization. This TRAIN-only audit does not authorize a branch, replacement condition, selection or threshold change and is not a dependency for owner qualification decisions.

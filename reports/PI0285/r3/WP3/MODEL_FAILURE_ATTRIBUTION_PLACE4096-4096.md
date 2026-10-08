# PI0285 R3 WP3 PLACE4096: actual TRAIN8 attribution

Status: COMPLETE saved-array CPU attribution, not new qualification or selection. 4/8 success. Learning causality remains **UNPROVEN**. No model, gradient, optimizer, GPU query, render, physics, teacher or PBS calls were added. Prior teacher and 1024/2048 reports and active DOSE work are unchanged.

The source and independent CPU receipt SHA256 bind all eight completed 32s episodes. All 640 saved model first8 prefixes equal recorded commands exactly; query StateV1 equals the actual pre-command recorder state exactly. TCP/opening reconstruction and gripper actuator polarity pass. Existing independent CPU receipt also binds RGB/input and contact scoring. Intended→installed→after-same-control-achieved XYZ/SO3 errors, installation residual onset and tracking residual onset are reported per episode. Residual thresholds 1mm/10mrad and 1cm/100mrad are descriptive, not quality gates. No runtime input, first8 indexing or command copy mismatch was found.

Aggregate gate counts are **639 ACCEPT, 1 WARN, 0 REJECT**. The sole warning is layout 285107 query 0 at task time 0.0s: ANGULAR_SPEED_WARN, observed 2.5017649227rad/s versus TEST_ONLY warning threshold 2.0rad/s. WARN is retained as WARN and is not a rejection. No quality threshold changed. This corrects the earlier worker instruction's all-ACCEPT assumption.

Six episodes have sustained grasp and sustained lift; five achieve grasp+lift+horizontal transfer. IK holds total 860 across five episodes. For every hold, installed arm target equals CURRENT pre-command achieved qpos exactly; installed FK agrees with pre-command TCP within 1e-5, as declared by pinned ManiSkill IK-None fallback. IK holds also appear in two successful episodes, so they do not uniquely explain failure.

| Layout | Result/first missing requirement | Grasp predicate onset/loss (s) | First zone (s) | First task hold (s) | IK first/count | Required hold |
|---:|---|---|---:|---:|---|---:|
| 285100 | NO_SUSTAINED_GRASP | UNOBSERVED/UNOBSERVED | UNOBSERVED | UNOBSERVED | UNOBSERVED/0 | 0/41 |
| 285101 | NO_HORIZONTAL_TRANSFER | 1.91/3.22 | UNOBSERVED | UNOBSERVED | 3.70/412 | 0/41 |
| 285102 | SUCCESS_CONTROL | 2.30/2.82 | 14.94 | 16.27 | 3.10/92 | 41/41 |
| 285103 | SUCCESS_CONTROL | 2.98/12.82 | 5.61 | 13.86 | UNOBSERVED/0 | 41/41 |
| 285104 | SUCCESS_CONTROL | 2.25/2.82 | 15.79 | 22.60 | 4.85/39 | 41/41 |
| 285105 | SUCCESS_CONTROL | 2.35/13.92 | 11.89 | 14.54 | UNOBSERVED/0 | 41/41 |
| 285106 | NO_SUSTAINED_GRASP | UNOBSERVED/UNOBSERVED | UNOBSERVED | UNOBSERVED | 25.95/105 | 0/41 |
| 285107 | NEVER_ENTERED_FULL_OBJECT_ZONE | 7.12/12.82 | UNOBSERVED | UNOBSERVED | 19.60/212 | 0/41 |

The table reports the first raw grasp-predicate onset. Sustained qualification occurs after the original continuous interval; both times are preserved in JSON. For285101 these are1.91s and2.21s respectively.

285100 never achieves sustained grasp/lift/transfer or placement, with zero IK holds.285101 qualifies grasp at 2.21s and lift at 2.72s but loses grasp 3.22s before its first IK hold 3.70s; horizontal transfer and placement never qualify.285106 never achieves sustained grasp/lift or placement; its first IK hold is much later 25.95s and green_unmoved fails all 41 required hold samples.

285107 retains the late placement failure: qualified grasp begins 7.12s and first drops 12.82s; lift first drops 8.37s. Its first IK hold is 19.60s. The red cube never enters the full-object blue-zone predicate and never attains the full task-hold predicate. At every required sample 28..30s, zone/table/static/not_grasped/TCP_clear fail (each 0/41), while green_unmoved passes 41/41. The JSON retains all 41 failing hold samples, all aggregate predicate transitions, the original query 0 warning, and exact 28/29/30s command/installed/achieved/object/gripper snapshots. This is an observed late placement/hold failure; no fictitious earlier placement success or loss time is assigned.

Success controls 285102/103/104/105 pass all 41 hold samples. Early contact or lift losses can be recovered in successful episodes; first contact loss is therefore not automatically task failure. Temporal ordering of contact, warning and IK events is preserved, but no unique learning or physical cause is claimed. The unchanged scorer and closed raw arrays remain evidence truth; the report does not authorize a new branch or selection.

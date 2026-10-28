# PI0285 R3 WP3: closed model episode attribution

Status: COMPLETE descriptive saved-array audit of PLACE1024/2048 TRAIN8, not a new qualification. Learning causality remains **UNPROVEN**. Existing teacher reports are unchanged. Model/render/teacher/physics/optimizer/PBS calls added: **0**.

All 16 episodes bind their original panel and independent CPU receipt, RESULT, physics, commands, installation, observation and query hashes. All 1,280 saved model query prefixes were checked: decoded first8 equals recorded commands exactly; query StateV1 equals the pre-command observation exactly. StateV1 independently reconstructs recorded TCP/opening to float32 precision. Gripper command polarity is correct. Model phase is recorded -1 throughout, consistent with no teacher phase supplied online. No runtime input, copy, first8 shift or actuator mismatch was found.

Times below are task-relative (one second neutral excluded). Observation/command i is physics sample 5*i; same-control achieved state is 5*(i+1). JSON retains all 1,280 input/output identities, event-local forces, contact predicates, preceding/following queries, command/measured opening, IK events and raw/install/achieved errors. Contact proof is the unchanged native two-finger force threshold >=0.5N and directed angle <=85 degrees, independently equal to recorded grasp. Descriptive installation thresholds 1mm/10mrad and tracking thresholds 1cm/100mrad locate residuals; they are not scoring exceptions or quality gates.

- PLACE4096-1024: 0/8 success; sustained grasp 3/8, sustained lift 3/8, grasp+lift+transfer 3/8. IK holds 1874 across 7/8 episodes; failed episodes with no IK holds 1; gate rejects 0. First missing requirement counts: {"NEVER_ENTERED_FULL_OBJECT_ZONE": 1, "NO_SUSTAINED_GRASP": 5, "PLACEMENT_ENTERED_BUT_REQUIRED_HOLD_NOT_COMPLETE": 2}.
- PLACE4096-2048: 2/8 success; sustained grasp 6/8, sustained lift 4/8, grasp+lift+transfer 4/8. IK holds 685 across 3/8 episodes; failed episodes with no IK holds 3; gate rejects 0. First missing requirement counts: {"NEVER_ENTERED_FULL_OBJECT_ZONE": 1, "NO_SUSTAINED_GRASP": 2, "NO_SUSTAINED_LIFT": 2, "PLACEMENT_ENTERED_BUT_REQUIRED_HOLD_NOT_COMPLETE": 1, "SUCCESS_CONTROL": 2}.

| Point | Layout | First missing requirement | First qualified grasp loss (s) | First zone exit (s) | First IK hold (s) | IK holds | Required hold |
|---|---:|---|---:|---:|---:|---:|---:|
| 1024 | 285100 | NEVER_ENTERED_FULL_OBJECT_ZONE | 10.17 | UNOBSERVED | UNOBSERVED | 0 | 0/41 |
| 1024 | 285101 | NO_SUSTAINED_GRASP | UNOBSERVED | UNOBSERVED | 9.60 | 380 | 0/41 |
| 1024 | 285102 | PLACEMENT_ENTERED_BUT_REQUIRED_HOLD_NOT_COMPLETE | UNOBSERVED | 8.04 | 26.00 | 93 | 0/41 |
| 1024 | 285103 | PLACEMENT_ENTERED_BUT_REQUIRED_HOLD_NOT_COMPLETE | UNOBSERVED | 10.47 | 22.40 | 100 | 0/41 |
| 1024 | 285104 | NO_SUSTAINED_GRASP | UNOBSERVED | UNOBSERVED | 17.85 | 233 | 0/41 |
| 1024 | 285105 | NO_SUSTAINED_GRASP | UNOBSERVED | UNOBSERVED | 12.80 | 283 | 0/41 |
| 1024 | 285106 | NO_SUSTAINED_GRASP | UNOBSERVED | UNOBSERVED | 10.40 | 371 | 0/41 |
| 1024 | 285107 | NO_SUSTAINED_GRASP | UNOBSERVED | UNOBSERVED | 9.60 | 414 | 0/41 |
| 2048 | 285100 | NO_SUSTAINED_GRASP | UNOBSERVED | 17.24 | UNOBSERVED | 0 | 0/41 |
| 2048 | 285101 | NO_SUSTAINED_LIFT | 3.22 | UNOBSERVED | 14.40 | 255 | 0/41 |
| 2048 | 285102 | SUCCESS_CONTROL | 8.77 | UNOBSERVED | UNOBSERVED | 0 | 41/41 |
| 2048 | 285103 | SUCCESS_CONTROL | 6.22 | UNOBSERVED | UNOBSERVED | 0 | 41/41 |
| 2048 | 285104 | NO_SUSTAINED_GRASP | UNOBSERVED | UNOBSERVED | 11.80 | 348 | 0/41 |
| 2048 | 285105 | PLACEMENT_ENTERED_BUT_REQUIRED_HOLD_NOT_COMPLETE | 31.85 | UNOBSERVED | 15.20 | 82 | 0/41 |
| 2048 | 285106 | NO_SUSTAINED_LIFT | 22.05 | UNOBSERVED | UNOBSERVED | 0 | 0/41 |
| 2048 | 285107 | NEVER_ENTERED_FULL_OBJECT_ZONE | 22.82 | UNOBSERVED | UNOBSERVED | 0 | 0/41 |

Grasp loss is a contact-predicate transition and can be intentional release: the two 2048 successes lose grasp at 8.77s/6.22s, retain the placement zone and first satisfy task hold at 10.04s/6.83s, then pass all 41 required hold samples. Each failed episode never attains the full task-hold predicate; a first task-hold loss is therefore unobserved rather than assigned an invented failure time.

1024 layout285100 achieves grasp/lift/transfer with no IK hold but never enters the full object zone. Layouts285102/103 briefly enter and leave the zone at 8.04s/10.47s, while their first IK holds are much later at 26.00s/22.40s; neither releases/withdraws for the required hold. 2048 layout285100 transfers the object without sustained grasp and briefly enters the zone at 15.93s, exits 17.24s, with no IK hold. These are distinct observed failure paths.

Several contact drops precede later IK holds; some failed episodes have no IK holds. Large intended-to-installed errors (up to 13.79cm at 1024,6.60cm at 2048) co-occur with IK fallback, whose pinned fallback installs the CURRENT pre-command achieved joint qpos (_start_qpos), not the previous installed target. Across all 2,559 IK holds the installed arm target equals pre-command qpos exactly and target FK agrees with pre-command TCP within 1e-5; source and array hashes bind this declared behavior. This explains the immediate installation discrepancy at those controls, but does not prove that IK caused the earlier grasp/placement failure or identify a unique learning cause. Zero gate rejects does not imply successful contact or placement. Event-local evidence and ordering are preserved in the JSON/CSV.

Unknowns: causal learning deficit, whether history/representation/contact dynamics would improve the learned policy, and any 4096 outcome. Next bounded check is the same saved-array attribution for a newly CPU-closed eligible endpoint. No new model selection or collection is authorized by this report.

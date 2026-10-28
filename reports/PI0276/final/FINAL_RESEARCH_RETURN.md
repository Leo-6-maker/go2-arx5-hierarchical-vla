# PI0276 — fixed StackCube task-learning result

Execution: COMPLETE. Physical task quality: VALID_FAIL.
Source-Panda physical simulation; NON_REALTIME. No target WBC or hardware execution.

| Lane | First success | Persistent success | Wilson 95% |
|---|---:|---:|---|
| TEST/BASE0 | 0/32 | 0/32 | [0.0, 0.1071791982550706] |
| TEST/TASK8192 | 0/32 | 0/32 | [0.0, 0.1071791982550706] |
| TEST/eef | 31/32 | 31/32 | [0.8425573617998744, 0.9944621398359969] |
| TEST/joint | 31/32 | 31/32 | [0.8425573617998744, 0.9944621398359969] |
| TRAIN_PHYSICS/TASK8192 | 0/16 | 0/16 | [0.0, 0.1936076805344365] |

Paired TEST: gained 0, lost 0, net gain 0/32; planned net-gain target >=8: False.

Source admission retained DEV8/8, TRAIN93/96 and VAL15/16; all original failures and the complete VAL20009 horizon failure remain in SOURCE_INDEX.json.
Offline384 and group16 tables report the fixed TRAIN/VAL panel, whole-H40 versus executed prefix8, gripper and original gate outcomes.
Command-to-reference installation, reference-to-actual tracking and object-task completion are separate columns in PHYSICAL144.csv.
Measured contacts, penetration, object transitions, joint-limit excursions and gripper saturation do not establish hardware safety.
BASE0 is AGG4096 weights with this new task processor; it is not the historical AGG policy. Only TASK8192 is evaluated; no intermediate selection.
Full4096/8192 Trainer and data/output/video archive identities are bound by FINAL_ARTIFACT_AUDIT.json after fixed-revision readback.
Selected/deployment: NONE. Stop for frozen-WBC composition or a separately authorized targeted task-failure decision.

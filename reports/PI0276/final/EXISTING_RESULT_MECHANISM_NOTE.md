# PI0276: fixed-result interpretation

This is a read-only aggregation of existing registered results, not a new evaluation or causal ablation. All original scores and failures remain.

| Lane | Sustained native grasp >=0.3s | Persistent stack |
|---|---:|---:|
| TEST joint expert |32/32|31/32|
| TEST EEF expert |32/32|31/32|
| TEST BASE0 |0/32|0/32|
| TEST TASK8192 |8/32|0/32|
| TRAIN TASK8192 |5/16|0/16|

Fine-tuning produced accepted offline references and some actual grasp acquisition. It did not produce complete stable stacking, even on the fixed TRAIN diagnostic. The result cannot be summarized as either no learned change or merely insufficient unseen-scene coverage.

The TASK TRAIN offline phase panel retains OPEN-command MAE of0.4777/0.5973/0.3847/0.4307 across groups0..3. CLOSE MAE is0.3383/0.1268/0.3703 for groups0..2; group3 has no CLOSED labels and remains undefined. These are conditional errors over the original genuine future commands, not measured pad opening and not a closed-loop success proxy. The complete TRAIN/VAL phase panel is in GRIPPER_PHASE_SUMMARY.json.

Command-to-actual EEF tracking, native IK holds, commanded opening, physical grasp and task release are distinct. TASK TEST has mean same-tick XYZ RMS0.012001m,3126 IK-hold ticks and9 gripper saturation ticks. These observations do not establish which component caused the task failures. Some geometrically plausible references can still mistime grasp, transfer or release, or enter states absent from the teacher replay.

The next decision should first examine existing TRAIN command/phase/gripper traces and actual object events, then preregister one matched intervention. No additional inference, teacher labels, physics, seed/loss search, output correction or target-WBC call is granted by this note. Source-Panda NON_REALTIME simulation only; selected/deployment remain NONE.

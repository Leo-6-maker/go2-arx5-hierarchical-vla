# WBC owner defect packet — PI0294 R4B D0

## Decision requested

Please review **arm servo realization under load**, together with body/arm prediction coupling. No controller patch has been applied. The evidence supports `ARM_SERVO_REALIZATION_BIAS` as the dominant orientation-error layer; the unique controller-design cause remains unresolved. No dropped documented feedforward, indexing, sign or actuator-transmission bug has been demonstrated.

The original EEF-only native hold remains a valid quality FAIL. It is neither GR00T policy failure nor a new gripper safety incident. All calculations here use saved state copies, with zero physics integration/model inference/rendering/PBS. The actual free root and frozen controller are untouched.

## Direct same-cycle evidence

356 native control records have measured arm and IMU-odometry messages at the exact same simulation stamp;87 lie in the final2s. These avoid the separate physics/control ZOH-age ambiguity. The IMU-to-root/mount transform is explicit. Sensor pipeline evaluation may precede integration by2ms; the compared fields belong to that same control snapshot.

| Offline FK state, actual body retained | Final2s XYZ RMS (m) | SO3 RMS (rad) |
|---|---:|---:|
| Actual measured arm |0.01030312|0.07042672|
| Only joint3 replaced by its recorded target |0.01023886|0.03362635|
| Only joint4 replaced by its recorded target |0.00311205|0.03518694|
| Joint3+4 replaced by recorded targets |**0.01758447**|0.00955994|
| All six joints replaced by recorded targets |**0.01605929**|0.01009321|

These are kinematic counterfactuals, not achieved physical rollouts or alternative quality passes. In particular, fixing the observed orientation difference in this frozen snapshot does not guarantee the position gate: joint3+4 replacement exceeds0.015m. Do not choose a one-joint controller intervention from the counterfactual score table.

The rotational site Jacobian agrees with six finite-difference probes (max error2.24e-10). Joint3/4 rotational change agrees with the Jacobian in these saved states (max residual5.85e-16rad). This is a geometric check, rather than adding two scalar joint errors and calling that a proof. SO3 chain products close to1.93e-14 Frobenius residual; scalar RMS angles are never added.

## Planner/body/consumer decomposition

Causal source-time ZOH, with both publisher source stamps and owner receipt availability; no fitted shift. In the final2s:

| State used for offline FK | Position RMS(m) | Orientation RMS(rad) |
|---|---:|---:|
| Candidate planned BASE + planned arm at arm preview time |0.00151067|0.00312610|
| Actual BASE + same planned arm |0.01605918|0.01011078|
| Actual BASE + published arm request |0.01606884|0.01011234|
| Actual BASE + last recorded native target |0.01607997|0.01011089|
| Actual physical TCP |0.01031833|0.07051246|

R1/R2 use the most recent causally observable policy candidate. **The active policy consumer ID was not logged.** 1557/1670 candidate arm previews match the published request within2e-6rad; this does not establish exact policy identity. Different BASE/arm preview times are original design choices, not an identified implementation bug. Mixed actual-body/future-plan states are explicitly counterfactual.

The recorded-native-target to actual-physical orientation segment is0.06794365rad RMS, far larger than the small request/recorded-target segment (~0.00057rad RMS). The exact same-cycle table independently supports the arm-realization finding. The current-body/future-arm combination is a mixed-state diagnostic and does not alone prove a base tracking defect. An additional same-future-time check compares each candidate preview with actual body state at that identical future timestamp:1646/1670 samples have future observations;24 late-tail futures remain unobserved. For313 source samples in the last2s, candidate-plan vs realized-base FK differences are0.01671156m /0.00885257rad RMS. Planned mount z is on average0.01661515m below actual. Exact consumer ID is missing and actual future execution includes replanning, so this establishes a candidate forecast discrepancy, not a unique low-level base cause. Do not claim all error belongs to arm servo.

Native observation-state FK reproduces1614 saved measured-EEF records: max position4.48e-8m, orientation2.46e-8rad. Model/frame inconsistency is not supported by this closure check.

## Torque evidence and what it cannot prove

Final2s same-control-snapshot means,87 samples:

| Joint | P term(Nm) | D residual inferred(Nm) | Recorded FF(Nm) | Recorded ctrl(Nm) |
|---|---:|---:|---:|---:|
|joint3|2.62072|2.84830|0|5.46902|
|joint4|0.89554|0.39097|0|1.28650|

`D residual = ctrl - FF - P` is algebraically inferred, **not a logged applied velocity or an independent validation of the PD equation**. P uses the native published float sensor, whereas SetCommand reads the underlying double sensor. Causal published dq can differ from the value actually consumed; applied dq remains UNKNOWN.

An independent force observation is available: for1670 saved physical samples, each arm generalized actuator force equals the model's gear1 clipped ctrl with max difference0Nm. The recorded mapping/transmission therefore has no unexplained torque loss in those samples. Limits are27Nm for joints1–3 and7Nm for joints4–6; see the actual bound supplement for saturation counts.

Zero-velocity reconstruction at saved post-state gives final2s gravity means5.46544Nm /1.28585Nm for joint3/4. Actual recorded force means are5.46366Nm /1.28532Nm. These separate sample sets are consistent with supporting load; they are not an exact synchronized full-dynamics identity. Recorded `qfrc_bias` includes gravity/Coriolis/centrifugal terms ([MuJoCo3.2.7 computation](https://mujoco.readthedocs.io/en/3.2.7/computation/index.html)). Only the independently reconstructed zero-velocity quantity is called gravity.

The saved qacc/force fields describe the pre-integration state; qpos is the post-integration state2ms later. Pre-step qpos and external generalized forces were not logged. Exact `M(q)vdot + bias` balance is UNKNOWN. The supplied post-state-M/pre-state-qacc residual is labeled a phase-mismatched diagnostic, not a failed balance test.

## Scope of the next decision

No current evidence justifies an autonomous gain/preview/feedforward-scale change, added gravity compensation or integral control. Those would be an explicit controller-version decision. No target bias cancellation, selective joint target correction or WBC replacement has been installed.

D0 already localizes the dominant orientation-error layer without another physical baseline. If the WBC owner needs exact servo/plan causation, the smallest extra logger should capture: active policy ID and evaluated state/input/sample times; measured q/dq actually used by SetCommand; applied q/dq/FF/gains and final ctrl; synchronized pre-step state/dynamics and external forces. A physical logger run or the declared BASE-soft-hold mode requires separately adopted D1 authority/window. Do not consume the four proposed starts merely because they exist on the unadopted card.

Future native dispatch now has a deployed-profile marker guard. A zero-physics check actually passed through enter/profile -> native ROS environment -> Python import; missing/wrong profile is rejected. This neither erases the original entry deviation nor requalifies the failed controller.

Original R4A FAIL, M2048 and all92 private evidence files remain intact. No v4 data/FB13 exists; selected/deployment=NONE.

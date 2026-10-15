# PI0294 R4B D0 — saved-evidence attribution and resource closeout

**D0_READONLY_ATTRIBUTION_COMPLETE. Primary evidence branch: ARM_SERVO_REALIZATION_BIAS.** Candidate body forecast/realization coupling is secondary; the unique controller-design cause remains unresolved. No concrete implementation defect has been demonstrated, no control patch applied, and no physical rerun performed.

Protocol fixed release:134389406df96711c97b8557d50ee620f741b3df, science canonical6182139684. Its proposed D1 execution is still **not adopted**. D0 used the original, actually ACTIVE R4 CPU/profile lease before its unchanged12:09:30SGT new-work cutoff; it created no R4B clock and reopened no lease. Native physical/data/FB13 chains stayed stopped throughout. Old R4A FAIL and original scores were not rewritten.

## Main result

The original physical hold remains0.069650907rad orientation RMS. Existing full MPC trajectories, native control targets, sensor/odometry, physics forces and actual model geometry suffice to localize most of the persistent orientation difference to the arm target-to-realization layer. This conclusion is supported by FK, a rotational Jacobian check and a separate same-native-timestamp analysis, rather than adding joint3/4 angles.

| Same native control/sensor stamp, final2s (87 snapshots) | XYZ RMS m | SO3 RMS rad |
|---|---:|---:|
| Measured arm + measured body |0.01030312|0.07042672|
| Joint3 target substituted offline |0.01023886|0.03362635|
| Joint4 target substituted offline |0.00311205|0.03518694|
| Joint3+4 targets substituted offline |**0.01758447**|0.00955994|
| All six targets substituted offline |**0.01605929**|0.01009321|

Counterfactuals are not actual controller performance. Joint3+4 substitution explains much of the orientation gap but worsens XYZ beyond0.015m; this cannot authorize a selective joint correction or a quality PASS. Geometric Jacobian finite differences agree to2.24e-10; SO3 chain composition closes to1.93e-14 without adding scalar RMS angles.

The separate causally aligned view contains1670 physical snapshots. Last2s candidate plan FK is0.00151067m /0.00312610rad from the reference; actual-body/planned-arm FK is0.01605918m /0.01011078rad. The request-to-recorded-target orientation segment is~0.00057rad RMS, whereas recorded-target-to-actual orientation is~0.06794rad. Exact consumed policy identity remains UNKNOWN;1557/1670 candidate previews match published arm requests within2e-6rad.

A separate **same future absolute timestamp** check avoids interpreting an intentional preview as current-time tracking error:1646 candidate forecasts have future observations;24 tail futures are unobserved. In313 source snapshots from the final2s, planned vs realized-body FK differs0.01671156m /0.00885257rad. Planned mount z is on average16.61515mm below actual. Subsequent replanning and unknown consumed-policy ID prevent treating this as a unique low-level body fault. See SAME_FUTURE_TIME_FORECAST_CHECK.json.

## Force evidence

Recorded arm actuator force equals gear1 clipped ctrl exactly across1670 samples. Saturation counts are zero for every arm actuator; minimum whole-window margins are joint3=13.72488Nm and joint4=3.56160Nm. Thus saved actuator transmission/limit evidence does not support an unexplained torque loss or saturation explanation.

At87 exact native control/sensor stamps in the final2s, joint3 P / inferred-D / FF / ctrl means are2.62072 /2.84830 /0 /5.46902Nm; joint4 means0.89554 /0.39097 /0 /1.28650Nm. **D is inferred algebraically, not directly measured.** Zero-velocity independent gravity reconstructions are approximately5.46544 /1.28585Nm in a different337-sample set. These quantities are consistent with load-bearing finite-PD behavior, not proof of a missing producer field or a unique gravity-compensation fix.

Pre-integration qacc/forces and post-integration qpos differ by2ms; pre-step qpos/external-force logs and actual applied dq_cmd are missing. Exact full dynamics accounting stays UNKNOWN. The stored phase-mismatched residual is diagnostic only. The definition of bias includes gravitational, Coriolis and centrifugal terms ([MuJoCo3.2.7](https://mujoco.readthedocs.io/en/3.2.7/computation/index.html#general-framework)); it is not labeled pure gravity except in the independently zero-velocity reconstruction.

## Engineering and decision

A minimal entry guard now rejects native launches without the deployed profile marker. The actual deployed enter/profile -> original native ROS environment -> Python import chain passed a zero-physics check, including missing/wrong-profile negatives. This fixes the future launcher path only; it does not erase the original entry deviation or provide new physical authority.

No implementation defect in indices/units/signs/dropped documented feedforward was reproduced. No change to WBC, gains, preview, velocity scale, costs, solver, friction, inertia, target or safety was made. D0 already supplies a WBC-owner packet; it does not require spending an unadopted physical baseline slot. If exact plan/servo causation is needed, the packet identifies the missing synchronous logger fields for a separately adopted D1. Adding gravity/integral compensation or changing gains is a controller-version decision.

All D1 modes and patch verification remain NOT_RUN. Fullbody Teacher/v4/FB13 remain NOT_RUN/NOT_PRODUCED/NOT_TRAINED. M2048 is retained, with its prior16-case panel still NOT_RUN; no new Hopper work was requested. selected/deployment=NONE.

## Resources and private preservation

Recorded D0 CPU through the first private seal and same-future check: **6.865379931s**, one core at a time, original R4 allowance. Stage-by-stage records and original clock are in D0_CPU_RESOURCE_RECEIPT.json. Post-publication metadata readback/closeout is recorded separately if applicable; budgets were not reset.

New native starts/integrations/rendering/GR00T or leg-policy queries/PBS/GPU/training/deletion/public raw uploads: **all0**. FK, geometric Jacobians, pure-state forward dynamics and zero-velocity gravity reconstruction were offline only, with no live simulator state.

The original92-file private archive was not modified. First D0 incremental private seal: **21 files /8980240 bytes**, every file read back, tree `a017d64baf53396c88df552a4413625d9acc74c89640d274b78afc44c40dcd61`. It preserves both initial and corrected source-causal analysis views. Later same-future and owner-note clarification are distinct small additions; they do not overwrite the initial seal. Raw reconstructed pose/force rows and private source-schema material stay private. Only self-authored tools, aggregate receipts and the WBC-owner note are published.

Read WBC_OWNER_DEFECT_NOTE.md for the concrete handoff; ATTRIBUTION_DECISION.json distinguishes established layer evidence from unresolved controller cause. The original negative control result is retained throughout.

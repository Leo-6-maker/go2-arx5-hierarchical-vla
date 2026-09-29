# PI0258 A0: observed controller contract and remaining scope

Authority: 8944d569d9f83baa74f0cb28c1113ba9659e528a / Issue11 5889346948.
Read-only reconstruction of all1500 recorded corrected-B ticks. New physics, actors and GR00T:0.
Original PI0256 FAILs and raw files remain unchanged.

## Actual actor conditioning

The actual PI0252/runtime/PI0226_pytorch3d_shim transforms.py implements
`matrix[..., :, :2]` while claiming PyTorch3D compatibility. The upstream
[matrix_to_rotation_6d source](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_rotation_6d)
uses the first two rows. The recorded current arm observation slots15:21 match
the shim column flattening within3.135e-7; their discrepancy from row flattening
reaches1.783021. Recorded actor history tails exactly equal the recorded current
observations, and those histories exactly equal the saved actor inputs.
Thus this is an actual conditioning representation difference, not a filename-only hypothesis.

The native rotation-error observation is a bounded quaternion vector proxy,
sin(theta/2)*axis, expressed in the actual base frame. It is NOT the full SO3
log despite its function name. It reconstructs within2.660e-7. Full log residuals
are retained separately for diagnosis. Position-error and target-position
observations reconstruct within1.747e-7. No fitted rotation offset was used.

Chain: native canonical quaternion -> world goal -> inverse-base quaternion ->
target orientation observation / heading-frame trajectory preview -> history ->
arm actor15D -> arm6 and body plan9 -> dog actor -> actuators.
The target6D difference is directly evidenced; full numerical preview replay is
not yet claimed. The compatibility shim also services preview orientation.
Original checkpoint training-import identity is not established by its filename;
the discrepancy does not yet prove the training-time convention or uniquely
explain the observed physical failure. No shim, WBC, gate, gain or actor was changed.

The resolved native config uses end_to_end arm actions, clip10, action scale.25,
and soft-limit factor.9. Every saved arm joint-position target exactly equals
.25*clip(raw_actor,-10,10). The arm raw max reaches11.5091, so clipping is real.
The9 body-plan channels remain below1 in these recordings. Finger DOFs have no
actor channel, but retain fixed-zero position drives; calling them freely
passive would be inaccurate.

## Joint and phase findings

| Bfix | Soft arm ticks | Gripper-only ticks | Hard-bound ticks | Hold XYZ RMS mm | Hold SO3 RMS rad |
|---|---:|---:|---:|---:|---:|
|00|175|75|110|7.672|1.317755|
|01|199|51|72|13.272|1.376905|
|02|0|250|0|15.075|.940290|
|03|0|250|0|10.627|.770726|
|04|165|85|74|7.147|1.152224|
|05|0|250|0|8.179|.497186|

All cases retain250/250 aggregate soft flags and250/250 gripper soft flags;
legs have0 soft violations. The arm hard penetrations are small: joint4 up to
.0001271rad, joint5 up to
.0000532rad; see the CSV for exact values/units. They are preserved, not rounded
away. Native termination occurs0 times in these six records. Soft bounds are
native penalty/external admission semantics, not an observed termination here.
All actual bounds, durations and penetrations are in per_joint_margins.csv.

Moving:96 ticks through1.95s. Terminal continuation:154 ticks beyond1.95s.
Hold angular-speed RMS remains.4323–.8672rad/s; hold XYZ speed RMS.0332–.0821m/s.
Gripper flags therefore do not explain the failed orientation/stability conjunction.

Original C0 established a finite position hit at its source attitude; Bfix rigidly
transported the full curve and orientation to randomized reset FK, then held to5s.
Neither original success nor accurate canonical loading proves arbitrary SE3
holding capability. Native sources contain an orientation tracking reward and
threshold, but the actual original training orientation-support distribution
remains UNKNOWN.

## Track status and handoff

CONTROLLER_EVIDENCE_PARTIAL: the source/record characterization is delivered;
training-time convention/support and causal correction remain unresolved.
Optional A1 is not launched while its conditioning integrity question is open.
A later physical version can propose the CPU-proved representation correction;
this goal does not silently repair and repeat physics.

B is independent: it uses the original row-packed StateV1/label conversion,
independent FK and its own isolated renderer/data admission. A's failure does not
block B. selected_checkpoint=NONE; deployment_selected_checkpoint=NONE.

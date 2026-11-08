# WM02 full-H40 trajectory adaptation V1

This is a pure NumPy upper-level geometry/reference adapter. It starts no
process, reads no model or dataset, fits no parameter, advances no physics,
and grants no native or model-domain qualification. Existing panels,
processors, source outcomes, labels, failures and CLOSED receipts retain
their original meaning. This is neither distillation nor a new Actor.

The first recipe uses the existing TARGET shadow as a trace geometry fixture
with `geometry_scale=.03`, explicitly `NON_VLA_CALIBRATION`. Later declared
grades are `.1`, `.25`, `.5`, and `1.`. Actual native qualification uses fresh
identities and the original admission/certifier/receiver in one environment,
one grade at a time; this module does not iterate or execute them. Only
`alpha=1` retains full raw geometry. A scaled trace is not a VLA task result,
source-task success, target-task match, or demonstrated transfer.

## Input and exact interface

`go2arx5_vla.world_model.fusion_trajectory_adapter.TrajectoryAdaptation` is a
frozen dataclass with this positional/keyword constructor:

```python
TrajectoryAdaptation(
    decoded_H40, query_TCP_source, T_source_from_target,
    decision_TCP_target, initial_reference_TCP_target,
    base_pose_hold, geometry_scale,
)
```

`decoded_H40` is the original Processor's **already decoded** finite floating
`[40,10]` physical EEF10: XYZ metres, native **row-packed** rot6d, mechanical
gripper fraction. All 40 original knots and original dtype/endian/C-order
bytes are retained in an immutable array with a SHA256 identity. This is an
array-byte hash, not an invented source archive/file hash; the caller retains
the original checkpoint/processor/observation/payload/file pins separately.
The only rotation conversion is the original `matrix10` row convention's
Gram–Schmidt orthogonalization. No Processor decode, normalization, relative
action accumulation, XYZ/rotation clipping, or knot deletion occurs.

The next four inputs are finite proper rigid `[4,4]` TCP transforms. The frame
map is the measured rigid `source_from_target`, not an object/TCP fit. The
fixed measured `base_pose_hold` is `[7]` with a unit quaternion; it is retained
and bound, never inferred from an EEF pose or changed by this adapter.

## Declared geometry and gripper intervention

With `F=source_from_target`, the target-world raw poses are
`M_i=inv(F) @ source_i` and the mapped query is `Q=inv(F) @ query_source`.
The explicit query-to-decision relocation is `L=decision @ inv(Q)`;
`U_i=L @ M_i`. An actual live same-anchor query should make `L` identity.
This calibration relocation must not be described as observed achieved motion.

For `D=decision`, desired translation is
`D.p + alpha*(U_i.p-D.p)`. Desired rotation is
`D.R @ Exp(alpha*Log(D.R.T @ U_i.R))`, the per-knot shortest SO(3) geodesic.
At a principal-log branch near pi, scaled neighboring rotations can be far
apart; retiming uses the **final** scaled rotations, not alpha times raw speed.

Source gripper fractions must first lie in `[0,1]`; out-of-range/nonfinite
values are rejected without clamping. The explicit new task adaptation is
`target_fraction=(.002+.040*source_fraction)/.044`: source range maps to the
target working stroke `[.002,.042]` metres, with `.044` metre mechanical
travel. This affine mapping applies equally to every comparison arm and is
not a second decode or actuator/WBC change. Geometry scale does not shrink
the gripper range. The prepended anchor holds working-open `.042/.044`.
The implementation retains the original endpoint arithmetic
`closed=.04*.05; open=.044-closed`, including floating-point identity with the
existing strict working-target guard; no guard tolerance is widened.

## Desired versus nominal reference and clock

The fixed INIT transport is deliberately world-displacement translation:

```text
p_reference = INIT.p + (p_desired - decision.p)
R_reference = INIT.R @ decision.R.T @ R_desired
```

It is not full SE(3) left multiplication on XYZ, and no subsequent measured
motion reanchors the frozen trajectory. The adapter retains separate mapped,
desired, nominal-reference and raw identities. Intent semantics are versioned
`INTENT_POSE_REFERENCE` (`desired_outcome_provenance=INTENT_POSE`), not actual
source-outcome supervision or proof that a target executes the intent.

Index zero is the measured decision desired pose and INIT nominal reference,
both with working-open grip. A separate integer-duration bridge reaches the
first adapted knot. All 39 remaining segments have one uniform integer time
stretch. Within each segment the time progress is rest-to-rest quintic
`s(u)=10u^3-15u^4+6u^5`, along the unchanged straight XYZ/grip and shortest SO(3)
geometry. All 40 knots are exact and have indices `bridge_atoms+i*retime_factor`.
These are **50ms quintic samples**; native sample interpolation remains
unchanged, so this is not a claim of analytic quintic execution at 500Hz.

Duration bounds include quintic peak factors `max(s')=1.875` and
`max(abs(s''))=10*sqrt(3)/3`: translation `.05m/s` and `.25m/s²`, rotation
`.25rad/s` and `1rad/s²`, physical finger `.04m/s`. Original native dt `.002s`,
atom dt `.05s`, and 25 native ticks per atom remain unchanged. These Cartesian
sample bounds do not guarantee native IK, joint velocity/acceleration,
collision, between-grid acceleration, contact, or model-support PASS.

Read-only `desired_path` / `reference_path` (alias `nominal_reference_path`)
have shape `[L,10]`; `end_atom=L-1`. `preview(elapsed_atoms=0)` returns exactly
40 nominal EEF10 rows at indices `min(elapsed_atoms+j,end_atom)`, including
final-pose holds. `desired_at(elapsed_atoms)` returns the corresponding
desired row. Clocks are nonnegative Python integers. Renewals retain the
same arrays and cannot consume later observations or reset progress.

The preview index is an issue-time grid. Native pose interpolation can track
the next pose over an atom; the separate `gripper.bin` receiver holds the
issued first-row grip. A varying-grip native integration therefore needs
explicit per-atom issue/consume evidence. Neither H40 suffix presence nor
`desired_at(e+1)` proves that the next grip has already been consumed.

## Evidence and full-domain bounds

`evidence()` returns JSON-safe input/array identities, frame/relocation,
gripper mapping, all 40 knot indices/native tick offsets, bridge/stretch,
whole-path geometry/rate diagnostics, and explicit unestablished qualification.
It writes nothing. `trajectory_domain_envelope(adaptation, candidates)` is
an offline finite catalogue of constant world-XYZ corrections. It includes
every nominal point and every translated candidate, with full XYZ extrema,
rotation and grip summaries. It grants no correction-join rate, model-domain,
native or execution permission. Old local XY qualification cannot establish
coverage of Z, rotation, grip, contact, H40 or this full trajectory.

Synthetic tests verify independent noncommuting frame/relocation/INIT
equations, all grades/all knots, raw bytes and immutability, complete gripper
range, integer timing and quintic samples, sampled rates/accelerations,
near-pi rotation branches, clock suffix/tail holds and full-path bounds.
They load no neural/physics runtime and are not actual native evidence.

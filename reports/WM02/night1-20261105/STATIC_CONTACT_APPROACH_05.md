# A concrete static contact approach, while Core remains unreachable

Source **`1f91227f3eff90b9d7f4ce9795e6c141f11e554e`** implements and actually runs a static, constrained six-joint arm approach on the saved **full Go2+ARX5** model. It reuses native step2,503 from the passed `night1-c07-original-q24-single-01` recording, original compiled scene SHA-256 `101b88b87966fceff190327b1fa630ad154b08b388928884c91500ac257713d8`, and the same original joint limits/contact-pair rules. The base, legs, object and original fingers are held at that recorded pose while solving arm IK. Original files are never modified. This is not a fixed-base source simulator substituted for target dynamics.

Two conditions were inspected and retained:

| Static proposal | Actual result |
|---|---|
| Descend to cube center | First24 waypoints clear; waypoint25 at TCP z0.469893m hits the table with **x5_link6 wrist geometry**, penetration0.240328mm. Waypoint26 is not evaluated after that collision. |
| Descend to cube center+0.020m | All26 static waypoints feasible in68 IK iterations. Final target z0.479893m, TCP position error0.118951mm, orientation error proxy0.0000336rad. No unallowed geometry contacts. |

For the second condition,13 additional **static finger-position intervals** respect the existing0.04m/s,50ms working-travel bound and reach17.5mm finger positions. Both original pads geometrically overlap the object's upper portion; maximum reported overlap is1.893039mm, below the unchanged4mm task-pair penetration rule. There are no forbidden geometry contacts. Original finger positions are restored after inspection. Static overlap is **not a measured grasp force, supported grasp, release or contact qualification**.

The fixed-source candidate run exited0 with process CPU0.323885s and wall0.323950s. Native integration, emitted actuator commands, expert labels, WM updates and Actor updates are all0. The earlier center-height and height-offset exploratory probes remain preserved; the fixed-source verification does not add independent scene samples or learning data. The arm waypoint timing/dynamics are not qualified. A native approach must still be rebuilt from the then-current stable target state and pass graded single-environment pose/gripper checks.

Actual implemented command on the pinned private recording:

```bash
PYTHONPATH=src "$WM02_REPLAY_PYTHON" -B -m scripts.wm02.inspect_contact_approach_kinematics \
  --config "$WM02_PRIVATE/CONFIG.private.json" \
  --output "$WM02_PRIVATE/APPROACH_STATIC.private.json" \
  --grasp-height-offset-m 0.020 --probe-gripper-geometry
```

This removes one geometric ambiguity from G2: a naive cube-center target collides with the actual wrist/table, while an upper-object grasp candidate is kinematically feasible at this snapshot. It does not remove the dynamic initialization/clock blocker, waive original safety, change the table/robot/policy or establish full task success. C08 still has no remote build/native receipt because both known SSH entry routes time out. The next remote operation remains readonly identity reconciliation, then the already prepared C08 build if no job exists. The original NIGHT1 deadline is unchanged.

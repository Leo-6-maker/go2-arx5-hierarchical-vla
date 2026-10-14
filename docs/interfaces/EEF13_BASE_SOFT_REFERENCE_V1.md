# EEF13 + bounded BASE intent -> native WBC soft reference contract V1

Date: 2026-10-15. Status: SPECIFICATION_TO_IMPLEMENT_AND_QUALIFY, not an implementation or execution receipt. Used by PI0294 R4A. This contract distinguishes a policy velocity preference, fixed safety bounds, WBC soft tracking, and actual downstream velocity.

## 1. Evidence and actual native interface

Static inspection in this review used the provided private `wbc_mujoco_minimal.tar.gz`: 215194129 bytes, SHA256 `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`. The older `WBCv1.zip` is 159225884 bytes, SHA256 `3c59a6d5f8f924df0b838ad2d501d3a853274d741db191a9de453f58212063ee`. These are different controller families, not interchangeable weights. No private package source, weights, mesh or exact host path is republished here.

Verified source locations inside the current MuJoCo package:
- `src/planning/go2_x5_planning/README.md`, section 6: optional BASE pose alongside EEF; BASE-only causes EEF to move with BASE.
- `go2_x5_planning/ee_trajectory_node.py::_message`: `state_trajectory` is 7D EEF pose or 14D EEF+BASE poses, quaternion XYZW; `input_trajectory` is zero with native input_dim=12.
- `src/ctrl/go2_x5_ocs2/config/task_floating.info::baseTracking`: enabled, weights x/y/yaw=5/5/2, z/pitch/roll=0.
- `FloatingBaseTrackingCost.h`: 7D EEF-only disables BASE tracking cost; 14D activates the weighted quadratic BASE-pose error. EEF is also a soft tracking cost, not guaranteed hard priority.
- `WbcBridgeCore.cpp::ToObservation`: the MPC BASE origin is the arm mounting plane, `p_mount = p_IMU + R * mount_offset`, not automatically the trunk or IMU origin.
- `WbcBridgeCore.cpp`: MPC outputs are converted to downstream body velocity, yaw/body angular velocity and posture commands; native acceleration limiting exists. This is a WBC OUTPUT path, not evidence that arbitrary VLA vx/vy/wz fields can be fed into the high-level JSON input.

The exact deployed package must be re-identified by the execution owner. A newer package with a real high-level velocity-reference API can be supported only after its parser, subscriber, consumer and constraints are traced and tested. Do not select a different native/policy-aware profile merely because it is present.

Conclusion: native compatibility is achieved by integrating bounded high-level BASE intent into the already supported BASE SOFT POSE reference, unless an independently verified newer API proves an equivalent velocity-reference route. Do not write policy velocity into the 12D zero input trajectory or publish directly to the downstream locomotion command channel.

## 2. Policy action and meanings

New versioned action `EEF13_BASEINTENT_V1`, shape H40 x 13 at 20 Hz:

| Slice | Quantity | Units / semantic |
|---|---|---|
| 0:3 | EEF target position | metres, absolute world/task-frame on disk |
| 3:9 | EEF target orientation | existing row-major rot6d |
| 9 | commanded gripper opening | existing normalized 0..1 convention |
| 10 | preferred BASE forward velocity | m/s, planar yaw frame at MPC mounting-plane origin |
| 11 | preferred BASE lateral velocity | m/s, same frame |
| 12 | preferred heading rate dpsi/dt | rad/s about world vertical, NOT automatically body-frame angular z |

The first 10 dimensions retain physical meanings, not necessarily the old fitted normalization range. The model outputs desired movement; it does not output its own permission to violate speed limits, a leg gait, a joint trajectory, force or torque.

Separate non-learned envelope fields contain velocity/acceleration maxima, TTL, pose/travel operating region and safety checks. Freeze these before data collection and evaluation. Initial commissioning ceilings are at most abs(vx)=0.15m/s, abs(vy)=0.10m/s, abs(dpsi)=0.30rad/s; these are conservative EXPERIMENT LIMITS, not certified hardware limits. Use the minimum of these and the current validated native operating envelope. Initial per-axis acceleration ceilings are 0.30/0.20m/s^2 and 0.60rad/s^2, also reduced to any tighter qualified native limit. Do not retune the controller's own weights or limits to match them.

Log raw policy preference, envelope status, installed BASE pose reference, WBC-selected downstream command and achieved base motion separately. Nonfinite or genuinely out-of-envelope model outputs are rejected and counted, not silently clipped into a successful trial. Numerical tolerance only follows a predeclared round-trip tolerance. Any task-independent slew transformation must be explicit, identical for Teacher/model and part of the bridge version; the rollout with no such modification is separately identified.

`base_preference_valid=false` means OMIT the BASE reference so the native EEF-only WBC can choose a base solution. `true` plus zero velocity means request a stationary BASE soft-pose preference. These are NOT equivalent. This flag is protocol metadata, not a hidden task selector.

## 3. BASE integration and frame binding

Define B at the native MPC mounting-plane origin, with yaw aligned to the body's heading and vertical aligned to the world gravity axis. At observation time t obtain actual world pose W_T_B(t), actual EEF W_T_E(t), simulation/ROS time and transforms. Retain roll, pitch, height and full mount transform for exact observation conversion even though the high-level intent is planar.

For each 0.05s command interval integrate the planar body-yaw twist using an SE(2) exponential:

`W_T_B_ref[k+1] = W_T_B_ref[k] * Exp_SE2([vx[k], vy[k], dpsi[k]] * 0.05)`.

Embed the result as world x/y/yaw BASE poses; fill non-targeted z/roll/pitch consistently from the declared binding state, without pretending zero-weight native costs track them. Keep wrap/unwrap handling explicit. For small-angle implementations, document and bound numerical error against Exp_SE2.

EEF relative decoding happens ONCE using actual W_T_E(t), independently of the future BASE trajectory. Do not attach the learned EEF path to the moving BASE after decoding. Send explicit `ee` AND `base` trajectories to avoid native BASE-only hand-following semantics.

If source odometry velocity is measured at the IMU/trunk rather than the mounting plane, transform the origin with the angular cross-offset term, then rotate into the declared yaw frame. Do not merely rotate velocity components. Likewise dpsi/dt is not body omega_z under roll/pitch; native WbcBridgeCore performs its own conversion.

Use fixed world-frame waypoints after observation binding. The native receiver otherwise binds relative frames to a LATER MPC observation. Preserve the original observation anchor, reference epoch and source clock explicitly in an outer transaction/bridge; stale action TTL causes rejection. Repeated identical packets must not re-integrate BASE motion, reset the time origin or double execute.

## 4. Time/horizon and expiry

H40 EEF/gripper points refer to t+0..1.95s; the 40 BASE velocity samples apply to intervals [t+k*0.05, t+(k+1)*0.05). BASE integration consequently produces 41 endpoints through t+2.0s. The native reference manager may request a closed 2s/0.05s window (41 or merged-knot points).

The bridge may hold the final EEF/gripper at t+2.0s for interpolation coverage; this is labelled bridge padding, NOT a 41st learned target or submitted training command. Prefix8 execution covers exactly 0.4s. All clocks are simulation/observation clocks; wall-clock inference delay is recorded independently.

The native manager stopping publication does NOT clear the last MPC target. Define expiry, disable-base-preference, cancel and HOLD explicitly. In simulation, pause physical integration and advance no old command while awaiting a new observation-bound action; continue only after a qualified action/ACK handshake. Never assume that an expired ROS message alone stops a moving robot. Hardware remains out of scope.

## 5. Observation schema and data lineage

Recommended versioned state groups (19 physical fields before official padding):
- measured EEF10;
- measured BASE planar pose relative to reset odometry frame: x, y, yaw (3);
- measured BASE planar velocity and heading rate in the same declared conventions (3);
- measured BASE height, roll, pitch (3).

All are actually observable simulation/odometry estimates, not goal pose, object oracle state, MPC future plan, Teacher phase, WBC hidden activations or desired base command disguised as measurement. Record estimators and timestamp alignment. Retain both body-mounted RGB and wrist RGB; the old world-fixed body_rgb is a different camera domain.

The new official modality schema must independently define EEF pose, gripper scalar and BASE velocity vector. Do not let the official relative-pose transform act a second time on BASE velocities. Verify masks, unit scale, padding, encode/decode, paired language and state timing with actual images. EEF13 policy features are not native 14D WBC pose targets, nor its 12D MPC state/input.

Each raw full-body episode saves:
- exact plant/controller/leg-policy/bridge/calibration hashes;
- reset pose and settling interval, free-root 6D motion, 12 leg and 6 arm joints plus actual gripper states;
- observed RGB/time/EEF/BASE state;
- Teacher requested EEF and BASE preference, bridge accepted references, WBC dispatched commands and actual motion;
- high-rate contacts, stability, limits, failures, live/shadow accounting;
- task, scene group, split, seed and acceptance reason as audit-only fields.

Old fixed-base EEF10 demonstrations have NO demonstrated BASE intention. Do not append three zero targets and call them full-body labels; do not infer BASE preference by differentiating EEF. Missing BASE supervision requires an explicit per-dimension loss mask and a verified compatible loader. The first new mobile fit uses full-body supervised data only by default; old data remain the baseline and regression source. Any later masked replay mixture is a separately sealed data decision.

Training labels use the Teacher's valid task-level requested/installed preference, not the WBC's corrective or achieved velocity. Train on full-body execution data only as explicitly labelled target-simulator adaptation; keep all attempts and independent task specifications to expose WBC feasibility filtering. Do not claim strict cross-embodiment independence merely because actions exclude joints.

## 6. Qualification before physics and model tests

Required CPU contract tests: positive/negative vx,vy,yaw; yaw wrap; tilted-base mount origin; zero vs absent preference; combined EEF-world-hold plus moving BASE; nonfinite/out-of-envelope; stale/duplicate/reordered packets; reset/cancel; 40-to-41 endpoint accounting; both rotation conventions; exact mask13 and optional state19 padding; absolute disk labels to one relative EEF conversion; checksum of fixed controller inputs/outputs.

Required actual native integration tests: all 12 legs actuated with ground contact, floating root not welded/teleported, BASE reference visibly received by baseTracking consumer, full command mode for translations, real achieved signed translation/yaw, EEF world stability while BASE moves, actual gripper contact and release, same plant/controller semantics under one and multiple isolated workers.

Test tolerances and operating envelopes are sealed BEFORE scores are seen. Use the prior valid EEF hold gate (position RMS <=0.015m, orientation RMS <=0.05rad) and source-clock residual <=0.025s where applicable; if they fail, report the failure instead of silently relaxing them. Existing fixed-base S1 does not qualify the distributed full-body OCS2+leg-controller pipeline.

## 7. Source context

Repository evidence: `docs/coordination/2026-10-10_WBC_PACKAGE_COMPARISON_AND_TARGET_LEARNING_REVIEW.md` at `ca2b738575e3c37a753454f3b59f227331045408`; research charter `docs/RESEARCH_CHARTER_TASK_SPACE_MANIPULATION_V1.md` at that same ref. Archive members were statically read in this review; no private executable/model was run.

The general distinction between references, costs and soft constraints follows official OCS2 documentation: https://leggedrobotics.github.io/ocs2/optimal_control_modules.html . No private package byte content is included here; consumers read their already-authorized copy.

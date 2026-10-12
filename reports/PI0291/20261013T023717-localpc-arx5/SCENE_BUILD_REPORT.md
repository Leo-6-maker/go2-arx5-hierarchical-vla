# PI0291 Native ARX5 Scene Build

Run: `20261013T023717-localpc-arx5`. Authorization: `15cc7957488fb879f5f050e3602327592da7036b`. This N1 record binds the first actual execution source `302ad022f3d3a74dcaba72aa46e59282effad215`; subsequent pre-pilot driver revisions require their own qualification and source receipts.

## Identity And Boundary

The scene is `FIXED_BASE_ARX5_TASK_SIM`, not the prior independent floating tool, a mobile WBC result, or hardware data. The delivered native Go2-X5 six-axis arm, mount, link dimensions, mesh, inertias, axes, limits and existing collision shapes are retained. Original private WBC files are read-only and were not executed. No third-party Isaac asset or new engine was used.

The minimal delivery archive hash was recomputed locally as `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`. Native MJCF, ARX5 URDF and SOURCE_MANIFEST hashes were recomputed on both local-PC and Corelab and match [ASSET_PROVENANCE.json](ASSET_PROVENANCE.json). The prior full 11,338-payload checksum audit was not repeated in N1.

WBCv1 ZIP identity `3c59a6d5f8f924df0b838ad2d501d3a853274d741db191a9de453f58212063ee` is historically documented, but its current copy was not located by bounded known-path checks. A different old 963,451,251-byte archive was not substituted or read as that 159,225,884-byte ZIP. WBCv1 is not a dependency of this scene. Private asset redistribution permission is not inferred from possession; no robot mesh or WBC source enters GitHub.

## Fixed Support And Articulation

The native `base_link` remains at world `[0,0,0.40]` m, and the arm mount offset `[0.05,0,0.10]` m places `x5_base_link` at `[0.05,0,0.50]` m. Removing only the base freejoint directly anchors the base to the world. Leg joints are removed after their fixed rotations are baked into the native bodies: left/right hip `+0.1/-0.1`, thigh `0.8`, calf `-1.5` rad. This is a supported robot geometry, not a claim of dynamically balanced standing or locomotion.

The actual compiled independent model has `nq=15`, `nv=14`, `nu=8`, no equality constraint and no mocap body: six arm revolute joints, two opposed finger slides, and one freely dynamic cube. Actuator indices 0-5 are the six arm motors; 6-7 are independent fingers. The old 18-DOF WBC sensors/leg-policy layout is not modified or claimed compatible.

The TCP is the actual same-tick MuJoCo `x5_ee` site, with link6 offset `[0.1424,0,0.0001057]` m. Independent analytical composition of the native chain differs from the static site transform by at most `4.44e-16`. It is not replaced by the other project's 86.57 mm TCP. [JOINT_TCP_GRIPPER_MAPPING.json](JOINT_TCP_GRIPPER_MAPPING.json) records the full named mapping.

## Driver And Gripper Assumptions

Physical arm motion comes from finite joint motor torque, bounded by native efforts `[27,27,27,7,7,7]` Nm. IK generates only executor joint targets; the high-level reference and label remain world EEF SE(3) plus gripper. Each 2 ms tick retains target slew, motor clipping, actual contacts and safety checks. There are no per-tick writes to physical arm/object qpos, no free-tool Cartesian wrench, no object force, weld, snap or attachment.

Native link7/8 visual bodies receive opposed prismatic joints from the URDF structure, each with 44 mm stroke. The normalized simulation channel is `g=0` closed and `g=1` open. Total inner gap is `0.001048 + q_left + q_right` metres, not a single 44 mm total opening. Compiled pad geometry independently yields 1.048 / 45.048 / 89.048 mm at each-finger strokes 0 / 22 / 44 mm.

Finger masses/inertias are delivered native values. Added finite pad boxes, their friction/contact solver, finger motor force/slew/damping/armature, arm viscous damping and independent controller gains are explicitly SIM_ONLY modeling assumptions, not physical calibration. Pads retain the URDF pad inner-face frames; 3 mm box half-width is placed outward from each inner face. Motor bounds are 8 N per finger, target slew 40 mm/s. Actual finger positions, fraction and inner gap remain distinct from desired targets.

### Pre-Pilot V2 Performance Engineering

After preserving the actual C29100/C29101/C29102 V1 originals, the owner approved `PI0291_REAL_CHAIN_V2_EXACT_TARGET_IK_CACHE` before any pilot. V2 reuses the six-iteration IK solution while the exact committed EEF target matrix is unchanged and invalidates on any target change or explicit cache reset. Each 2 ms motor slew, force clipping, actual state/contact reconstruction and safety check remains. The arm/cube physical state is never rewritten by this cache.

The IK warm-start numerical trajectory is not asserted bitwise equivalent to V1; V2 is an explicitly new pre-pilot driver identity, not a post-seal logging patch. The independent zero-integration `benchmark_driver_cache.py` checks saved C29100 commands, actual-state snapshot/contact equivalence for duplicate-forward removal, cache invalidation, independent FK residuals and measured CPU performance. Its actual receipt and a distinct C29103 physical qualification must be bound by the owner before V2 pilot admission. This N1 C29100 success and original source/physics hashes remain unchanged.

The first zero-step benchmark at source `3f8692c94ffcf65c0ef475e3b48d2ba37dfe9be2` failed its strict combined duplicate-forward equivalence assertion before writing numeric metrics. The proposed forward removal is rejected: the original explicit step `mj_forward` followed by the snapshot's `mj_forward` has been restored. No threshold is relaxed to label that proposal equivalent. A second bounded diagnostic preserves per-field/contact error metrics, the original 1e-12 state / 1e-10 N force criteria, and separate cache gates even when forward equivalence fails. The accepted V2 performance change is IK caching only, pending its distinct commissioning qualification.

## Task And Sensors

The table top is 0.44 m, selected from native-arm reachability. T1 retains the 40 mm / 50 g cube and 160 mm target region. Static IK checked source and goal TCP heights at table + 24 / 104 / 154 mm: all C29100 target position errors are below `6.18e-7` m and orientation errors below `7.15e-6` rad. Initial actual contacts are empty and initial safety is valid. These checks used zero integrations and one rendered pair, and alone do not prove grasping.

Both actual cameras are 256 x 256 at 20 Hz. `body_rgb` is honestly a world-fixed external rig, not a dog-mounted camera. `wrist_rgb` is parented to link6 and follows the real arm. Saved per-frame K, world-optical transform, cube keypoint projections, visibility, time and snapshot ID use the same simulation tick. Static exemplar calibration and original-frame review are in [CAMERA_CALIBRATION.json](CAMERA_CALIBRATION.json); a later full per-episode independent audit is still required for data admission.

## First Actual Contact Evidence

Root's C29100 run completed 32 simulated seconds, 16,000 integrations and 641 actual paired RGB/state snapshots. Its saved scorer reports sustained bilateral grasp, sustained lift, carry, actual release/withdrawal, and all 41/41 hold points passing with safety valid. Maximum measured lift is `0.0828952863` m; carry is `0.1590528403` m. This is one commissioning success, not an eight-pilot qualification or formal TRAIN64/DEV16 bank.

Physics SHA256: `19a788c6a6b4d999f0462ecf0ed3a040048f694e9d23a1a01d97b406ff9dc338`. Original saved dual RGB video: 641 frames, 20 fps, 1,067,840 bytes, SHA256 `7ea2c4f44c650830df51539e08f63deca2d235f2f4ab350da4fad61e2eaf63e9`. Video receipt verifies complete decoding; it adds no physics or new render.

Actual wrist PNGs at 4, 6, 9, 9.5, 13 and 28 s were visually inspected. The cube is clear during lift/carry/lower/release and after withdrawal/hold; no prior tool-domain palm occlusion was reused. Actual body frame at 4 s visibly includes native Go2, articulated ARX5, table, cube and blue region. Original-frame hashes are recorded in the calibration receipt. These images were not post-hoc rerenders or substituted training observations.

## Reproduction Entry

The exact private MJCF parent is resolved through the owner's private binding, not embedded in public reports. In the existing isolated MuJoCo 3.2.7 environment, with the frozen source and original private assets, the headless zero-integration command is:

```bash
MUJOCO_GL=egl \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
LIBGL_ALWAYS_SOFTWARE=1 CUDA_VISIBLE_DEVICES='' LP_NUM_THREADS=4 \
PYTHONPATH=src python -m scripts.pi0291_arx5_data.static_probe \
  --asset-root "$PI0291_NATIVE_ASSET_ROOT" --ids C29100 \
  --output "$NEW_STATIC_DIAGNOSTIC_DIR" --render
```

Use a new output directory and account its one diagnostic pair; no `MUJOCO_EGL_DEVICE_ID` is set for the Mesa path. Running a physical episode is separate and requires the owner's preserved ledger, source seal and valid budget. WBC calls, GR00T neural calls and hardware commands remain zero. `selected=NONE; deployment=NONE`.

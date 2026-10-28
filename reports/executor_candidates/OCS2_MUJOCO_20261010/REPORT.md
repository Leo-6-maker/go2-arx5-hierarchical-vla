# OCS2 + MuJoCo native reproduction — 2026-10-10

## Result

**NATIVE_REPRODUCTION_PASS**, limited to delivered native compilation, executable linkage and the package's bounded actual MuJoCo hold test. External reference tracking and object manipulation are **NOT_RUN**. This is an independently recorded executor candidate; the historical PI0287 verdict and evidence remain unchanged.

The user confirmed that this package represents the future Go2 + ARX5 deployment route. The package archive SHA256 is `6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52`. All 11,338 delivered checksum entries still match after compilation.

| Check | Actual result |
| --- | --- |
| Compilation | 20 ROS packages and `rl_sim_mujoco` completed; exit 0 |
| Local SSD build | 206.895 s wall / 689.650 child CPU s, two build jobs |
| Dynamic linkage | Simulator, controller and MPC executable pass `ldd -r` |
| Actual native hold | PASS using unchanged delivered predicates and timeouts |
| Feedback | 2,621 odometry / 2,621 arm / 2,908 EEF messages |
| Control | 850 MPC policies / 1,100 arm / 1,100 base commands |
| Final state | WBC mode; base z = 0.376829 m |
| Process closure | No owned candidate simulator/controller process remains |

The initial explicit linkage probe omitted the installed ROS multiarch library directory. Its FAIL receipt is preserved. Adding that directory resolves the missing `liburdfdom_model.so.4.0` and corresponding symbols without recompilation or source changes.

## Reproduction identity and engineering changes

- Delivered profile `native`, `task_floating.info`, `arx5_ac1_floating.urdf`, leg policy `N1_s29_175632`.
- ROS domain 228, localhost-only discovery; RViz disabled for the delivered headless hold test. No hardware driver was launched.
- Pinocchio 4.1.0 and Coal 3.0.3 match the delivery's dependency record. Actual rclcpp is 28.1.21; delivery validation used 28.1.22. This is a recorded dependency difference, not a claim of identical system installations.
- Missing dependencies were extracted from eight apt packages into a project-owned directory. Only absolute dependency CMake prefixes were relocated; no system packages, WBC weights/gains, solver settings or delivered source were modified.
- Shared-home compilation was stopped at the owned build process group and retained as engineering preparation. NFS metadata stalls motivated a fresh source-identical SSD working copy. This was not a scientific episode retry.
- The smoke harness changes only its source-root location and output directory. Its full predicates, sequence and timeouts remain unchanged. Original delivery validation files are preserved.

Corelab runtime source: `/var/cache/users/liuyu-nfs/Projects/go2-arx5/executor-candidates/wbc-mujoco-6dd13098/wbc_mujoco_minimal`.
Raw reproduction logs and receipts: sibling `reproduction/` directory. Original archive and extraction remain under `/home/liuyu-nfs/Projects/go2-arx5/private/wbc_mujoco_minimal_20261009_6dd13098/`.
Private source, weights and binaries are not included in this public report.

## Actual interface and remaining task requirements

The existing JSON EEF interface is `/go2_x5/ee/trajectory_json` → measured-state-bound reference manager → `/mobile_manipulator_mpc_target` → OCS2 → arm PD and RL legs. EEF poses use position + XYZW quaternion. Default references publish at 10 Hz with a 2 s window and 0.05 s samples. The native floating model has 12 state and 12 input dimensions; it does not use the historical Isaac Gym actor's 201-dimensional input or `TrajectoryBatch` interface.

The native MJCF has 12 leg and 6 arm joints, no independently actuated gripper, and no named body/wrist cameras. A fixed gripper model/TCP in the planning URDF does not establish grasp actuation or contact success.

Per the user's supplied migration review, a **new continuation protocol** must define the external-reference and grasp qualification, before replacing the formally bound historical execution route. It needs fixed reference conditions/budgets, requested/installed/achieved pose recording, native safety and numerical tolerances, reference timing/handoff semantics, gripper/contact and synchronized camera additions, object-success criteria, and full executor identity. Thresholds must be frozen before reading those task outcomes.

The hold test has not exercised RViz dragging, explicit HOME key 3, moving EEF references, 0.4 s replacement, independent gripper motion, dual-camera synchronization or grasp/carry/place. No target dataset or GR00T adaptation has started. No WBC optimizer update or hardware episode occurred. Formal target-executor freeze is pending; `selected=NONE`, `deployment=NONE`.

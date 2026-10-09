# PI0288 — bounded native startup closeout

Authority: `PI0288_MUJOCO_TASK_ENV_AND_EXPERT_DATA_GO_V1`, release `ede6d93b0ced2b0d409f992e7168b066da1f2c32`, Issue #11 canonical #6087400691.

**Terminal: ENV_READY_INTERFACE_BLOCKED. Overall execution PARTIAL.** Native compiled environment is available; task environment, G0 clock contract, six reference conditions, grasp/place and training input are not qualified. This is an exact early safety stop, not a completed task dataset or controller tracking verdict. Selected/deployment remain NONE.

## Observed startup problem

One new, isolated setup episode `55f23c661cdc4c8aa50a8731928dc101` ran one real MuJoCo integration, 0.002 simulation seconds. The supplied loader constructs `mjData` at model defaults. It does not select the supplied `home` keyframe. A separate CPU load with **zero integrations** reproduced all four calf joints at q=0, versus raw upper limit -0.83776 rad: each starts 0.83776 rad outside its limit.

| Joint | First integrated q (rad) | Raw upper limit | Positive violation |
|---|---:|---:|---:|
| FL calf | -0.00823329349 | -0.83776 | 0.82952670651 |
| FR calf | -0.00823329364 | -0.83776 | 0.82952670636 |
| RL calf | -0.00823312166 | -0.83776 | 0.82952687834 |
| RR calf | -0.00823312178 | -0.83776 | 0.82952687822 |

The registered immediate hinge threshold is 0.001 rad. The observer called the original integrator exactly once, then raised the native SIGINT safety stop. The original owner result says `Feedback unavailable`; raw physics and simulator logs establish that safety stopped before the ROS simulator initialization completed. The original result is preserved; `ADJUDICATION.json` records the independent cause. An arm joint also showed a 0.0001403338-rad negative excursion, retained as a numerical warning.

This stop does not show MPC failed a movement command: no new movement reference was issued and no MPC/arm/base feedback was received. It does not invalidate the earlier bounded hold smoke or imply that a `home` replacement is safe. The old smoke lacked this raw startup safety observer.

PI0288 section3 expressly rejects an illegal initial condition and forbids temporary pose substitution. Applying `home`, changing qpos0, or suppressing the startup limit violation would change this execution condition; none was applied. The six G1 slots remain **NOT_RUN**, not a fabricated 0/6 quality result. Further native starts are blocked by a zero-step preflight check.

## Actual implementation and checks

- Reused original three binaries, native task/model and RL assets. All nine selected current pin hashes match; parent full 11,338-file readback is referenced separately. No private controller source, mesh, weights or binary is published.
- Original integration modules cover the six quantitative references, native JSON, named state/control mapping, provisional scene/gripper/cameras, contact-based teacher, desired-command recording, task scoring, H40 windows and data admission.
- A read-only `mj_step` observer copies state for refreshed same-time TCP FK. Native callbacks must be null, model identity is guarded, and nonfinite state is fail-stopped. A zero-step check confirms private-copy FK leaves original integration state byte-identical. The observer adds a safety stop and logging; it does not change controller output or physics parameters.
- Local CPU suite: **13 passed, 1 skipped**. The skipped native-source callback check ran separately on Corelab and passed: valid/invalid acceptance preserves original behavior, and original send/publish are each called once. Zero neural/model optimization queries.
- The scene overlay was CPU-parsed against actual delivered URDF/MJCF. Existing finger inertials remain unchanged; no duplicate tool mass. This is not a loaded/stepped/rendered task scene or an equipped reference PASS.

G0 source interface is documented, including actual MPC policy target readback. Clock mapping, reset/expired-reference negatives, actual URDF-FK versus MuJoCo-TCP parity, installed/dispatch tracking and two-camera runtime remain unobserved. No mocked receipt replaces them.

## Separate gripper specification gap

User confirmation establishes the new OCS2+MuJoCo package as the future deployment route. It does not identify the deployed 2023/2025 gripper or verify its working specifications. Delivered model: TCP from link6 `(0.1424,0,0.0001057)` m; fixed finger masses 0.08924/0.089402 kg; declared slide 0–0.044 m and model limits 5 m/s/100 N. These are model declarations, not measured hardware operation limits.

Missing: deployment gripper model/revision, verified operating speed/force, and contact-pad geometry/material/friction. Native meshes have no verified pad segmentation. The clarification was requested once. Scene remains PROVISIONAL_GRIPPER, cameras SIM_PROVISIONAL_CAMERA; no TASK_EXECUTOR_FREEZE.

## Budget and remaining outcomes

One setup slot, one integration; zero fixed references, teacher/pilot/formal data, RGB pairs, GR00T forwards, optimizer updates or hardware actions. Per-launch upper reservations were recorded before launch. RL forward/MPC attempts/direct dispatch counters remain UNOBSERVED and retain their reserved bounds, rather than being asserted zero. Exact total CPU core time was not observed; no all-cap PASS is asserted. Native owned processes have stopped. No H200/PBS/mail/monitor/cleanup/merge.

Raw small startup/CPU evidence was read back byte-for-byte (9 files, 16,893 bytes). Full native launch logs remain on the isolated SSD project directory. No task video exists: the safety stop preceded the first task RGB frame. No synthetic video substitutes for it.

### Next scientific decision needed

Bind a legal native initialization/reset contract before resuming G0/G1: explicitly decide whether the supplied `home` keyframe is the registered start and how pre-controller transitions are judged. `home` has not been tried or claimed as a cure. Preserve this failed startup and cumulative accounting. Working gripper specifications are additionally required for formal G2 freeze. Existing reference, scene and teacher CPU modules can be reused; downstream data success remains unmeasured.

Recompute this conclusion using `py -3.12 scripts/pi0288_mujoco/closeout.py`. Original evidence files are not rewritten.

# Source-backed capability matrix

Evidence: S0/platform_source_readback.json and S0/resolved_native_config_cpu.json. Nine external controller/asset components were fully hashed again and matched the existing admitted source receipt. This does not claim a newly hashed full private tree.

| Capability | Verified source/config | Qualification boundary |
|---|---|---|
| Native WBC | Original arm/dog 039999 actors; 6 arm actions | Only registered expert motion episodes authorized |
| Physics/control | 0.005 s physics × decimation 4 = 0.02 s | Collector reads actual simulator time; reset's implicit step counts |
| EEF action | H40 × 0.05 s, unchanged gate/trajectory loader | Desired labels, achieved measurements separate |
| TCP | link6 offset [0.1424,0,0.0001057] m | Existing native convention retained |
| Frame | World-aligned rotation, immutable translation anchor | Full 4×4 transform saved; nonzero rotation CPU negative tested |
| Sensors | Existing body/wrist 96×96 RGB, measured StateV1 | Actual camera projection/view matrices and timestamps captured |
| Geometry | Two URDF prismatic fingers and collision pads, 0–0.044 m | Geometry exists; executable parallel-gripper transport unverified |
| State/action gripper | Existing unexecuted neutral 0.5 channel | Cannot establish gripping ability; actual joint feedback audit only |
| New scenes | User-owned spheres, axes, contour and fixed collision box | Robot/controller/dynamics remain unchanged |
| Contact skills | No verified object-state task/executor chain | MANIPULATION_PLATFORM_DECISION_REQUIRED |

Nominal FK TCP [0.2901,0,0.5966057] m uses nominal base [0,0,0.34] m and default arm q=0. It is a geometric calculation, not the actual settled pose. Registry geometry derives from it before any outcome. The necessary chain reach bound is 0.9605681129013536 m; it is not an IK or physical feasibility proof.

New expert episodes must record real reset tensors/RNG/images, achieved/desired TCP, base state, contact/termination, gate decisions, camera matrices and unexecuted gripper status. Identical seeds alone do not prove matched counterfactual reset images.

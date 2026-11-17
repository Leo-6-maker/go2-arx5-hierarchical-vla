# E2 native profile cold repair — zero scientific effects

The successor now preserves the frozen native interpreter/NumPy2.5.4/MuJoCo3.2.7/ROS profile. The learner and GPU provider retain sla-pt/NumPy1.26.4. The common spec builder had incorrectly overwritten the native interpreter with the learner's path. A zero-node,zero-integration preflight confirmed the difference;it also found that the new native scope imported a learner module requiring SciPy,which is absent from the original native environment.

The minimal repair restores the per-family native interpreter and moves the identical V5.2 package contract into a lightweight metadata module. Its validation function AST is unchanged. The E2 preflight checks the case interpreter against its pinned original template. In the real frozen native environment,the ROS message type-support libraries,Pose,native capture/consumer modules and actual B1 manifest contract now import successfully.53 focused CPU checks pass. No dynamics or new model generation was used for this repair.

The old followup was cooperatively stopped only after independently confirming WAIT_ACTUAL_E1_ENDPOINTS,no CURRENT_CHILD,no offline/E2 directories and one owned process. It is terminal with PID0,empty cgroup,CLOSED lease and its original PRESERVED_NO_REPLAY result retained. Its source/spec/result remain unchanged. B1/B2 training units and source were untouched;B2 continued on the same process.

New managed successor: `wm02-sla-mobile-f2-v52-e1-e2-followup-20261117-02`. Source `2797d0c2c458667e241aa0d24d6d0b005f3caf1a`;seal `4895567ec1295459b3e8c51cbe9d67d069278a5c7107d15381cb9ddaff20df42`;spec `9d0d02c39749219116498a85ec0323e0a2c78cf89e6cee9813a3d7f3421eaf01`. A separate read confirmed the new unit and advancing wait state. Its deadline remains exactly1794967938.2895093;no allocation or campaign extension occurred.

This is technical repair1 for the E2 native profile. The same frozen split,ROLL96 and24 E2 condition identities remain. Its600 offline generations and24 native requests were all verified unstarted in the prior capsule;none is replayed. The new service reuses the actual E1 endpoints and waits for B2,then follows the original bounded sequence. WBC,legs,MPC,solver,arm PD,H40/P10/TTL/2ms,contact/STOP,scoring and all old failures remain unchanged.

New effects attributable to this repair: **0 updates /0 generations /0 native requests /0 integration**. This profile check does not establish a new mobile-grasp result.

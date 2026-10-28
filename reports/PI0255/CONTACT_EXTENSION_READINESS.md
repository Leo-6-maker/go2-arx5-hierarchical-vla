# Contact extension readiness — specification only

Status: MANIPULATION_PLATFORM_DECISION_REQUIRED. No pushing, grasping, welding, object teleport, new attachment, controller or robot modification is authorized here.

The current URDF has x5_gripper_joint and x5_joint8 prismatic joints, finger collision pads, effort/velocity limits, and a fixed gripper-center TCP. The controller exposes six arm actions. Source geometry and finger gains do not establish an executable language-policy → gripper command → measured aperture/force path; the existing 0.5 parallel channel is unexecuted.

PUSH_TO_REGION future admission card: bind a real movable object, collision shapes/material/friction/mass, independently measured object pose and goal region; verify genuine contact, no teleport/weld, object displacement and sustained region occupancy; record robot/object contact and forbidden-collision events. Validate observations and frozen WBC capability in an isolated approved contact protocol first.

GRASP_LIFT_PLACE future admission card: bind actual joint/actuator command transport and measured aperture/force/contact feedback, object mass/payload and collision geometry, grasp/lift/placement predicates and release. Require measured object lift and stable placement without virtual attachment. Verify gripper range and force semantics before any claimed skill trial.

Missing contact readiness does not block authorized M1/M2/M3 expert motion pilots. Their success would demonstrate non-contact motion data feasibility, not manipulation or generalization.

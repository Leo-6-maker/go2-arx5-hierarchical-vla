"""Finite MPlib task-path preparation from a saved native state.

CPU kinematics only: no mj_step, controller, model inference, launcher or runtime
admission. Joint trajectories are planning witnesses, never WBC/student actions.
Private asset derivatives and witness arrays stay outside source Git.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from .persistent_approach_draft import pose
from .task_contacts import allowed

URDF_SHA256 = '403ce252c525437d66aa361a8a65b90bfe2b69dad0fcb1f7a17109582c7261bc'
JOINTS = tuple('joint'+str(i) for i in range(1, 7)) + ('gripper_joint', 'joint8')
NATIVE_JOINTS = tuple('x5_joint'+str(i) for i in range(1, 7)) + ('task_left_slide', 'task_right_slide')
VERSION = 'ARX5_SAVED_STATE_MPLIB_TASK_PATH_PREPARATION_V1'


def grasp_precontact_candidates(object_world):
    """Fixed OBB-derived T1 family, a new task geometry proposal only.

    Declare every candidate before planning. Symmetric 30-degree tool tilts are
    grasp choices, not corrections fitted to WBC tracking residuals. All failed
    plans remain in the candidate table. Native contact/clearance is still needed.
    """
    obj = pose(object_world)
    if obj[2, 2] < .999:
        raise ValueError('UPRIGHT_TABLETOP_CUBE_GEOMETRY_REQUIRED')
    downward = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    result = []
    for name, angle in (('VERTICAL', 0.), ('TILT_NEG30', -np.pi/6), ('TILT_POS30', np.pi/6)):
        c, s = np.cos(angle), np.sin(angle)
        ry = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        goal = np.eye(4)
        goal[:3, :3] = obj[:3, :3] @ downward @ ry
        goal[:3, 3] = obj[:3, 3]+[0., 0., .08]
        result.append(dict(id=name, goal_world=goal, tool_local_y_tilt_rad=angle,
            height_above_object_center_m=.08, launch_allowed=False,
            task='T1_PRECONTACT_GEOMETRY_PROPOSAL_NOT_SUCCESS_DEMO'))
    return result


def push_precontact_candidates(object_world, goal_xy, pad_geometry_zero_travel):
    """Direction-aware T2 family using measured native pad support geometry.

    Zero-travel geometry is the original scene helper's geometric origin, not a
    command to close to a mechanical endpoint. Actual finite gripper targets and
    closure/contact qualification remain unchanged and separately required.
    """
    obj = pose(object_world)
    target = np.asarray(goal_xy, dtype=float)
    if obj[2, 2] < .999 or target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError('UPRIGHT_OBJECT_AND_FINITE_PLANAR_PUSH_GOAL_REQUIRED')
    delta = target-obj[:2, 3]
    if np.linalg.norm(delta) < .10:
        raise ValueError('DECLARED_PUSH_DIRECTION_AND_TASK_DISTANCE_REQUIRED')
    direction = delta/np.linalg.norm(delta)
    pads = []
    if len(pad_geometry_zero_travel) != 2:
        raise ValueError('EXACTLY_TWO_MEASURED_NATIVE_PAD_BOXES_REQUIRED')
    for row in pad_geometry_zero_travel:
        center = np.asarray(row['center_tcp'], dtype=float)
        axes = np.asarray(row['axes_tcp'], dtype=float)
        half = np.asarray(row['half_size'], dtype=float)
        if (center.shape != (3,) or axes.shape != (3, 3) or half.shape != (3,)
                or not all(np.isfinite(x).all() for x in (center, axes, half)) or np.any(half <= 0)):
            raise ValueError('FINITE_NATIVE_PAD_BOX_GEOMETRY_REQUIRED')
        frame = np.eye(4)
        frame[:3, :3] = axes
        pose(frame)
        pads.append((center, axes, half))
    yaw = np.arctan2(direction[1], direction[0])
    c, s = np.cos(yaw), np.sin(yaw)
    heading = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    downward = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    support = float(np.abs(direction @ obj[:2, :3]) @ np.full(3, .02))
    result = []
    for name, angle in (('VERTICAL', 0.), ('TILT_NEG30', -np.pi/6), ('TILT_POS30', np.pi/6)):
        c, s = np.cos(angle), np.sin(angle)
        rotation = heading @ downward @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        front = max(float((rotation @ center)[:2] @ direction
            + np.abs(direction @ (rotation @ axes)[:2]) @ half) for center, axes, half in pads)
        bottom = min(float((rotation @ center)[2]
            - np.abs((rotation @ axes)[2]) @ half) for center, axes, half in pads)
        goal = np.eye(4)
        goal[:3, :3] = rotation
        goal[:2, 3] = obj[:2, 3]-direction*(support+front+.025)
        goal[2, 3] = .44-bottom+.004+.08
        result.append(dict(id=name, goal_world=goal, direction=direction.copy(),
            front_support_m=front, bottom_offset_m=bottom,
            tool_local_y_tilt_rad=angle, launch_allowed=False,
            task='T2_PRECONTACT_GEOMETRY_PROPOSAL_NOT_SUCCESS_DEMO'))
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_private_urdf(source, output):
    """Preserve original collision geometry/limits; no automatic SRDF search.

    Only visuals and mesh path spelling change. Adjacent-link exclusions are
    explicit. This arm-only URDF cannot certify all Go2/leg/table collisions.
    """
    source, output = Path(source).resolve(), Path(output)
    if digest(source) != URDF_SHA256:
        raise ValueError('EXACT_REVIEWED_ARX5_URDF_REQUIRED')
    if output.exists():
        raise FileExistsError('PRIVATE_PLANNING_DERIVATIVE_ALREADY_EXISTS')
    tree = ET.parse(source)
    robot = tree.getroot()
    for link in robot.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
    meshes = {}
    for mesh in robot.findall('.//mesh'):
        uri = mesh.attrib['filename']
        if not uri.startswith('package://go2_x5_description/'):
            raise ValueError('UNREVIEWED_COLLISION_MESH_URI')
        resolved = (source.parents[2] / uri.removeprefix('package://')).resolve()
        if not resolved.is_relative_to(source.parents[1]) or not resolved.is_file():
            raise ValueError('EXISTING_PACKAGE_COLLISION_MESH_REQUIRED')
        meshes[uri] = digest(resolved)
        # MPlib 0.2.1 resolves mesh names relative to the derived URDF directory.
        mesh.attrib['filename'] = os.path.relpath(resolved, output.resolve())
    output.mkdir(mode=0o700)
    urdf, srdf = output/'planning.private.urdf', output/'planning.private.srdf'
    tree.write(urdf)
    semantic = ET.Element('robot', name=robot.attrib['name'])
    for joint in robot.findall('joint'):
        ET.SubElement(semantic, 'disable_collisions',
            link1=joint.find('parent').attrib['link'],
            link2=joint.find('child').attrib['link'], reason='Adjacent')
    ET.ElementTree(semantic).write(srdf)
    for path in (urdf, srdf):
        path.chmod(0o600)
    return urdf, srdf, meshes


@dataclass
class SavedNativeState:
    model: object
    data: object
    qpos: np.ndarray
    joint_addresses: np.ndarray
    arm_limits: np.ndarray
    mount: np.ndarray
    tcp: np.ndarray
    site: int
    state_role: str = 'SAVED_ACTUAL_OBSERVATION'


def planned_future_configuration(native, arm_joints):
    """Independent hypothetical stage state. Never call it achieved feedback."""
    import mujoco
    q = np.asarray(arm_joints, dtype=float)
    if (q.shape != (6,) or not np.isfinite(q).all()
            or np.any(q < native.arm_limits[:, 0]) or np.any(q > native.arm_limits[:, 1])):
        raise ValueError('FINITE_LIMITED_PLANNED_FUTURE_JOINT_WITNESS_REQUIRED')
    data = mujoco.MjData(native.model)
    qpos = native.qpos.copy()
    qpos[native.joint_addresses[:6]] = q
    data.qpos[:] = qpos
    mujoco.mj_forward(native.model, data)
    tcp = np.eye(4)
    tcp[:3, :3], tcp[:3, 3] = data.site_xmat[native.site].reshape(3, 3), data.site_xpos[native.site]
    return SavedNativeState(native.model, data, qpos, native.joint_addresses.copy(),
        native.arm_limits.copy(), native.mount.copy(), tcp, native.site,
        state_role='PLANNED_FUTURE_CONFIGURATION_NOT_OBSERVED')


def load_saved_native(model_path, expected_sha256, sample):
    """Load one pinned closed model and actual observation into a private copy."""
    import mujoco
    if digest(model_path) != expected_sha256:
        raise ValueError('CLOSED_NATIVE_MODEL_HASH_MISMATCH')
    model = mujoco.MjModel.from_binary_path(str(model_path))
    if (model.nq, model.nv, model.nu) != (34, 32, 20):
        raise ValueError('FULL_GO2_ARX5_TASK_MODEL_REQUIRED')
    qpos = np.asarray(sample['qpos'], dtype=float)
    qvel = np.asarray(sample['qvel'], dtype=float)
    ctrl = np.asarray(sample['ctrl'], dtype=float)
    if (qpos.shape != (34,) or qvel.shape != (32,) or ctrl.shape != (20,)
            or not all(np.isfinite(x).all() for x in (qpos, qvel, ctrl))):
        raise ValueError('COMPLETE_FINITE_ACTUAL_STATE_REQUIRED')
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'x5_ee')
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'x5_base_link')
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in NATIVE_JOINTS]
    if min([site, body, *joints]) < 0:
        raise ValueError('ACTUAL_TCP_MOUNT_AND_JOINT_NAMES_REQUIRED')
    tcp = np.eye(4)
    tcp[:3, 3], tcp[:3, :3] = data.site_xpos[site], data.site_xmat[site].reshape(3, 3)
    if not np.allclose(tcp, pose(sample['tcp']), atol=1e-10, rtol=0):
        raise ValueError('SAVED_ACTUAL_TCP_NATIVE_FK_MISMATCH')
    mount = np.eye(4)
    mount[:3, 3], mount[:3, :3] = data.xpos[body], data.xmat[body].reshape(3, 3)
    return SavedNativeState(model, data, qpos.copy(), model.jnt_qposadr[joints],
        model.jnt_range[joints[:6]].copy(), mount, tcp, site)


def native_configuration_witness(native, joint_path, max_segment_rad=.02, *, planning_object_motion=None):
    """Check dense static configurations with the FULL original native model.

    No dynamics, controller tracking, continuous swept-volume or gripper-contact
    qualification follows. Root/legs/object retain the saved configuration only
    in this private witness. That is not a weld or reset of any physical worker.
    """
    import mujoco
    q = np.asarray(joint_path, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or not 0 < len(q) <= 10000 or not np.isfinite(q).all():
        raise ValueError('FINITE_SIX_JOINT_PLANNING_WITNESS_REQUIRED')
    if np.any(q < native.arm_limits[:, 0]-1e-10) or np.any(q > native.arm_limits[:, 1]+1e-10):
        raise ValueError('PLANNED_NATIVE_JOINT_LIMIT_REJECT_NO_CLIP')
    if max_segment_rad != .02:
        raise ValueError('FIXED_STATIC_SAMPLING_RULE_REQUIRED')
    dense = [q[0]]
    for a, b in zip(q, q[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b-a))/max_segment_rad)))
        if len(dense)+count > 50000:
            raise ValueError('FINITE_STATIC_CONFIGURATION_SAMPLE_CAP')
        dense.extend(a+(b-a)*(i/count) for i in range(1, count+1))
    bad, tcp = [], []
    try:
        for index, point in enumerate(dense):
            native.data.qpos[:] = native.qpos
            native.data.qpos[native.joint_addresses[:6]] = point
            mujoco.mj_forward(native.model, native.data)
            if planning_object_motion is not None:
                planning_object_motion(native.model, native.data)
                mujoco.mj_forward(native.model, native.data)
            actual = np.eye(4)
            actual[:3, :3] = native.data.site_xmat[native.site].reshape(3, 3)
            actual[:3, 3] = native.data.site_xpos[native.site]
            tcp.append(actual)
            for contact in native.data.contact:
                names = [mujoco.mj_id2name(native.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    for g in (contact.geom1, contact.geom2)]
                if not allowed(*names) or contact.dist < -.004:
                    bad.append(dict(sample=index, pair=names, distance_m=float(contact.dist)))
    finally:
        native.data.qpos[:] = native.qpos
        mujoco.mj_forward(native.model, native.data)
    return dict(status='DISCRETE_NATIVE_CONFIGURATION_REJECT' if bad else 'DISCRETE_NATIVE_CONFIGURATION_CHECK_PASS',
        samples=len(dense), max_joint_segment_rad=max_segment_rad, violations=bad,
        whole_robot_model_used=True, continuous_swept_collision_proved=False,
        physical_execution_qualified=False), np.asarray(tcp)


def prepare_path(*, native, urdf, srdf, goal_world, planning_object_motion=None):
    """One finite original-goal planning call; no seed/goal replacement loop."""
    import mplib
    if mplib.__version__ != '0.2.1':
        raise ValueError('REVIEWED_MPLIB_0_2_1_REQUIRED')
    goal = pose(goal_world)
    mplib.set_global_seed(294)
    planner = mplib.Planner(urdf=urdf, srdf=srdf, move_group='gripper_center',
        joint_vel_limits=np.full(6, .7), joint_acc_limits=np.ones(6))
    if tuple(planner.user_joint_names) != JOINTS:
        raise ValueError('EXPLICIT_NATIVE_TO_PLANNER_JOINT_ORDER_REQUIRED')
    q = native.qpos[native.joint_addresses].copy()
    if (not np.allclose(planner.joint_limits[:6], native.arm_limits, atol=3e-6, rtol=0)):
        raise ValueError('PLANNER_NATIVE_ARM_LIMIT_MISMATCH')
    planner.pinocchio_model.compute_forward_kinematics(q)
    local = planner.pinocchio_model.get_link_pose(planner.link_name_2_idx['gripper_center'])
    world = native.mount @ np.asarray(local.to_transformation_matrix())
    error = float(np.max(np.abs(world-native.tcp)))
    if error > 1e-8:
        raise ValueError('PLANNER_NATIVE_TCP_FRAME_MISMATCH')
    planner.set_base_pose(mplib.Pose(native.mount))
    planner.robot.set_qpos(q, full=True)
    receipt = dict(version=VERSION, launch_allowed=False,
        input_state_role=native.state_role,
        evidence_role='STATIC_KINEMATIC_PLANNING_NOT_WBC_EXECUTION',
        native_TCP_frame_max_abs_difference=error, seed=294, IK_initializations=20,
        RRT_planning_time_s=2., normal_sample_period_s=.05,
        joint_witness_is_student_or_controller_action=False,
        dynamic_reachability_proved=False, new_physics=0, new_neural_queries=0)
    if planner.check_for_self_collision(q):
        return dict(receipt, status='PLANNER_START_SELF_COLLISION_REJECT'), None
    result = planner.plan_pose(mplib.Pose(goal), q, time_step=.05,
        planning_time=2., fix_joint_limits=False)
    if result['status'] != 'Success':
        return dict(receipt, status='PLANNING_FAILED_NO_FEASIBLE_PATH_CERTIFIED',
            planner_reason=result['status'], impossibility_proved=False), None
    witness = np.asarray(result['position'])
    if witness.ndim != 2 or witness.shape[1] != 6:
        raise ValueError('EXPLICIT_SIX_ARM_JOINT_PATH_REQUIRED')
    check, tcp = native_configuration_witness(native, witness, planning_object_motion=planning_object_motion)
    return dict(receipt, status=check['status'], native_configuration_check=check,
        planner_duration_s=float(result['duration'])), dict(
            joint_witness=witness, time_s=np.asarray(result['time']), dense_TCP_world=tcp,
            dense_TCP_samples_are_not_timed_20Hz_action_labels=True)

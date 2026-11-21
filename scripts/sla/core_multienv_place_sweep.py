"""Fixed-base geometry shadow over a saved TRAIN carry-to-placement boundary."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.multitask_carry_geometry_v6 import world_yaw
from go2arx5_vla.world_model.multitask_score_v6 import footprint_inside
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_hopper_carry_audit import stopped_pairs

CONDITION = 'MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-red-left'
QUERIES_SHA = '94fc6fb7202403df377b0d40f68c91baef132d0a1c4a14d13060928f4b5fcd84'
ORIGINAL = dict(offset_mm=[8, 8], additional_yaw_deg=0)
CANDIDATE = dict(offset_mm=[4, 6], additional_yaw_deg=-15)


def is_below(model, body, root):
    while body and body != root:
        body = model.body_parentid[body]
    return body == root


def site_pose(data, site):
    out = np.eye(4)
    out[:3, :3] = data.site_xmat[site].reshape(3, 3)
    out[:3, 3] = data.site_xpos[site]
    return out


def quat(rotation):
    orientation = np.empty(4)
    mujoco.mju_mat2Quat(orientation, np.ascontiguousarray(rotation).ravel())
    return orientation if orientation[0] >= 0. else -orientation


def rotation_vector(rotation):
    result = np.empty(3)
    mujoco.mju_quat2Vel(result, quat(rotation), 1.)
    return result


def pose_path(a, b):
    delta = rotation_vector(b[:3, :3] @ a[:3, :3].T)
    angle = float(np.linalg.norm(delta))
    axis = delta/angle if angle else np.array([0., 0., 1.])
    distance = np.linalg.norm(b[:3, 3]-a[:3, 3])
    segments = max(1, int(np.ceil(max(distance/.0025, angle/np.deg2rad(2)))))
    q, rotation = np.empty(4), np.empty(9)
    for ratio in np.linspace(0., 1., segments+1)[1:]:
        row = np.eye(4)
        row[:3, 3] = (1.-ratio)*a[:3, 3]+ratio*b[:3, 3]
        mujoco.mju_axisAngle2Quat(q, axis, ratio*angle)
        mujoco.mju_quat2Mat(rotation, q)
        row[:3, :3] = rotation.reshape(3, 3) @ a[:3, :3]
        yield row


class ArmShadow:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        measured = TwoObjectState(model, 'red')
        self.site, self.fingers = measured.tcp, measured.finger_q
        root = model.body('x5_base_link').id
        arm = {i for i in range(model.nbody) if is_below(model, i, root)}
        tool_chain = {i for i in arm if is_below(model, model.site_bodyid[self.site], i)}
        joints = [i for i in range(model.njnt)
                  if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_bodyid[i] in tool_chain]
        if len(joints) != 6 or not np.all(model.jnt_limited[joints]):
            raise ValueError('Expected the six bounded joints on the x5 tool chain')
        self.q = model.jnt_qposadr[joints]
        self.v = model.jnt_dofadr[joints]
        self.ranges = model.jnt_range[joints]
        self.target = model.geom('task_cube_geom').id
        self.other = model.geom('task_blue_cube_geom').id
        self.table = model.geom('task_table').id
        cube = model.geom_bodyid[self.target]
        free = model.body_jntadr[cube]
        if model.body_parentid[cube] != 0 or model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Expected the original unparented red free body')
        self.free = model.jnt_qposadr[free]
        pads = {model.geom(name).id for name in ('task_left_pad', 'task_right_pad')}
        collision = [i for i in range(model.ngeom) if model.geom_bodyid[i] in arm
                     and (model.geom_contype[i] or model.geom_conaffinity[i])]
        if not pads <= set(collision):
            raise ValueError('Missing frozen arm collision geoms or pads')
        self.pairs = {
            'arm_to_blue': [(g, self.other) for g in collision],
            'arm_to_table': [(g, self.table) for g in collision],
            'nonpad_arm_to_red': [(g, self.target) for g in collision if g not in pads],
            'held_red_to_blue': [(self.target, self.other)],
            'red_to_table': [(self.target, self.table)],
        }
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.buffer = np.zeros(6)

    def solve(self, goal):
        model, data = self.model, self.data
        start = data.qpos[self.q].copy()
        for _ in range(80):
            mujoco.mj_fwdPosition(model, data)
            actual = site_pose(data, self.site)
            xyz = goal[:3, 3]-actual[:3, 3]
            angle = rotation_vector(goal[:3, :3] @ actual[:3, :3].T)
            residual = (float(np.linalg.norm(xyz)), float(np.linalg.norm(angle)))
            if residual[0] <= .00025 and residual[1] <= np.deg2rad(.35):
                return dict(matched=True, xyz_mm=1000*residual[0], angle_deg=float(np.rad2deg(residual[1])),
                            largest_joint_move_rad=float(np.max(np.abs(data.qpos[self.q]-start))))
            mujoco.mj_jacSite(model, data, self.jacp, self.jacr, self.site)
            weight = .08
            jac = np.vstack((self.jacp[:, self.v], weight*self.jacr[:, self.v]))
            step = jac.T @ np.linalg.solve(jac @ jac.T + .001**2*np.eye(6), np.r_[xyz, weight*angle])
            step *= min(1., .12/max(1e-12, float(np.max(np.abs(step)))))
            data.qpos[self.q] = np.clip(data.qpos[self.q]+step, self.ranges[:, 0], self.ranges[:, 1])
        return dict(matched=False, xyz_mm=1000*residual[0], angle_deg=float(np.rad2deg(residual[1])))

    def evaluate(self, goal, desired_object, grip):
        data = self.data
        data.qpos[self.fingers] = grip
        fit = self.solve(goal)
        if not fit['matched']:
            return dict(fit=fit)
        data.qpos[self.free:self.free+7] = np.r_[desired_object[:3, 3], quat(desired_object[:3, :3])]
        mujoco.mj_fwdPosition(self.model, data)
        gaps = {}
        for name, pairs in self.pairs.items():
            gap, x, y = min((float(mujoco.mj_geomDistance(self.model, data, a, b, .25, self.buffer)), a, b)
                            for a, b in pairs)
            gaps[name] = dict(gap_mm=1000*gap, contact=[self.model.geom(i).name for i in (x, y)])
        contacts = [dict(geom1=int(c.geom1), geom2=int(c.geom2), dist=float(c.dist)) for c in data.contact]
        return dict(fit=fit, gaps=gaps, forbidden=stopped_pairs(self.model, dict(contacts=contacts), 'task_cube_geom'))

    def evaluate_one(self, sample, geometry, candidate):
        self.data.qpos[:] = sample['qpos']
        self.data.qvel[:] = 0.
        mujoco.mj_fwdPosition(self.model, self.data)
        arm_initial = site_pose(self.data, self.site)
        object_initial = np.asarray(sample['object_pose'])
        tool_from_object = np.linalg.solve(arm_initial, object_initial)
        object_from_tool = np.linalg.inv(tool_from_object)
        jaw = np.asarray(sample['qpos'])[self.fingers]
        zone = geometry['zones']['left']
        half = geometry['object_half_sizes']['red']
        top = geometry['table_top_world_z']
        travel = object_initial.copy()
        travel[:2, 3] = np.asarray(zone['center_world_XY']) + .001*np.asarray(candidate['offset_mm'])
        travel[:3, :3] = world_yaw(np.deg2rad(candidate['additional_yaw_deg'])) @ travel[:3, :3]
        placed = travel.copy()
        placed[2, 3] = top + half[2]
        start_descent, rest = travel @ object_from_tool, placed @ object_from_tool
        withdraw = rest.copy()
        withdraw[2, 3] += .060
        phases = {
            'CARRY_TURN': [(obj @ object_from_tool, obj, jaw) for obj in pose_path(object_initial, travel)],
            'LOWER': [(tcp, tcp @ tool_from_object, jaw) for tcp in pose_path(start_descent, rest)],
            'OPEN': [(rest, placed, jaw+(np.full(2, .040)-jaw)*r) for r in np.linspace(.05, 1., 20)],
            'WITHDRAW': [(tcp, placed, np.full(2, .040)) for tcp in pose_path(rest, withdraw)],
        }
        evaluated, first_problem = {}, None
        for phase, points in phases.items():
            rows = []
            for i, (tcp, obj, gripper) in enumerate(points):
                result = self.evaluate(tcp, obj, gripper)
                rows.append(result)
                if not result['fit']['matched']:
                    first_problem = first_problem or dict(phase=phase, waypoint=i, type='IK', fit=result['fit'])
                    break
                if result['forbidden']:
                    first_problem = first_problem or dict(phase=phase, waypoint=i, type='CONTACT',
                                                         contact=result['forbidden'][0])
            matched = [x for x in rows if x['fit']['matched']]
            evaluated[phase] = dict(planned=len(points), evaluated=len(rows), IK_matched=len(matched),
                                    forbidden_waypoints=sum(bool(x['forbidden']) for x in matched),
                                    lowest_gaps={name:min((x['gaps'][name] for x in matched), key=lambda g:g['gap_mm'])
                                                 for name in self.pairs} if matched else {},
                                    worst_joint_move_rad=max((x['fit']['largest_joint_move_rad'] for x in matched),
                                                             default=None))
            if len(matched) != len(points):
                break
        return dict(native_step=sample['native_step'], footprint_inside=bool(footprint_inside(placed, half, zone)),
                    phases=evaluated, first_problem=first_problem)


def audit(folder):
    spec_path = folder/'EXECUTION_SPEC.json'
    spec = json.loads(spec_path.read_bytes())
    raw = folder/'native-01/LIVE_DATA/QUERIES.private.json'
    scene = Path(spec['model_path'])
    if (mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None
            or spec['panel_entry_id'] != CONDITION or spec['collection_split'] != 'TRAIN'
            or digest(raw) != QUERIES_SHA or digest(scene) != spec['frozen_files'][str(scene)]):
        raise PermissionError('Only the frozen F1 failed TRAIN trajectory is in this shadow panel')
    captures = [q for q in json.loads(raw.read_bytes()) if q['phase'] == 'CARRY'][-8:]
    if len(captures) != 8 or not all(q['actual_state']['native_grasp'] for q in captures):
        raise ValueError('Expected eight saved grasped carry states')
    with forbid_integration(mujoco) as attempted:
        shadow = ArmShadow(mujoco.MjModel.from_binary_path(str(scene)))
        values = [dict(candidate, rows=[shadow.evaluate_one(dict(q['actual_state'], native_step=q['native_step']),
                       spec['multitask_geometry'], candidate) for q in captures])
                  for candidate in (ORIGINAL, CANDIDATE)]
    if attempted:
        raise RuntimeError('Geometry audit attempted native integration')
    return dict(stage='F1_RED_LEFT_FIXED_BASE_WHOLE_ARM_SHADOW', condition=CONDITION, scene=digest(scene),
                spec=digest(spec_path), queries=QUERIES_SHA, snapshots=len(captures), result=values,
                native_integrations=0, native_requests=0, model_generations=0, optimizer_updates=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=Path)
    print(json.dumps(audit(p.parse_args().case), allow_nan=False))

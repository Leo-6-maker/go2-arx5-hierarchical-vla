"""Feedback teachers; push reference uses measured physical pad support."""
from __future__ import annotations

import numpy as np
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from scripts.pi0291_arx5_data.teacher import Teacher as PickPlaceTeacher
from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_slerp, quat_to_matrix
from .scene import push_geometry

PUSH_STAGES = ('approach', 'close', 'align', 'contact', 'push', 'detach', 'withdraw', 'hold')


class PushTeacher:
    def __init__(self, card, initial, engine):
        self.card, self.initial = card, initial.copy()
        direction = np.asarray(card['goal_xy'])-initial[:2, 3]
        self.geometry = push_geometry(engine, direction)
        self.rotation = np.asarray(self.geometry['rotation'])
        self.direction = np.asarray(self.geometry['direction'])
        self.stage, self.stage_since, self.events = 'approach', 0., []
        self.lifted, self.carried = False, False
        cube_support = float(np.abs(self.direction @ initial[:2, :3]) @ np.full(3, .02))
        self.offset = cube_support+self.geometry['front_support_m']
        self.source_tcp = np.r_[initial[:2, 3]-self.direction*self.offset, self.geometry['tcp_height_m']]
        self.goal_tcp = np.r_[np.asarray(card['goal_xy'])-self.direction*self.offset+.001*self.direction,
                              self.geometry['tcp_height_m']]
        self.final_tcp = None

    def target(self, state):
        target = np.eye(4)
        target[:3, :3] = self.rotation
        target[:3, 3] = self.source_tcp
        opening = 0.
        if self.stage in ('approach', 'close'):
            target[:2, 3] -= self.direction*.025
            target[2, 3] += .08
            if self.stage == 'approach':
                opening = 1.
        elif self.stage == 'align':
            target[:2, 3] -= self.direction*.012
        elif self.stage == 'contact':
            target[:2, 3] += self.direction*.001
        elif self.stage == 'push':
            target[:3, 3] = self.goal_tcp
        elif self.stage == 'detach':
            target = self.final_tcp.copy()
            target[:2, 3] -= self.direction*.04
        elif self.stage in ('withdraw', 'hold'):
            target = self.final_tcp.copy()
            target[:2, 3] -= self.direction*.04
            target[2, 3] += P['withdraw_clearance']
            opening = 1.
        return target, opening

    def observe(self, state):
        time_s = state['time_s']
        self.lifted |= bool(state['object_pose'][2, 3]-self.initial[2, 3] > .005)
        self.carried |= bool(np.linalg.norm(state['object_pose'][:2, 3]-self.initial[:2, 3]) >= .10)
        target, _ = self.target(state)
        reached = (np.linalg.norm(target[:3, 3]-state['tcp'][:3, 3]) <= P['position_stage_tolerance_m']
                   and geodesic_angle(target[:3, :3], state['tcp'][:3, :3]) <= .04)
        contact = bool(state['left_object_contact'] or state['right_object_contact'])
        new = None
        if self.stage == 'approach' and reached:
            new = 'close'
        elif self.stage == 'close' and state['gripper_fraction'] <= .05:
            new = 'align'
        elif self.stage == 'align' and reached:
            new = 'contact'
        elif self.stage == 'contact' and contact:
            new = 'push'
        elif self.stage == 'push' and reached and self.carried:
            new = 'detach'
            self.final_tcp = state['tcp'].copy()
        elif self.stage == 'detach' and reached and not state.get('robot_object_contact', contact):
            new = 'withdraw'
        elif self.stage == 'withdraw' and reached:
            new = 'hold'
        if new:
            self.events.append(dict(time_s=time_s, previous=self.stage, stage=new, actual_contact=contact,
                                    actual_displacement_m=float(np.linalg.norm(state['object_pose'][:2, 3]-self.initial[:2, 3]))))
            self.stage, self.stage_since = new, time_s

    def plan(self, state):
        initial = state['tcp']
        target, opening = self.target(state)
        offset = np.arange(40)/20
        delta = target[:3, 3]-initial[:3, 3]
        distance = np.linalg.norm(delta)
        poses = np.repeat(initial[None], 40, axis=0)
        speed = .04 if self.stage in ('contact', 'push', 'detach') else P['reference_speed_m_s']
        poses[:, :3, 3] += np.minimum(1., speed*offset/max(distance, 1e-12))[:, None]*delta
        angle = float(geodesic_angle(initial[:3, :3], target[:3, :3]))
        for i, t in enumerate(offset):
            poses[i, :3, :3] = quat_to_matrix(quat_slerp(matrix_to_quat(initial[:3, :3]), matrix_to_quat(target[:3, :3]),
                                                        min(1., P['reference_angular_speed_rad_s']*t/max(angle, 1e-12))))
        return dict(poses=poses, opening=np.full(40, opening), offset_s=offset, stage=self.stage,
                    anchor_pose=initial.copy(), decision_time=state['time_s'])


def make_teacher(card, initial, engine):
    if card['task'] == 'T1':
        return PickPlaceTeacher(card, initial)
    if card['task'] == 'T2':
        return PushTeacher(card, initial, engine)
    raise ValueError('Unregistered task')

"""Source-only geometric feedback Teachers for door and single-book skills.

Plans persist in world coordinates between stage entries. New H40 decisions
occur only every eight actual 20Hz observations. No joint command, reset,
physics execution, successful-demo admission or student inference occurs here.
"""
from copy import deepcopy
from itertools import product
import math

import numpy as np

from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_slerp, quat_to_matrix
from .object_relations import pose, inverse, door_turn, book_insertion

OPEN, CLOSED = .042/.044, .002/.044
HELD_STAGES = {'close', 'verify_grasp', 'turn', 'lift', 'preinsert', 'insert'}


def straight_path(start, goal):
    """Geometric SE3 reference; contact/collision feasibility remains unproved."""
    start, goal = pose(start), pose(goal)
    distance = np.linalg.norm(goal[:3, 3]-start[:3, 3])
    angle = float(geodesic_angle(start[:3, :3], goal[:3, :3]))
    ticks = max(1, math.ceil(max(distance/.06, angle/.3)/.05))
    if ticks > 640:
        raise ValueError('SOURCE_STAGE_PATH_EXCEEDS_32_SECONDS')
    amount = np.linspace(0, 1, ticks+1)
    result = np.repeat(start[None], ticks+1, axis=0)
    result[:, :3, 3] += amount[:, None]*(goal[:3, 3]-start[:3, 3])
    q0, q1 = matrix_to_quat(start[:3, :3]), matrix_to_quat(goal[:3, :3])
    for i, weight in enumerate(amount):
        result[i, :3, :3] = quat_to_matrix(quat_slerp(q0, q1, weight))
    return result


class SourceSkillTeacher:
    def __init__(self, card):
        if card.get('task') not in ('DOOR', 'BOOK'):
            raise ValueError('SOURCE_DOOR_OR_BOOK_TASK_REQUIRED')
        self.card, self.task = deepcopy(card), card['task']
        self.withdraw_rotation=None
        if 'withdraw_tcp_rotation_world'in card or 'port_revision'in card:
            from .cabinet_repair02 import checked_card
            side=card['cabinet']['orientation']
            if not checked_card(card,side):raise ValueError('ONLY_EXPLICIT_PORT02_RELEASED_WRIST_RETURN')
            self.withdraw_rotation=np.asarray(card['withdraw_tcp_rotation_world'],dtype=float)
        self.spec = self.card['door' if self.task == 'DOOR' else 'book']
        self.grasp_offset = pose(self.spec['object_to_tcp_grasp'])
        direction = np.asarray(self.spec['approach_direction_object'], dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1) > 1e-8:
            raise ValueError('DECLARED_UNIT_OBJECT_APPROACH_DIRECTION_REQUIRED')
        self.direction = direction.copy()
        self.stage, self.events, self.stop_reason = 'approach', [], None
        self.index, self.last_plan = -1, -1
        self.initial_time = self.grasp_since = self.hold_since = None
        self.state = self.path = self.goal = self.path_start = None
        self.preinsert = self.insert = self.release = self.withdraw = None
        self.grasp_goal = self.retreat_direction = None
        self.lift_initial_object_z = None
        self.candidate_complete = False

    def _transition(self, stage):
        self.events.append(dict(observation_index=self.index, time_s=self.state['time_s'],
            previous=self.stage, stage=stage, actual_grasp=bool(self.state['native_grasp'])))
        self.stage, self.path, self.goal = stage, None, None

    def _reached(self):
        return (self.goal is not None
                and np.linalg.norm(self.state['tcp'][:3, 3]-self.goal[:3, 3]) <= .004
                and geodesic_angle(self.state['tcp'][:3, :3], self.goal[:3, :3]) <= .04)

    def _book_supported_inside(self):
        s = self.state
        size = np.asarray(self.spec['book_size_m'])
        bounds = np.asarray(self.spec['cavity_bounds_shelf_m'])
        book_shelf = inverse(self.spec['shelf_world']) @ s['object_pose']
        points = np.array(list(product((-1., 1.), repeat=3)))*size/2
        points = points @ book_shelf[:3, :3].T+book_shelf[:3, 3]
        return (bool(s.get('object_supported_by_shelf', False))
                and bool(np.all(points >= bounds[:, 0]-.001) and np.all(points <= bounds[:, 1]+.001))
                and bool(np.linalg.norm(s['object_linear_velocity']) <= .02)
                and bool(np.linalg.norm(s['object_angular_velocity']) <= .1))

    def _door_at_target(self):
        s = self.state
        return (abs(float(s['hinge_angle_rad'])-self.spec['target_angle_rad']) <= .035
                and abs(float(s['hinge_velocity_rad_s'])) <= .05)

    def _book_targets(self):
        s = self.state
        targets = book_insertion(actual_tcp_world=s['tcp'], actual_book_world=s['object_pose'],
            target_book_world=self.spec['target_book_world'], shelf_world=self.spec['shelf_world'],
            book_size_m=self.spec['book_size_m'], cavity_bounds_shelf_m=self.spec['cavity_bounds_shelf_m'],
            actual_grasp_valid=bool(s['native_grasp']))
        return targets.preinsert_tcp_world, targets.insert_tcp_world

    def _withdraw_direction(self):
        return (-pose(self.spec['shelf_world'])[:3, 0] if self.task == 'BOOK'
                else self.state['object_pose'][:3, :3] @ self.direction)

    def _update_hold(self):
        if self.stage != 'hold': return
        s = self.state
        condition = (not s['robot_object_contact'] and s['gripper_fraction'] >= .85
                     and np.linalg.norm(s['tcp'][:3, 3]-s['object_pose'][:3, 3]) >= .08
                     and (self._door_at_target() if self.task == 'DOOR' else self._book_supported_inside()))
        if not condition: self.hold_since = None
        elif self.hold_since is None: self.hold_since = s['time_s']
        self.candidate_complete = bool(condition and s['time_s']-self.hold_since >= 2.)

    def observe(self, state):
        """Consume an actual packet once. Time alone cannot advance a stage."""
        if self.stop_reason:
            raise PermissionError(self.stop_reason)
        s = deepcopy(state)
        s['tcp'], s['object_pose'] = pose(s['tcp']), pose(s['object_pose'])
        if s.get('safety_valid') is not True:
            self.stop_reason = 'ORIGINAL_NATIVE_SOURCE_SAFETY_STOP'
            raise PermissionError(self.stop_reason)
        if not math.isfinite(s['time_s']) or not 0 <= s['gripper_fraction'] <= 1:
            raise ValueError('FINITE_ACTUAL_CLOCK_AND_GRIPPER_REQUIRED')
        if self.initial_time is None:
            self.initial_time = s['time_s']
        if self.index >= 640 or abs(s['time_s']-self.initial_time-(self.index+1)*.05) > 1e-8:
            raise ValueError('EXACT_ACTUAL_20HZ_SEQUENCE_REQUIRED')
        if self.task == 'DOOR' and (not math.isfinite(s['hinge_angle_rad']) or not math.isfinite(s['hinge_velocity_rad_s'])):
            raise ValueError('ACTUAL_PASSIVE_HINGE_FEEDBACK_REQUIRED')
        self.index += 1
        self.state = s
        if s['native_grasp']:
            if self.grasp_since is None: self.grasp_since = s['time_s']
        else:
            self.grasp_since = None
            if self.stage in ('turn', 'lift', 'preinsert', 'insert'):
                self.stop_reason = 'ACTUAL_GRASP_LOST_REQUIRES_NEW_RECOVERY_FROM_THIS_STATE'
                raise PermissionError(self.stop_reason)
        # Stage changes take effect on the next normal Prefix8 boundary. The
        # high-rate owner still stops immediately on actual native safety.
        if self.index % 8:
            self._update_hold()
            return self.stage
        reached, stage = self._reached(), self.stage
        if stage == 'approach' and reached: self._transition('align')
        elif stage == 'align' and reached: self._transition('close')
        elif stage == 'close' and s['native_grasp']: self._transition('verify_grasp')
        elif stage == 'verify_grasp' and self.grasp_since is not None and s['time_s']-self.grasp_since >= .3:
            self.lift_initial_object_z = float(s['object_pose'][2, 3])
            self._transition('turn' if self.task == 'DOOR' else 'lift')
        elif stage == 'turn' and self._door_at_target():
            self.release = s['tcp'].copy(); self._transition('release')
        elif stage == 'lift' and reached and s['object_pose'][2, 3]-self.lift_initial_object_z >= .06:
            self.preinsert, self.insert = self._book_targets()
            self._transition('preinsert')
        elif stage == 'preinsert' and reached: self._transition('insert')
        elif stage == 'insert' and reached and self._book_supported_inside():
            self.release = s['tcp'].copy(); self._transition('release')
        elif stage == 'release' and not s['native_grasp'] and s['gripper_fraction'] >= .85:
            self.withdraw = s['tcp'].copy()
            direction = self._withdraw_direction()
            self.withdraw[:3, 3] += direction*.13
            if self.withdraw_rotation is not None:
                self.withdraw[:3,:3]=self.withdraw_rotation
            self._transition('withdraw')
        elif stage == 'withdraw' and reached and not s['robot_object_contact']:
            self._transition('hold')
        self._update_hold()
        return self.stage

    def _build_stage_path(self):
        s, stage = self.state, self.stage
        if stage == 'approach':
            self.grasp_goal = s['object_pose'] @ self.grasp_offset
            self.retreat_direction = s['object_pose'][:3, :3] @ self.direction
            goal = self.grasp_goal.copy(); goal[:3, 3] += self.retreat_direction*.08
        elif stage == 'align': goal = self.grasp_goal.copy()
        elif stage in ('close', 'verify_grasp'): goal = self.grasp_goal.copy()
        elif stage == 'turn':
            reference = door_turn(hinge_world=self.spec['hinge_world'], axis_hinge=[0, 0, 1],
                actual_tcp_world=s['tcp'], actual_handle_world=s['object_pose'],
                actual_angle_rad=s['hinge_angle_rad'], target_angle_rad=self.spec['target_angle_rad'],
                angle_limits_rad=self.spec['angle_limits_rad'], actual_grasp_valid=bool(s['native_grasp']))
            self.path = reference.tcp_world.copy(); goal = self.path[-1].copy()
        elif stage == 'lift':
            goal = s['tcp'].copy(); goal[2, 3] += .08
        elif stage == 'preinsert': goal = self.preinsert.copy()
        elif stage == 'insert': goal = self.insert.copy()
        elif stage == 'release': goal = self.release.copy()
        elif stage in ('withdraw', 'hold'): goal = self.withdraw.copy()
        else: raise ValueError('UNKNOWN_SOURCE_SKILL_STAGE')
        if self.path is None: self.path = straight_path(s['tcp'], goal)
        self.goal, self.path_start = goal, self.index

    def plan(self):
        if self.stop_reason: raise PermissionError(self.stop_reason)
        if self.index < 0 or self.index % 8 or self.index == 640 or self.last_plan == self.index:
            raise ValueError('ONE_H40_PER_EIGHT_ACTUAL_OBSERVATIONS')
        if self.path is None: self._build_stage_path()
        positions = np.minimum(self.index-self.path_start+np.arange(40), len(self.path)-1)
        self.last_plan = self.index
        return dict(poses=self.path[positions].copy(), opening=np.full(40, CLOSED if self.stage in HELD_STAGES else OPEN),
            offset_s=np.arange(40)*.05, stage=self.stage,
            anchor_pose=self.state['tcp'].copy(), decision_time=self.state['time_s'],
            role='PRIVILEGED_SOURCE_TEACHER_UNCOMMITTED_REFERENCE',
            persistent_path_start_observation=self.path_start,
            path_end_requires_actual_geometry=True, task_success=False)

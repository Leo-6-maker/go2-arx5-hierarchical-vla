"""Bounded contact-aware Teacher for the existing M_FREE EEF contract."""

from collections import deque
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from ..trajectory.so3 import matrix_to_quat, quat_to_matrix
from .fusion_trajectory_adapter import MECHANICAL_TRAVEL_M
from .mobile_continuous_teacher_v2 import rotation_increment, rotation_vector
from .mobile_grasp_budget_v3 import BudgetTeacher
from .multitask_carry_geometry_v6 import world_yaw
from .multitask_contact_geometry_d3 import (
    AttachmentWitness, ContactGeometry, SupportClock, SUPPORT_WAIT_TICKS,
    pose_error, relative_margin, sampling_margin,
)
from .multitask_placement_geometry_d2 import Knot, transform, translate
from .multitask_score_v6 import BASE_MOTION, LIFT, OPEN_FRACTION
from .multitask_staged_placement_d2 import (
    StagedPlacementTeacher, WAYPOINT_TOLERANCE_M, supported_opening_waypoints,
)
from .multitask_teacher_d2 import MAX_PLACE_S, PlacementUnavailable, time_estimate
from .multitask_teacher_v6 import MultitaskTeacher

VERSION = 'MFREE_MULTITASK_TEACHER_D3_R0'
MECHANISMS = (
    'INTERNAL_RELATIVE_CLEARANCE', 'APPROACH_TOPOLOGY', 'EXTERNAL_CARRY_ROUTE',
    'SUPPORTED_RELEASE', 'DESCEND_CLOSE_PREFIX', 'MOBILE_HOLD_SIMULTANEITY',
)
APPROACH_STAGES = ('PREGRASP', 'DESCEND', 'CLOSE')
HELD_STAGES = ('LIFT', 'HOLD', 'CARRY', 'LOWER', 'SUPPORTED', 'RESEAT')
OPEN_STAGES = ('OPEN', 'RELEASED', 'WITHDRAW', 'SETTLE')
APPROACH_SWING_YAW_DEG = (-38., -30., -20., -12., 12., 20., 30., 38.)
APPROACH_TILTS_DEG = ((-12., 0.), (0., 0.), (12., 0.),
                      (0., -18.), (0., -9.), (0., 9.), (0., 18.))
APPROACH_SIDE_STEPS_M = (.016, .028)
APPROACH_GRASP_OFFSETS_M = (0., .002, .004, .006, .008)
CARRY_SIDE_STEPS_M = (.018, .032)
HOLD_FEEDBACK_LIMIT_M = .015
RESEAT_LIMIT_M = .001
ATTACHMENT_DRIFT_LIMIT_M = .010


@dataclass(frozen=True)
class RouteWaypoint:
    position: np.ndarray
    rotation: np.ndarray
    descent: np.ndarray
    grasp_offset: np.ndarray


def horizontal_away(selected, other):
    delta = np.asarray(selected)[:2, 3] - np.asarray(other)[:2, 3]
    distance = np.linalg.norm(delta)
    if distance < 1e-9:
        raise PlacementUnavailable('D3_NO_DECLARED_LATERAL_SEPARATION', {})
    return np.r_[delta / distance, 0.]


class ContactTeacher(StagedPlacementTeacher):
    def __init__(self, initial, condition, geometry, instruction, model, *, ablated=None):
        super().__init__(initial, condition, geometry, instruction, model)
        if ablated not in (*MECHANISMS, None):
            raise ValueError('D3_ONLY_PREREGISTERED_SINGLE_MECHANISM_ABLATIONS')
        self.ablated = ablated
        self.family = condition['family']
        self.shadow = ContactGeometry(model, self.task.color)
        if self.task.operation == 'PLACE':
            self.placement_geometry = self.shadow
        self.other_color = 'blue' if self.task.color == 'red' else 'red'
        self.support = SupportClock(self.task.color, geometry)
        self.support_until = None
        self.reseat_used = False
        self.open_elapsed_ticks = 0
        self.approach_waypoint = None
        self.descent_waypoint = None
        self.grasp_offset = np.zeros(3)
        self.carry_waypoint = None
        self.carry_side_target = None
        self.pregrasp_ready = False
        self.root_history = deque(maxlen=2)
        self.held_relative_history = deque(maxlen=4)
        self.previous_actual = None
        self.attached_baseline = None
        self.hold_displacement_m = 0.

    def enabled(self, name):
        return self.ablated != name

    def observe_native(self, sample):
        self.support.observe(sample)

    def departure(self, sample):
        return horizontal_away(sample['object_pose'], sample['objects'][self.other_color]['pose'])

    def approach_options(self, sample):
        yield self.grasp_rotation, None
        if self.family == 'F2' and self.task.color == 'red' and self.enabled('APPROACH_TOPOLOGY'):
            away = self.departure(sample)
            center = np.asarray(sample['object_pose'])[:3, 3] + [0., -.0017, .080]
            for yaw, offset, (roll, pitch) in product(
                    APPROACH_SWING_YAW_DEG, APPROACH_SIDE_STEPS_M, APPROACH_TILTS_DEG):
                local_tilt = rotation_increment(np.deg2rad([roll, pitch, 0.]))
                rotation = world_yaw(np.deg2rad(yaw)) @ self.grasp_rotation @ local_tilt
                side = center + away * offset
                for bias in APPROACH_GRASP_OFFSETS_M:
                    yield rotation, RouteWaypoint(side, rotation, side - [0., 0., .027], away * bias)

    def carry_options(self, sample):
        yield None
        if self.family == 'F2' and self.task.color == 'blue' and self.enabled('EXTERNAL_CARRY_ROUTE'):
            away = self.departure(sample)
            tangent = np.array([-away[1], away[0], 0.])
            for distance, side in product(CARRY_SIDE_STEPS_M, (0., -1., 1.)):
                yield distance * (away + side * tangent)

    def route_knots(self, sample, route, approach, detour):
        knots = super().approach_knots(sample, route)
        if knots is None:
            return None
        result = list(knots)
        root_at_start = np.asarray(sample['qpos'])[:7].copy()
        tcp_at_start = result[0].tcp
        def root(tcp):
            position = root_at_start.copy()
            position[:2] += tcp[:2, 3] - tcp_at_start[:2, 3]
            return position
        if approach is not None:
            # Recompute the carried attachment from the same capture point
            # that the online descent will use.
            grasp = translate(result[2].tcp, approach.grasp_offset)
            attachment = np.linalg.solve(grasp, result[2].object_pose)
            landing = self.landing_pose(grasp, result[2].object_pose, route)
            if landing is None:
                return None
            lift = translate(grasp, [0., 0., route.transit_raise])
            high = translate(landing, [0., 0., route.transit_raise])
            for i, tcp in ((2, grasp), (3, grasp), (4, lift), (5, high),
                           (6, landing), (7, landing), (8, translate(landing, route.withdrawal))):
                planar = root(landing if i >= 5 else tcp)[:2]
                obj = tcp @ attachment if i in (4, 5, 6) else result[2].object_pose
                if i in (7, 8):
                    obj = landing @ attachment
                result[i] = replace(result[i], tcp=tcp, object_pose=obj, root_xy=planar)
        for index, knot in enumerate(result):
            result[index] = replace(knot, root_xy=np.r_[knot.root_xy, root_at_start[2:]])
        if detour is not None:
            lift = result[4]
            tcp = translate(lift.tcp, detour)
            result.insert(5, Knot(tcp, tcp @ np.linalg.solve(lift.tcp, lift.object_pose),
                                  lift.finger_q, root(tcp), 'carry'))
        if approach is not None:
            anchor = result[0]
            tcp = transform(approach.rotation, approach.position)
            result[1] = Knot(tcp, anchor.object_pose, anchor.finger_q, root(tcp), 'approach')
            down = transform(approach.rotation, approach.descent)
            result.insert(2, Knot(down, anchor.object_pose, anchor.finger_q, root(down), 'approach'))
        return result

    def set_approach(self, approach):
        if approach is not None:
            self.approach_waypoint = approach
            self.descent_waypoint = approach.descent
            self.grasp_offset = approach.grasp_offset

    def check_approach_route(self, sample, knots, adaptive_pinch):
        sampling = sampling_margin(sample)[0]
        result = self.shadow.check(sample, knots, sampling)
        if not adaptive_pinch or result['phase'] not in ('close', 'lift', 'carry', 'lower', 'open', 'withdraw'):
            return result
        index = next(i for i, k in enumerate(knots) if k.stage == 'close')
        fingers = self.shadow.pinch_contact_widths(sample, knots[index-1])
        if fingers is None:
            return dict(result, passed=False, reason='D3_NO_TWO_PAD_GEOMETRIC_CONTACT_RANGE')
        revised = [replace(k, finger_q=fingers) if k.stage in ('close', 'lift', 'carry', 'lower')
                   else k for k in knots]
        return dict(self.shadow.check(sample, revised, sampling), geometric_contact_widths_m=fingers.tolist())

    def select_route(self, sample):
        choices = []
        approaches = tuple(self.approach_options(sample))
        detours = tuple(self.carry_options(sample))
        approach_checks = {}
        for raw, (rotation, bypass), detour in product(self.candidate_routes(), approaches, detours):
            grasp_rotation = (raw.grasp_rotation if bypass is None and raw.grasp_rotation is not None
                              else rotation)
            route = replace(raw, grasp_rotation=grasp_rotation)
            if bypass is not None:
                key = (id(bypass), route.grasp_raise)
                if key not in approach_checks:
                    approach = self.hold_knots(sample, grasp_rotation, bypass, route.grasp_raise)
                    approach_checks[key] = self.check_approach_route(sample, approach, True)
                if not approach_checks[key]['passed']:
                    continue
            knots = self.route_knots(sample, route, bypass, detour)
            if knots is not None and (duration := time_estimate(knots)) <= MAX_PLACE_S:
                choices.append((duration, route, bypass, detour, knots))
        choices.sort(key=lambda option: option[0])
        attempts = []
        for seconds, route, bypass, detour, knots in choices:
            verdict = self.check_approach_route(sample, knots, bypass is not None)
            attempts.append(dict(**verdict, estimated_motion_s=seconds,
                                 approach=None if bypass is None else bypass.position.tolist(),
                                 carry=None if detour is None else detour.tolist()))
            if verdict['passed']:
                self.route = route
                self.raise_m = route.grasp_raise
                self.grasp_rotation = route.grasp_rotation
                self.set_approach(bypass)
                self.carry_waypoint = detour
                self.corridor_evidence = dict(attempts=attempts, root_hypothesis='MEASURED_SE3_WITH_PLANAR_FOLLOW')
                return
        raise PlacementUnavailable('D3_NO_COMPLETE_PREGRASP_CANDIDATE', dict(
            attempts=attempts, rejected_approach_corridors=sum(not r['passed'] for r in approach_checks.values())))

    def hold_knots(self, sample, rotation, approach, height):
        tcp, obj = (np.asarray(sample[key]) for key in ('tcp', 'object_pose'))
        q = np.asarray(sample['qpos'])[:7]
        fingers = np.asarray(sample['finger_q'])
        pre = transform(rotation, obj[:3, 3] + [0., -.0017, .080])
        grasp = transform(rotation, obj[:3, 3] + [0., -.0017, .012 + height]
                           + rotation @ [-.012, 0., 0.])
        if approach is not None:
            grasp = translate(grasp, approach.grasp_offset)
        lifted = translate(grasp, [0., 0., .026])
        closed = np.full(2, .020)
        def root(pose):
            return q + np.r_[pose[:2, 3] - tcp[:2, 3], np.zeros(5)]
        points = [Knot(tcp, obj, fingers, q, 'approach')]
        if approach is not None:
            bypass = transform(approach.rotation, approach.position)
            points.append(Knot(bypass, obj, fingers, root(bypass), 'approach'))
            down = transform(approach.rotation, approach.descent)
            points.append(Knot(down, obj, fingers, root(down), 'approach'))
        else:
            points.append(Knot(pre, obj, fingers, root(pre), 'approach'))
        points.extend((Knot(grasp, obj, fingers, root(grasp), 'approach'),
                       Knot(grasp, obj, closed, root(grasp), 'close'),
                       Knot(lifted, lifted @ np.linalg.solve(grasp, obj), closed, root(lifted), 'lift')))
        return points

    def select_hold(self, sample):
        heights = dict.fromkeys((self.raise_m, .012, .016, .020))
        if self.family == 'F2' and self.task.color == 'red' and self.enabled('APPROACH_TOPOLOGY'):
            heights.update(dict.fromkeys((0., .004)))
        attempts = []
        for (rotation, approach), height in product(self.approach_options(sample), heights):
            knots = self.hold_knots(sample, rotation, approach, height)
            if time_estimate(knots) > 60.:
                continue
            verdict = self.check_approach_route(sample, knots, approach is not None)
            attempts.append(dict(**verdict, grasp_raise_m=height,
                                 approach=None if approach is None else approach.position.tolist()))
            if verdict['passed']:
                self.set_approach(approach)
                self.grasp_rotation = rotation
                self.raise_m = height
                self.corridor_evidence = dict(attempts=attempts)
                return
        raise PlacementUnavailable('D3_NO_HOLD_APPROACH_CORRIDOR', dict(attempts=attempts))

    def follow_approach_waypoint(self, sample):
        waypoint = self.approach_waypoint
        if waypoint is None:
            return None
        tcp = np.asarray(sample['tcp'])
        position_error = np.linalg.norm(tcp[:3, 3] - waypoint.position)
        angle_error = np.linalg.norm(rotation_vector(waypoint.rotation @ tcp[:3, :3].T))
        if position_error <= WAYPOINT_TOLERANCE_M and angle_error <= np.deg2rad(3.):
            self.approach_waypoint = None
            self.stage = 'DESCEND'
            return None
        self.rotation = waypoint.rotation
        return waypoint.position

    def grasp_goal(self):
        return super().grasp_goal() + self.grasp_offset

    def grasp_phase_goal(self, sample):
        if self.descent_waypoint is not None and self.stage == 'DESCEND':
            if np.linalg.norm(sample['tcp'][:3, 3] - self.descent_waypoint) > WAYPOINT_TOLERANCE_M:
                return self.descent_waypoint
            self.descent_waypoint = None
        return None

    def hold_goal(self, sample):
        goal = BudgetTeacher.goal(self, sample)
        if (self.stage == 'HOLD' and sample['native_grasp']
                and self.enabled('MOBILE_HOLD_SIMULTANEITY')):
            actual = np.asarray(sample['base_pose'])[:2, 3]
            deficit = BASE_MOTION - float((actual - self.base_origin) @ self.direction)
            if deficit > 0.:
                self.hold_displacement_m = max(self.hold_displacement_m,
                                               min(HOLD_FEEDBACK_LIMIT_M, deficit + WAYPOINT_TOLERANCE_M))
            goal = goal + np.r_[self.hold_displacement_m * self.direction, 0.]
        return goal

    def release_goal(self, sample):
        tick = sample['native_step']
        if self.stage in ('SUPPORTED', 'RESEAT'):
            if self.support.confirmed(tick):
                self.open_start = tick - self.open_elapsed_ticks
                self.stage = 'OPEN'
                return self.placement_TCP
            if tick >= self.support_until:
                if self.stage == 'SUPPORTED' and sample['native_grasp'] and not self.reseat_used:
                    return self.reseat(sample, opened=False)
                raise PlacementUnavailable('D3_STABLE_SUPPORT_NOT_REACHED', dict(reseat=self.reseat_used))
            return self.placement_TCP
        if self.stage == 'OPEN':
            obj = sample['objects'][self.task.color]
            if not self.support.supported:
                if not sample['native_grasp'] or self.reseat_used:
                    raise PlacementUnavailable('D3_SUPPORT_LOST_NO_ADMISSIBLE_RESEAT',
                                               dict(grasp=sample['native_grasp'], reseat=self.reseat_used))
                return self.reseat(sample, opened=True)
            if ((tick - self.open_start) * .002 >= 2.
                    and sample['gripper_open_fraction'] >= OPEN_FRACTION
                    and not obj['native_grasp'] and not obj['hand_contact'] and self.support.confirmed(tick)):
                self.stage = 'RELEASED'
            return self.placement_TCP
        if self.stage == 'RELEASED':
            if not self.support.supported or sample['objects'][self.task.color]['hand_contact']:
                raise PlacementUnavailable('D3_RELEASED_OBJECT_NOT_SUPPORTED', {})
            self.withdrawal_goal = self.placement_TCP + self.withdrawal_direction
            self.stage = 'WITHDRAW'
        return super().goal(sample)

    def reseat(self, sample, *, opened):
        tick = sample['native_step']
        self.reseat_used = True
        self.open_elapsed_ticks = tick - self.open_start if opened else 0
        self.open_start = None
        self.stage = 'RESEAT'
        self.support_until = tick + SUPPORT_WAIT_TICKS
        self.placement_TCP = np.asarray(sample['tcp'])[:3, 3] - [0., 0., RESEAT_LIMIT_M]
        return self.placement_TCP

    def goal(self, sample):
        is_place = self.task.operation == 'PLACE'
        if not self.pregrasp_ready:
            if is_place:
                self.select_route(sample)
            elif self.enabled('DESCEND_CLOSE_PREFIX'):
                self.select_hold(sample)
            self.pregrasp_ready = True
        if self.stage == 'PREGRASP':
            goal = self.follow_approach_waypoint(sample)
            if goal is not None:
                return goal
        self.rotation = self.grasp_rotation if self.stage in APPROACH_STAGES else self.rotation
        goal = self.grasp_phase_goal(sample)
        if goal is not None:
            return goal
        if not is_place:
            return self.hold_goal(sample)
        if self.stage in ('SUPPORTED', 'RESEAT', 'OPEN', 'RELEASED') and self.enabled('SUPPORTED_RELEASE'):
            return self.release_goal(sample)
        goal = super().goal(sample)
        if self.stage == 'CARRY' and self.carry_waypoint is not None:
            tcp = np.asarray(sample['tcp'])
            if self.reposition_up is None:
                if self.carry_side_target is None:
                    self.carry_side_target = tcp[:3, 3] + self.carry_waypoint
                if np.linalg.norm(tcp[:3, 3] - self.carry_side_target) > WAYPOINT_TOLERANCE_M:
                    self.rotation = self.grasp_rotation
                    return self.carry_side_target
                self.carry_waypoint = None
        if self.stage == 'OPEN' and self.enabled('SUPPORTED_RELEASE'):
            self.open_start = None
            self.stage = 'SUPPORTED'
            self.support_until = sample['native_step'] + SUPPORT_WAIT_TICKS
        return goal

    def grip_trajectory(self, start):
        if self.stage == 'RESEAT':
            u = np.clip(self.open_elapsed_ticks * .002 / 2., 0., 1.)
            return np.full(40, u**3 * (10. + u * (-15. + 6. * u)))
        return super().grip_trajectory(start)

    def planned_prefix(self, sample, program):
        tcp, obj = (np.asarray(sample[k]) for k in ('tcp', 'object_pose'))
        attached = sample['native_grasp'] and self.stage in HELD_STAGES
        attachment = np.linalg.solve(tcp, obj)
        stage = {'PREGRASP': 'approach', 'DESCEND': 'approach', 'SUPPORTED': 'lower',
                 'RESEAT': 'lower', 'RELEASED': 'open', 'SETTLE': 'withdraw'}.get(self.stage, self.stage.lower())
        if stage == 'hold':
            stage = 'lift'
        fingers = np.asarray(sample['finger_q'])
        root = np.asarray(sample['qpos'])[:7]
        knots = [Knot(tcp, obj, fingers, root, stage)]
        for row in program.decoded_H40[:11]:
            command = matrix10(row)
            knots.append(Knot(command, command @ attachment if attached else obj, fingers, root, stage))
        if self.stage in OPEN_STAGES:
            wire = program.reference_path[:11, 9] * MECHANICAL_TRAVEL_M
            return supported_opening_waypoints(knots, wire, sample['objects'][self.task.color]['table_contact'])
        return knots

    def observed_root_sweep(self, knots):
        head, tail = self.root_history
        if np.array_equal(head, tail):
            return knots
        direction = quat_to_matrix(tail[[4, 5, 6, 3]]) @ quat_to_matrix(head[[4, 5, 6, 3]]).T
        turn = rotation_vector(direction)
        result = []
        for index, knot in enumerate(knots):
            amount = max(index - 1, 0) / 10.
            q = knot.root_xy.copy()
            q[:3] += amount * (tail[:3] - head[:3])
            R = rotation_increment(amount * turn) @ quat_to_matrix(q[[4, 5, 6, 3]])
            q[3:7] = matrix_to_quat(R)[[3, 0, 1, 2]]
            result.append(replace(knot, root_xy=q))
        return result

    def check_approach_atom(self, sample, previous, program, atom):
        """Recheck the next existing P10 atom at its actual 50 ms boundary."""
        if self.stage not in APPROACH_STAGES or not self.enabled('DESCEND_CLOSE_PREFIX'):
            return
        tcp = np.asarray(sample['tcp'])
        obj = np.asarray(sample['object_pose'])
        fingers = np.asarray(sample['finger_q'])
        root = np.asarray(sample['qpos'])[:7]
        phase = 'close' if self.stage == 'CLOSE' else 'approach'
        nominal, following = map(matrix10, program.decoded_H40[atom:atom+2])
        next_tcp = transform(following[:3, :3] @ nominal[:3, :3].T @ tcp[:3, :3],
                             tcp[:3, 3] + following[:3, 3] - nominal[:3, 3])
        anchor = Knot(tcp, obj, fingers, root, phase)
        sampling, _ = sampling_margin(sample)
        endpoint = Knot(next_tcp, obj, fingers, root, phase)
        paths = [[anchor, endpoint]]
        if previous is not None:
            if sample['native_step'] - previous['native_step'] != 25:
                raise ValueError('D3_CONTACT_ATOM_REQUIRES_PREVIOUS_CONFIRMED_50MS')
            before = np.asarray(previous['tcp'])
            old_root = np.asarray(previous['qpos'])[:7]
            previous_rotation = quat_to_matrix(old_root[[4, 5, 6, 3]])
            rotation = quat_to_matrix(root[[4, 5, 6, 3]])
            forecast_root = root.copy()
            forecast_root[:3] += root[:3] - old_root[:3]
            forecast_root[3:7] = matrix_to_quat(rotation @ previous_rotation.T @ rotation)[[3, 0, 1, 2]]
            forecast_tcp = transform(tcp[:3, :3] @ before[:3, :3].T @ tcp[:3, :3],
                                     tcp[:3, 3] + tcp[:3, 3] - before[:3, 3])
            forecast_fingers = np.clip(2. * fingers - previous['finger_q'], 0., MECHANICAL_TRAVEL_M)
            paths.append([anchor, Knot(forecast_tcp, obj, forecast_fingers, forecast_root, phase)])
        for waypoints in paths:
            verdict = self.shadow.check(sample, waypoints, sampling)
            if not verdict['passed']:
                raise PlacementUnavailable('D3_MEASURED_APPROACH_ATOM_CLEARANCE',
                                           dict(verdict, atom=atom, prior_actual_50ms=previous is not None))

    def decide(self, sample, reference, base):
        self.root_history.append(np.asarray(sample['qpos'])[:7].copy())
        witness = AttachmentWitness.measured(sample)
        sampling, radius = sampling_margin(sample)
        attached = sample['native_grasp']
        if not attached:
            self.attached_baseline = None
            self.held_relative_history.clear()
        elif self.attached_baseline is None:
            self.attached_baseline = np.linalg.solve(witness.tcp, witness.object_pose)
        local_attachment = np.linalg.solve(witness.tcp, witness.object_pose)
        if self.previous_prediction is not None:
            displacement = np.linalg.norm(witness.tcp[:3, 3] - self.previous_prediction[:3, 3])
            angle = np.linalg.norm(rotation_vector(witness.tcp[:3, :3] @ self.previous_prediction[:3, :3].T))
            self.residuals.append((float(displacement), float(angle)))
            if attached and self.previous_actual[1]:
                self.held_relative_history.append(self.previous_actual[0].relative_residual(
                    witness, self.previous_prediction, radius))
        program, evidence = MultitaskTeacher.decide(self, sample, reference, base)
        knots = self.planned_prefix(sample, program)
        margin = self.margin(witness.tcp, witness.object_pose)
        relative = None
        if (attached and self.stage in HELD_STAGES and self.held_relative_history
                and self.enabled('INTERNAL_RELATIVE_CLEARANCE')):
            drift = pose_error(local_attachment, self.attached_baseline, radius)
            if drift > ATTACHMENT_DRIFT_LIMIT_M:
                raise PlacementUnavailable('D3_HELD_OBJECT_RELATIVE_MOTION_OUTSIDE_DOMAIN', {'drift_m': drift})
            relative = relative_margin(sampling, self.held_relative_history, radius)
        checks = []
        if self.stage not in APPROACH_STAGES or self.enabled('DESCEND_CLOSE_PREFIX'):
            if self.stage in APPROACH_STAGES:
                margin = sampling
            checks.append(self.shadow.check_relative(sample, knots, margin, relative))
            if checks[-1]['passed'] and len(self.root_history) == 2 and self.stage in APPROACH_STAGES:
                checks.append(self.shadow.check_relative(sample, self.observed_root_sweep(knots), margin, None))
            if not checks[-1]['passed']:
                raise PlacementUnavailable('D3_FILTERED_P10_GEOMETRY_REJECTION',
                                           dict(checks=checks, stage=self.stage))
        self.previous_actual = (witness, attached)
        self.previous_prediction = knots[-1].tcp
        evidence.update(teacher_version=VERSION, d2_full_route=self.corridor_evidence,
                        d2_remaining_placement=self.remaining_evidence,
                        d2_filtered_P10=checks[-1] if checks else None,
                        d3_contact_audit_only=dict(ablated=self.ablated, support_frames=self.support.streak,
                            supported=self.support.supported, reseat=self.reseat_used,
                            margin_m=margin, internal_margin_m=relative, prefix_checks=checks,
                            controller_selected_reference='UNKNOWN'))
        self.corridor_evidence = self.remaining_evidence = None
        return program, evidence

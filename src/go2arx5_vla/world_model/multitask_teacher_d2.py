"""Teacher-only full placement planning with a fresh swept check before P10."""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from .mobile_continuous_teacher_v2 import ANGULAR_SPEED, TRANSLATION_SPEED, rotation_vector
from .mobile_grasp_budget_v3 import BudgetTeacher
from .multitask_carry_geometry_v6 import world_yaw
from .multitask_placement_geometry_d2 import (
    Knot, PlacementGeometry, OPEN_QPOS_M, POSITION_STEP_M, ANGLE_STEP_RAD,
    landing_tcp, remaining_placement, transform, translate,
)
from .multitask_score_v6 import BASE_MOTION, LIFT, OPEN_FRACTION, footprint_inside
from .multitask_teacher_v6 import MultitaskTeacher

VERSION = 'MFREE_MULTITASK_TEACHER_D2_WHOLE_PLACEMENT'
MAX_PLACE_S = 90.
WITHDRAWALS = ((0., 0., .060), (-.020, 0., .056), (0., -.020, .056), (0., .020, .056))
CANDIDATE_YAW_OFFSETS = ((0., (0., 0.)), (-20., (.008, .008)), (20., (0., 0.)),
                         (30., (.004, -.004)), (0., (-.008, .008)), (-12., (.008, -.008)))


class PlacementUnavailable(ValueError):
    """Terminal Teacher feasibility result at the current exact paused clock."""

    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.evidence = evidence


@dataclass(frozen=True)
class Route:
    rotation: np.ndarray
    zone_xy: np.ndarray
    grasp_raise: float
    transit_raise: float
    withdrawal: np.ndarray
    grasp_rotation: np.ndarray | None = None


def time_estimate(knots):
    motion = sum(max(np.linalg.norm(a.tcp[:3,3]-b.tcp[:3,3])/TRANSLATION_SPEED,
                     np.linalg.norm(rotation_vector(b.tcp[:3,:3]@a.tcp[:3,:3].T))/ANGULAR_SPEED)
                 for a,b in zip(knots, knots[1:]))
    # Command-time estimate with simultaneous translation/rotation. CLOSE
    # and OPEN each reserve the original two-second finger profile.
    return float(motion + 5.)


class PlacementTeacher(MultitaskTeacher):
    open_qpos_m = OPEN_QPOS_M

    def __init__(self, initial, condition, geometry, instruction, model):
        super().__init__(initial, condition, geometry, instruction)
        self.placement_geometry = PlacementGeometry(model, self.task.color) if self.task.operation=='PLACE' else None
        self.start = initial['native_step']
        self.grasp_rotation = self.rotation.copy()
        self.zone = geometry['zones'].get(self.task.destination)
        self.route = None
        self.previous_prediction = None
        self.residuals = deque(maxlen=4)
        self.corridor_evidence = None
        self.remaining_evidence = None
        self.relocated = False

    def margin(self, pose, object_pose):
        # Half of the declared geometric sample spacing plus the recent
        # closed-loop translation/rotation residual at matched P10 endpoints.
        # The radial lever arm covers the measured tool-to-object offset.
        # This observed sensitivity allowance is not a dynamics error bound.
        radius = np.linalg.norm(pose[:3,3]-object_pose[:3,3]) + .025
        recent = max((delta + radius*angle for delta,angle in self.residuals), default=0.)
        return float(POSITION_STEP_M/2 + radius*ANGLE_STEP_RAD/2 + recent)

    def candidate_routes(self):
        # Before grasp these are pose hypotheses. Their target attachment will
        # be recomputed from the actual two-finger grasp on every held query.
        for yaw, shift in CANDIDATE_YAW_OFFSETS:
            rotation = world_yaw(np.deg2rad(yaw)) @ self.grasp_rotation
            xy = np.asarray(self.zone['center_world_XY']) + shift
            for lift in (.026, .041, .062):
                for raise_m in dict.fromkeys((self.raise_m, .016, .008)):
                    yield Route(rotation, xy, raise_m, lift, np.asarray(WITHDRAWALS[0]))

    def landing_pose(self, tcp, obj, route):
        attachment = np.linalg.solve(tcp, obj)
        half = self.geometry['object_half_sizes'][self.task.color]
        landing = landing_tcp(attachment, route.zone_xy, route.rotation, half,
                              self.geometry['table_top_world_z'])
        if not footprint_inside(landing @ attachment, half, self.zone):
            return None
        return landing

    def approach_knots(self, sample, route):
        obj, tcp = (np.asarray(sample[k]) for k in ('object_pose', 'tcp'))
        q = np.asarray(sample['qpos'])
        fingers = np.asarray(sample['finger_q'])
        rotation = self.grasp_rotation if route.grasp_rotation is None else route.grasp_rotation
        grasp_xyz = obj[:3,3] + [0.,-.0017,.012+route.grasp_raise] + rotation @ [-.012,0.,0.]
        grasp = transform(rotation, grasp_xyz)
        landing = self.landing_pose(grasp, obj, route)
        if landing is None:
            return None
        pregrasp = transform(rotation, obj[:3,3]+[0.,-.0017,.080])
        lifted = translate(grasp, [0.,0.,route.transit_raise])
        elevated_landing = translate(landing, [0.,0.,route.transit_raise])
        # Under M_FREE the base pose is not an output. Use planar follow
        # hypotheses to screen configurations, and repeat at the measured
        # mount below before issuing any placement command.
        def root(pose):
            return q[:2] + pose[:2,3]-tcp[:2,3]

        # The measured closure is recomputed at the first native grasp.
        # 20 mm is a geometric witness for the nominal 40 mm box span.
        closed = np.full(2, .020)
        attachment = np.linalg.solve(grasp, obj)
        landed = landing @ attachment
        return (
            Knot(tcp, obj, fingers, q[:2], 'approach'),
            Knot(pregrasp, obj, fingers, root(pregrasp), 'approach'),
            Knot(grasp, obj, fingers, root(grasp), 'approach'),
            Knot(grasp, obj, closed, root(grasp), 'close'),
            Knot(lifted, lifted@attachment, closed, root(lifted), 'lift'),
            Knot(elevated_landing, elevated_landing@attachment, closed, root(landing), 'carry'),
            Knot(landing, landed, closed, root(landing), 'lower'),
            Knot(landing, landed, np.full(2, self.open_qpos_m), root(landing), 'open'),
            Knot(translate(landing,route.withdrawal), landed, np.full(2, self.open_qpos_m), root(landing), 'withdraw'),
        )

    def select_route(self, sample):
        margin = self.margin(np.asarray(sample['tcp']), np.asarray(sample['object_pose']))
        candidates = []
        for route in self.candidate_routes():
            knots = self.approach_knots(sample, route)
            if knots is not None:
                duration = time_estimate(knots)
                if duration <= MAX_PLACE_S:
                    candidates.append((duration, route, knots))
        candidates.sort(key=lambda item:item[0])
        attempts = []
        for duration, route, knots in candidates:
            verdict = self.placement_geometry.check(sample, knots, margin)
            attempts.append(dict(**verdict, nominal_motion_reserve_s=duration,
                                 grasp_raise_m=route.grasp_raise, transit_raise_m=route.transit_raise,
                                 zone_XY=route.zone_xy.tolist(), rotation=route.rotation.tolist()))
            if route.grasp_rotation is not None:
                attempts[-1]['grasp_rotation'] = route.grasp_rotation.tolist()
            if verdict['passed']:
                self.route, self.raise_m = route, route.grasp_raise
                if route.grasp_rotation is not None:
                    self.grasp_rotation = route.grasp_rotation
                self.corridor_evidence = dict(attempts=attempts, base_hypothesis='PLANAR_TCP_FOLLOW')
                return
        raise PlacementUnavailable('D2_NO_COMPLETE_PREGRASP_CANDIDATE',
                                   dict(attempts=attempts, total_ranked_candidates=len(candidates)))

    def check_remaining(self, sample, landing, supported):
        tcp, obj = (np.asarray(sample[k]) for k in ('tcp', 'object_pose'))
        stage = self.stage
        margin = self.margin(tcp, obj)
        route = self.route
        withdrawal_choices = [route.withdrawal]
        if stage in ('CARRY', 'LOWER'):
            withdrawal_choices = list(map(np.asarray, WITHDRAWALS))
        attempts = []
        for direction in withdrawal_choices:
            knots = remaining_placement(tcp, obj, landing, np.asarray(sample['finger_q']),
                                        np.asarray(sample['qpos'])[:2], supported=supported,
                                        withdrawal=direction, open_qpos_m=self.open_qpos_m)
            verdict = self.placement_geometry.check(sample, knots, margin)
            attempts.append(dict(**verdict, withdrawal=direction.tolist()))
            if verdict['passed']:
                self.withdrawal_direction = direction
                break
        self.remaining_evidence = dict(native_step=sample['native_step'], all_sweeps=attempts,
                                       stage=stage, sampled_dynamic_residual_intervals=len(self.residuals))
        return any(x['passed'] for x in attempts)

    def grasp_correction(self, sample, tcp, obj):
        if not sample['native_grasp']:
            raise PlacementUnavailable('D2_NATIVE_ATTACHMENT_LOST_DURING_PLACE', {'stage':self.stage})
        landing = self.landing_pose(tcp, obj, self.route)
        if landing is None:
            raise PlacementUnavailable('D2_MEASURED_ATTACHMENT_OUTSIDE_FROZEN_REGION', {'stage':self.stage})
        return landing

    def reposition(self, sample, tcp, obj, remaining_s):
        """Try a checked, bounded way out before declaring the corridor lost."""
        if self.relocated:
            return None
        self.relocated = True
        route = self.route
        root = np.asarray(sample['qpos'])[:2]
        fingers = np.asarray(sample['finger_q'])
        attachment = np.linalg.solve(tcp,obj)
        spare = max(0.,self.transit_z-tcp[2,3])
        recover = translate(tcp,[0.,0.,min(.016,spare+.006)])
        center = np.asarray(self.zone['center_world_XY'])
        remaining_audit = self.remaining_evidence
        attempts = []
        for offset in ((.008,.008),(-.008,.008),(.008,-.008),(-.008,-.008),(0.,0.)):
            candidate = replace(route,zone_xy=center+offset)
            if np.array_equal(candidate.zone_xy,route.zone_xy):
                continue
            landing = self.landing_pose(tcp,obj,candidate)
            if landing is None:
                continue
            carry = landing.copy()
            carry[2,3] = max(recover[2,3],landing[2,3]+.012)
            prefix = (Knot(tcp,obj,fingers,root,'lift'),
                      Knot(recover,recover@attachment,fingers,root,'lift'),
                      Knot(carry,carry@attachment,fingers,root,'carry'))
            knots = prefix + remaining_placement(carry,carry@attachment,landing,fingers,root,
                                                 supported=False,withdrawal=candidate.withdrawal,
                                                 open_qpos_m=self.open_qpos_m)[1:]
            seconds = time_estimate(knots)-2.  # Grasp is already closed.
            if seconds > remaining_s:
                continue
            verdict = self.placement_geometry.check(sample,knots,self.margin(tcp,obj))
            attempts.append(dict(**verdict,zone_XY=candidate.zone_xy.tolist(),command_time_s=seconds))
            if verdict['passed']:
                self.route = candidate
                self.transit_z = carry[2,3]
                self.stage = 'CARRY'
                self.rotation = tcp[:3,:3]
                self.remaining_evidence = dict(original=remaining_audit,reposition=attempts)
                self.reposition_up = recover[:3,3]
                return self.reposition_up
        self.remaining_evidence = dict(original=remaining_audit,reposition=attempts)
        return None

    def goal(self, sample):
        if self.placement_geometry is None:
            return super().goal(sample)
        if self.route is None:
            self.select_route(sample)
        tcp, obj = (np.asarray(sample[k]) for k in ('tcp','object_pose'))
        zone = self.route.zone_xy
        if self.stage in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD'):
            self.rotation = self.grasp_rotation
            goal = BudgetTeacher.goal(self, sample)
            if self.stage != 'HOLD' or not sample['native_grasp'] or obj[2,3]-self.initial_cube_z < LIFT:
                return goal
            self.stage = 'CARRY'
            self.transit_z = max(tcp[2,3], self.lift_origin[2]+self.route.transit_raise)
            self.reposition_up = None
        if self.stage in ('CARRY','LOWER'):
            landing = self.grasp_correction(sample, tcp, obj)
            if self.reposition_up is not None:
                if np.linalg.norm(tcp[:3,3]-self.reposition_up) > .004:
                    return self.reposition_up
                self.reposition_up = None
            self.rotation = self.route.rotation
            can_enter = (np.linalg.norm(obj[:2,3]-zone) <= .004
                         and np.linalg.norm(rotation_vector(landing[:3,:3] @ tcp[:3,:3].T)) <= np.deg2rad(3.)
                         and (np.asarray(sample['base_pose'])[:2,3]-self.base_origin) @ self.direction >= BASE_MOTION)
            if self.stage == 'CARRY' and not can_enter:
                return np.r_[landing[:2,3], self.transit_z]
            supported = (sample['objects'][self.task.color]['table_contact']
                         and abs(tcp[2,3]-landing[2,3]) <= .005 and can_enter
                         and footprint_inside(obj, self.geometry['object_half_sizes'][self.task.color], self.zone))
            remaining = MAX_PLACE_S-(sample['native_step']-self.start)*.002
            if not self.check_remaining(sample, landing, supported):
                corrective = self.reposition(sample,tcp,obj,remaining)
                if corrective is not None:
                    return corrective
                raise PlacementUnavailable('D2_REMAINING_LOWER_OPEN_WITHDRAW_CLEARANCE',
                                           self.remaining_evidence)
            # Resolve early enough to finish jaw release, withdraw and settle.
            required = (np.linalg.norm(tcp[:3,3]-landing[:3,3])/TRANSLATION_SPEED
                        + np.linalg.norm(self.withdrawal_direction)/TRANSLATION_SPEED + 3.)
            if remaining < required:
                raise PlacementUnavailable('D2_INSUFFICIENT_NATIVE_TIME_FOR_COMPLETE_PLACE',
                    dict(remaining_s=remaining, nominal_completion_s=float(required), **self.remaining_evidence))
            self.stage = 'LOWER'
            if not supported:
                return landing[:3,3]
            self.stage = 'OPEN'
            self.open_start = sample['native_step']
            self.placement_TCP = tcp[:3,3].copy()
            self.withdrawal_goal = self.placement_TCP+self.withdrawal_direction
        if self.stage == 'OPEN':
            if (sample['native_step']-self.open_start)*.002 < 2. or (
                    sample['gripper_open_fraction'] < OPEN_FRACTION or sample['native_grasp']):
                return self.placement_TCP
            self.stage = 'WITHDRAW'
        if self.stage == 'WITHDRAW' and np.linalg.norm(tcp[:3,3]-self.withdrawal_goal) <= .004:
            self.stage = 'SETTLE'
        return self.withdrawal_goal

    def planned_prefix(self, sample, program):
        tcp, obj = (np.asarray(sample[k]) for k in ('tcp','object_pose'))
        attachment = np.linalg.solve(tcp, obj)
        stage = self.stage.lower()
        before_contact = stage in ('pregrasp', 'descend')
        model_stage = 'approach' if before_contact else stage
        if stage == 'settle':
            model_stage = 'withdraw'
        attached = sample['native_grasp'] and stage in ('lift','hold','carry','lower')
        root = np.asarray(sample['qpos'])[:2]
        fingers = np.asarray(sample['finger_q'])
        knots = [Knot(tcp, obj, fingers, root, model_stage)]
        for row in program.decoded_H40[:11]:
            command = matrix10(row)
            width = np.full(2, OPEN_QPOS_M*row[9]) if stage in ('close','open') else fingers
            knots.append(Knot(command, command@attachment if attached else obj,
                              width, root, model_stage))
        return knots

    def decide(self, sample, reference, base):
        if self.previous_prediction is not None:
            tcp = np.asarray(sample['tcp'])
            predicted = self.previous_prediction
            self.residuals.append((float(np.linalg.norm(tcp[:3,3]-predicted[:3,3])),
                float(np.linalg.norm(rotation_vector(tcp[:3,:3]@predicted[:3,:3].T)))))
        program, evidence = super().decide(sample, reference, base)
        if self.placement_geometry is None:
            evidence['teacher_version'] = VERSION
            return program,evidence
        prefix = self.planned_prefix(sample, program)
        if self.stage in ('LIFT','HOLD','CARRY','LOWER','OPEN','WITHDRAW','SETTLE'):
            margin = self.margin(np.asarray(sample['tcp']), np.asarray(sample['object_pose']))
            verdict = self.placement_geometry.check(sample, prefix, margin)
            if not verdict['passed']:
                raise PlacementUnavailable('D2_FILTERED_NEXT_P10_CLEARANCE', verdict)
        else:
            verdict = None
        self.previous_prediction = prefix[-1].tcp
        evidence.update(teacher_version=VERSION, d2_full_route=self.corridor_evidence,
                        d2_remaining_placement=self.remaining_evidence,
                        d2_filtered_P10=verdict, controller_selected_reference='UNKNOWN')
        self.corridor_evidence = None
        self.remaining_evidence = None
        return program,evidence

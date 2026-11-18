"""D2 candidate that executes the certified lift waypoint and models supported opening."""

from dataclasses import replace

import numpy as np

from .fusion_trajectory_adapter import MECHANICAL_TRAVEL_M, WORKING_OPEN_M
from .multitask_carry_geometry_v6 import world_yaw
from .multitask_score_v6 import LIFT
from .multitask_teacher_d2 import PlacementTeacher, PlacementUnavailable

VERSION = 'MFREE_MULTITASK_TEACHER_D2_STAGED_CALIBRATED'
WAYPOINT_TOLERANCE_M = .004


def supported_opening_waypoints(knots, wire_opening, supported):
    """Sweep from the measured jaw state through a monotone calibrated opening."""
    if not supported:
        raise PlacementUnavailable('D2_SHADOW_OPENING_REQUIRES_MEASURED_SUPPORT', {})
    wire = np.asarray(wire_opening)
    if (wire.shape != (len(knots) - 1,) or not np.isfinite(wire).all()
            or np.any(np.diff(wire) < -1e-12)):
        raise PlacementUnavailable('D2_SHADOW_OPENING_REQUIRES_MONOTONE_CALIBRATED_COMMAND', {})
    measured = knots[0].finger_q
    return [
        knots[0],
        *(replace(knot, finger_q=np.maximum(measured, opening))
          for knot, opening in zip(knots[1:], wire, strict=True)),
    ]


class StagedPlacementTeacher(PlacementTeacher):
    open_qpos_m = WORKING_OPEN_M

    def candidate_routes(self):
        # A higher transit is only selected after the complete sampled route
        # still fits the original task budget, including approach and exit.
        for route in super().candidate_routes():
            yield route
            if route.transit_raise == .062:
                yield replace(route, transit_raise=.082)
            if route.transit_raise == .026:
                for yaw in (-12., 12.):
                    yield replace(route, grasp_rotation=world_yaw(np.deg2rad(yaw)) @ self.grasp_rotation)

    def goal(self, sample):
        # Match the graph used in the pre-grasp sweep. The 20 mm task lift
        # alone need not reach the higher checked transport waypoint.
        if (self.route is not None and self.stage in ('LIFT', 'HOLD')
                and self.lift_origin is not None and sample['native_grasp']):
            tcp, obj = np.asarray(sample['tcp']), np.asarray(sample['object_pose'])
            target = self.lift_origin + [0., 0., self.route.transit_raise]
            ready = (tcp[2, 3] >= target[2] - WAYPOINT_TOLERANCE_M
                     and np.linalg.norm(tcp[:2, 3] - target[:2]) <= WAYPOINT_TOLERANCE_M
                     and obj[2, 3] - self.initial_cube_z >= LIFT)
            if not ready:
                self.stage = 'LIFT'
                self.rotation = self.grasp_rotation
                return target
        return super().goal(sample)

    def planned_prefix(self, sample, program):
        knots = super().planned_prefix(sample, program)
        if self.stage not in ('OPEN', 'WITHDRAW', 'SETTLE'):
            return knots
        opening = program.reference_path[:len(knots) - 1, 9] * MECHANICAL_TRAVEL_M
        supported = sample['objects'][self.task.color]['table_contact']
        return supported_opening_waypoints(knots, opening, supported)

    def decide(self, sample, reference, base):
        program, evidence = super().decide(sample, reference, base)
        evidence['teacher_version'] = VERSION
        return program, evidence

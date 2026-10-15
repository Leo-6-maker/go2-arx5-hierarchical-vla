"""Unadopted single-intervention prototype; cannot launch native physics.

This changes V1's whole-pose veto to changed-component admission. It IS a
scientific Teacher contract change, not the previously authorized clock repair.
"""
import numpy as np
from .persistent_approach_draft import PersistentApproachDraft,residual_zero
from go2arx5_vla.trajectory.so3 import geodesic_angle,matrix_to_quat,quat_to_matrix,quat_slerp

VERSION='FULLBODY_COMPONENT_PROGRESS_V2_DRAFT_NOT_AUTHORIZED'
LAUNCH_ALLOWED=False


class ComponentProgressDraft(PersistentApproachDraft):
    def _advance(self,cursor,segment,actual,dt):
        if segment==len(self.route):return cursor,segment
        goal=self.route[segment]
        # Completing an already reached PLANNED waypoint changes no reference
        # value. Actual contact/manipulation transitions remain separate.
        if residual_zero(cursor,goal):return cursor,segment+1
        candidate=cursor.copy();delta=goal[:3,3]-cursor[:3,3];distance=np.linalg.norm(delta)
        proposed=cursor[:3,3]+delta*min(1.,self.config.linear_speed_m_s*dt/max(distance,1e-15))
        position_limit=self.config.linear_speed_m_s*self.config.query_s
        if np.linalg.norm(proposed-actual[:3,3])<=position_limit+1e-10:
            candidate[:3,3]=proposed
        angle=float(geodesic_angle(cursor[:3,:3],goal[:3,:3]))
        q=quat_slerp(matrix_to_quat(cursor[:3,:3]),matrix_to_quat(goal[:3,:3]),
            min(1.,self.config.angular_speed_rad_s*dt/max(angle,1e-15)))
        proposed_rotation=quat_to_matrix(q)
        angular_limit=self.config.angular_speed_rad_s*self.config.query_s
        if geodesic_angle(proposed_rotation,actual[:3,:3])<=angular_limit+1e-10:
            candidate[:3,:3]=proposed_rotation
        if residual_zero(candidate,goal):segment+=1
        return candidate,segment

    def plan(self,**kw):
        result=super().plan(**kw)
        result.update(version=VERSION,launch_allowed=False,
            contract='ONLY_CHANGED_COMPONENTS_REQUIRE_NEW_LEAD_ADMISSION_NO_WHOLE_POSE_GUARANTEE',
            labels='CPU_PROPOSAL_ONLY_NOT_AUTHORIZED_FOR_INSTALLATION')
        return result

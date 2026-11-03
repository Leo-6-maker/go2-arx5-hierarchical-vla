"""Versioned standing object-task score, reusing every original GO2_PLACE_V1 predicate.

The historical full-body BASE-translation task remains unchanged. This offline
wrapper has no dynamic admission or native sensor/export qualification authority.
"""
import numpy as np
from go2arx5_vla.target_sim.scorer import score_episode,pose_matrix,boolean_array

TASK_ID='WM02_STANDING_T1_V1'
DEFINITION=dict(task_id=TASK_ID,underlying_object_task='GO2_PLACE_V1',cube_edge_m=.04,
    table_height_m=.44,goal_height_m=.46,episode_s=32.,sample_dt_s=.05,
    hold_interval_s=[28.,30.],grasp_sustained_s=.3,lift_m=.02,carry_xy_m=.1,
    source_goal_minimum_xy_m=.15,goal_square_half_width_m=.07,
    maximum_height_error_m=.005,maximum_linear_speed_m_s=.02,
    maximum_angular_speed_rad_s=.1,minimum_tcp_clearance_m=.08,
    BASE_translation_goal='NOT_PART_OF_THIS_SEPARATELY_VERSIONED_STANDING_TASK',
    native_safety='UNCHANGED_FULLBODY_MONITOR_REQUIRED',native_sensor_binding='PENDING_E3')


def score_standing_t1(data,*,goal_xy,initial_object_pose,complete=True,evidence_source='CPU_OFFLINE_NOT_NATIVE_QUALIFICATION'):
    n=len(data['time_s'])
    base=pose_matrix(data['base_pose'],'base_pose')
    if base.shape!=(n,4,4):raise ValueError('Actual free-body pose timeline is required')
    safety=boolean_array(data['native_fullbody_safety_valid'],(n,),'native_fullbody_safety_valid')
    original=score_episode(data,goal_center=[*goal_xy,.46],initial_object_pose=initial_object_pose,
        table_height_m=.44,complete=complete,evidence_source=evidence_source)
    # Preserve all common object/contact/gripper/clock/safety rules, add full-body safety.
    result=dict(original);result.update(task_id=TASK_ID,underlying_object_task_id=original['task_id'],
        native_fullbody_safety_valid=bool(safety.all()),persistent_success=bool(original['persistent_success']and safety.all()),
        maximum_BASE_translation_m=float(np.linalg.norm(base[:,:3,3]-base[0,:3,3],axis=1).max()),
        historical_fullbody_BASE_translation_score_changed=False,native_qualified=False,
        native_sensor_binding='PENDING_E3')
    return result

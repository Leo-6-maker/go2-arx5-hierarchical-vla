"""Enter the existing routed Teacher from an actual held book, never approach."""
from copy import deepcopy
import math
import numpy as np
from .libero_book_teacher import teacher_for_card,RoutedLiberoBookTeacher


def from_held_state(card,state,*,source_index,grasp_since,initial_object_z):
    if(type(source_index)is not int or not 8<=source_index<=592 or source_index%8
            or state.get('safety_valid')is not True or state.get('native_grasp')is not True
            or state.get('native_abort')or not state.get('left_object_contact')
            or not state.get('right_object_contact')or not state.get('opposing_contact')
            or not state.get('relative_motion_valid')or not state.get('gripper_state_valid')
            or not all(math.isfinite(v)for v in(grasp_since,initial_object_z,state['time_s']))
            or not 0<=state['time_s']-grasp_since<=32):
        raise PermissionError('ACTUAL_SAFE_BILATERAL_HELD_BOOK_AT_SOURCE_PREFIX_BOUNDARY')
    teacher=teacher_for_card(card)
    if not isinstance(teacher,RoutedLiberoBookTeacher):raise PermissionError('EXISTING_ROUTED_RELEASE_TEACHER_REQUIRED')
    # Keep the absolute source observation clock and actual grasp transform.
    # No simulator/actuator state is touched by the Teacher initializer.
    teacher.index=source_index-1;teacher.initial_time=state['time_s']-source_index*.05
    teacher.grasp_since=grasp_since;teacher.lift_initial_object_z=initial_object_z
    teacher.grasp_goal=np.asarray(state['tcp']).copy()
    teacher.retreat_direction=np.asarray(state['object_pose'])[:3,:3]@teacher.direction
    teacher.state=deepcopy(state)
    from .object_relations import pose
    teacher.state['tcp']=pose(state['tcp']);teacher.state['object_pose']=pose(state['object_pose'])
    rise=float(np.asarray(state['object_pose'])[2,3]-initial_object_z)
    if rise<.06:
        teacher.stage='lift'if state['time_s']-grasp_since>=.3-1e-10 else'verify_grasp'
    else:
        teacher.preinsert,teacher.insert=teacher._book_targets();teacher.stage='preinsert'
        from itertools import product
        size=np.asarray(teacher.spec['book_size_m'])/2;book=np.asarray(state['object_pose'])
        points=np.asarray(list(product((-1.,1.),repeat=3)))*size@book[:3,:3].T+book[:3,3]
        r=teacher.route;gap=r['clearance_m']
        lateral=points[:,1].min()>=r['corridor_y_min_m']+gap and points[:,1].max()<=r['corridor_y_max_m']-gap
        front=points[:,0].min()>=r['front_wall_max_x_m']+gap
        above=points[:,2].min()>=r['front_wall_top_z_m']+gap
        teacher.waypoint_index['preinsert']=2 if lateral and front else 1 if lateral and above else 0
    teacher.events.append(dict(event='ACTUAL_HELD_SOURCE_INITIALIZATION',observation_index=source_index,
        initial_teacher_stage=teacher.stage,actual_grasp=True,source_initial_object_z=initial_object_z,
        actual_source_grasp_since=grasp_since,no_reset=True))
    return teacher

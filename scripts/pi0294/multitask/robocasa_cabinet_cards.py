"""Declared cabinet port conditions; unexecuted and not original test seeds."""
from copy import deepcopy
import numpy as np
from .robocasa_cabinet_scene import VERSION
from .libero_book_cards import cards as book_cards


def cards():
    cabinet=np.eye(4);cabinet[:3,:3]=[[0,1,0],[-1,0,0],[0,0,1]]
    cabinet[:3,3]=[.53,0.,.59]  # original source table .44 plus half-height .15
    size=[.24,.22,.30];height=size[2]-.003;length=min(.24,height-.08)
    grasp=np.eye(4);grasp[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]]
    home=book_cards()['left']['initial_tcp'];result={}
    for orientation,direction in [('right',1),('left',-1)]:
        local=np.array([direction*(size[0]/2-.015),-size[1]/2,0.])
        hinge=cabinet.copy();hinge[:3,3]+=cabinet[:3,:3]@local
        result[orientation]=dict(id='NIGHT30_ROBOCASA_'+orientation.upper()+'_PORT01',task='DOOR',
            source_port=VERSION,task_duration_s=32.,warmup_s=5.,
            initial_tcp=deepcopy(home),source_base_card=dict(id='PINNED_CABINET_PORT_BASE',source_xy=[.36,-.10],
                goal_xy=[.53,0.],cube_yaw=0.),
            cabinet=dict(size_m=list(size),orientation=orientation,cabinet_world=cabinet.tolist()),
            door=dict(initial_angle_rad=0.,target_angle_rad=direction*.95*1.57,
                angle_limits_rad=[0.,1.57]if direction==1 else[-1.57,0.],
                hinge_world=hinge.tolist(),handle_radius_m=.013,handle_length_m=length,
                object_to_tcp_grasp=grasp.tolist(),approach_direction_object=[0.,-1.,0.]),
            instruction='Open the cabinet door.',original_benchmark_robot_or_seed=False,
            independent_demo_conditions=dict(native_all_doors_90_percent=True,
                actual_hinge_motion_and_handle_grasp=True,gripper_released=True,
                no_robot_object_contact=True,withdrawal_m=.08,
                angular_speed_max_rad_s=.05,continuous_hold_s=2.))
    return result

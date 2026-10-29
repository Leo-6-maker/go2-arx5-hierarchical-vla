"""Predeclared source-port poses and duration, never original benchmark seeds."""
import numpy as np
from .libero_book_scene import VERSION


def cards():
    from scripts.pi0291_arx5_data.scene import DOWNWARD_ROTATION
    semantic=np.diag([-1.,1.,-1.])
    yaw=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    tool_rotation=yaw@DOWNWARD_ROTATION
    offset=np.eye(4);offset[:3,:3]=semantic.T@tool_rotation;offset[2,3]=-.004
    native=np.eye(4);native[:3,3]=[.36,-.10,.43833]
    fixture=np.eye(4);fixture[:3,3]=[.50,0.,.43861]
    initial=np.eye(4);initial[:3,:3]=tool_rotation;initial[:3,3]=[.36,-.11215,.59]
    result={}
    for side,y in [('left',.14551),('right',-.14339)]:
        target=np.eye(4);target[:3,:3]=semantic
        target[:3,3]=[.50+.00631-.00047,y-.01215,.45075+.06937]
        result[side]=dict(id='NIGHT30_LIBERO_BOOK_'+side.upper()+'_PORT01',task='BOOK',
            source_port=VERSION,goal_side=side,task_duration_s=32.,warmup_s=5.,
            original_benchmark_robot_or_seed=False,
            port_condition='ARX5_FIXED_BASE_UPRIGHT_SINGLE_BOOK_TABLE_TO_ORIGINAL_CADDY',
            initial_book_native_world=native.tolist(),caddy_world=fixture.tolist(),initial_tcp=initial.tolist(),
            source_base_card=dict(id='PINNED_CADDY_PORT_BASE',source_xy=[.36,-.10],goal_xy=[.50631,y],cube_yaw=0.),
            book=dict(book_size_m=[.02868,.11038,.13440],object_to_tcp_grasp=offset.tolist(),
                approach_direction_object=[0.,0.,-1.],target_book_world=target.tolist()),
            instruction='Put the black book in the '+side+' compartment of the desk caddy.',
            independent_demo_conditions=dict(native_LIBERO_In=True,actual_caddy_support=True,
                gripper_released=True,no_robot_object_contact=True,withdrawal_m=.08,
                linear_speed_max_m_s=.02,angular_speed_max_rad_s=.1,continuous_hold_s=2.))
    return result

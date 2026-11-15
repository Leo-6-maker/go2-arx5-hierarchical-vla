import numpy as np
from go2arx5_vla.world_model.mobile_task_episode_v1 import TransportTeacher


def sample():
    tcp=np.eye(4);tcp[:3,3]=[.26,0,.72]
    obj=np.eye(4);obj[:3,3]=[.2565,0,.62]
    return dict(tcp=tcp,object_pose=obj,native_grasp=False,gripper_fraction=.95)


def test_carry_prefix_does_not_enter_transport_before_measured_trigger():
    initial=sample();teacher=TransportTeacher(initial,'F5');state=sample()
    program,evidence=teacher.decide(state,state['tcp'],[0,0,0,0,0,0,1],False)
    assert teacher.stage=='PICK'and evidence['stage']=='PREGRASP'
    assert program.reference_path.shape==(40,10)


def test_given_grasp_transport_keeps_closed_grip_and_original_fifty_ms_program():
    state=sample();state['native_grasp']=True
    teacher=TransportTeacher(state,'F4',held=True)
    program,evidence=teacher.decide(state,state['tcp'],[0,0,0,0,0,0,1],True)
    assert evidence['stage']=='CARRY'
    assert np.all(program.decoded_H40[:,9]==0.)
    assert np.isclose(program.decoded_H40[-1,0]-state['tcp'][0,3],.015)
    assert program.geometry_scale==1 and program.time_stretch_factor==1


def test_placement_waits_for_real_object_height_before_opening():
    initial=sample();teacher=TransportTeacher(initial,'F6',held=True)
    state=sample();state['native_grasp']=True;state['object_pose'][0,3]+=.1;state['object_pose'][2,3]+=.026
    program,evidence=teacher.decide(state,state['tcp'],[0,0,0,0,0,0,1],True)
    assert evidence['stage']=='LOWER'and program.decoded_H40[-1,9]==0
    state['object_pose'][2,3]=initial['object_pose'][2,3]
    program,evidence=teacher.decide(state,state['tcp'],[0,0,0,0,0,0,1],True)
    assert evidence['stage']=='OPEN'and np.isclose(program.decoded_H40[-1,9],1.)
    state['native_grasp']=False;state['robot_object_contact']=False
    program,evidence=teacher.decide(state,state['tcp'],[0,0,0,0,0,0,1],True)
    assert evidence['stage']=='WITHDRAW'

import copy
import numpy as np
import pytest
from go2arx5_vla.world_model.multitask_score_v6 import Score,Task,footprint_inside


def pose(x,y,z):
    p=np.eye(4);p[:3,3]=[x,y,z];return p


def setup(task='PLACE-red-left'):
    objects={c:dict(pose=pose(.4,0 if c=='red'else .12,.02),linear_velocity=np.zeros(3),native_grasp=False,hand_contact=False,table_contact=True)for c in('red','blue')}
    initial=dict(native_step=5703,objects=objects,base_pose=np.eye(4));geometry=dict(zones={z:dict(center_world_XY=[.5,y],half_size_XY=[.05,.04])for z,y in(('left',-.06),('right',.06))},
        direction_world_XY=[1.,0.],object_half_sizes={c:[.02]*3 for c in objects},table_top_world_z=0.)
    scorer=Score(task,initial,geometry)
    sample=dict(native_step=5703,time_s=5703*.002,objects=copy.deepcopy(objects),base_pose=pose(.10,0,0),tcp=pose(.5,-.06,.09),gripper_open_fraction=.95,safety_valid=True)
    sample['objects']['red'].update(pose=pose(.5,-.06,.041),native_grasp=True,hand_contact=True,table_contact=False)
    return scorer,sample


def stream(scorer,sample,first,last):
    for tick in range(first,last+1):
        sample['native_step']=tick;sample['time_s']=tick*.002;scorer.update(sample)
    return scorer.result()


def test_place_requires_real_lift_full_footprint_release_and_continuous_one_second():
    scorer,sample=setup();stream(scorer,sample,5703,5703)
    sample['objects']['red'].update(pose=pose(.5,-.06,.02),native_grasp=False,hand_contact=False,table_contact=True)
    assert not stream(scorer,sample,5704,6203)['task_success']
    assert stream(scorer,sample,6204,6204)['task_success']
    sample['objects']['red']['linear_velocity'][0]=.021
    assert not stream(scorer,sample,6205,6205)['task_success']


@pytest.mark.parametrize('fault',['wrong_object','wrong_destination','no_lift','closed_gripper','contact','no_retreat','airborne','base','gap','unsafe_state'])
def test_clear_negative_place_states_are_not_success(fault):
    scorer,sample=setup()
    if fault=='no_lift':sample['objects']['red']['pose'][2,3]=.02
    if fault=='wrong_object':sample['objects']['blue']['native_grasp']=True
    stream(scorer,sample,5703,5703)
    sample['objects']['red'].update(pose=pose(.5,-.06,.02),native_grasp=False,hand_contact=False,table_contact=True)
    if fault=='wrong_destination':sample['objects']['red']['pose'][1,3]=.06
    elif fault=='closed_gripper':sample['gripper_open_fraction']=.89
    elif fault=='contact':sample['objects']['red']['hand_contact']=True
    elif fault=='no_retreat':sample['tcp']=pose(.5,-.06,.05)
    elif fault=='airborne':sample['objects']['red']['table_contact']=False
    elif fault=='base':sample['base_pose']=pose(.079,0,0)
    elif fault=='unsafe_state':sample['safety_valid']=False
    first=5705 if fault=='gap'else 5704
    result=stream(scorer,sample,first,6306)
    assert not result['task_success']
    assert ('wrong_destination_release' in result['events']) == (fault=='wrong_destination')


def test_region_contains_complete_rotated_box_not_only_center():
    p=pose(0,0,.02);theta=np.pi/4;p[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    assert not footprint_inside(p,[.02]*3,dict(center_world_XY=[0,0],half_size_XY=[.025,.025]))
    assert footprint_inside(p,[.02]*3,dict(center_world_XY=[0,0],half_size_XY=[.03,.03]))


def test_hold_must_still_be_held_and_has_original_time_budget():
    scorer,sample=setup('HOLD-red')
    assert stream(scorer,sample,5703,6203)['task_success']
    sample['objects']['red']['native_grasp']=False
    assert not stream(scorer,sample,6204,6204)['task_success']
    sample['native_step']=35704;sample['time_s']=35704*.002
    with pytest.raises(ValueError,match='HOLD60'):scorer.update(sample)
    with pytest.raises(ValueError):Task.parse('PLACE-red-front')

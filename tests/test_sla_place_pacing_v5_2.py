import copy
import numpy as np
import pytest

from go2arx5_vla.world_model import sla_pick_place_v5 as place
from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert


def initial():
    obj=np.eye(4);obj[:3,3]=[.26,-.014,.62]
    tcp=np.eye(4);tcp[:3,3]=[.25,-.015,.66]
    return dict(object_pose=obj,tcp=tcp,native_grasp=True,gripper_fraction=.43,native_step=14203)


def teacher(cls):
    t=cls(initial());t.stage='TRANSFER';t.grip=0.;t.lift_TCP_z=.66
    if isinstance(t,place.PacedTeacher):t.reference_baseline=np.zeros(3)
    return t


def test_paced_version_keeps_task_score_goal_transport_and_full_native_clock():
    spec=dict(sla_pick_place_v5=dict(version=place.PACED_VERSION),task_learning_role=place.ROLE,
        task_learning_teacher_mode='NORMAL_TEACHER_V7')
    p=expert.native_protocol(spec)
    assert p['task_score']==place.score_contract()
    assert p['goal_offset_world_m']==[0.,.15,0.]
    assert p['prefix_atoms']==10 and p['policy_queries']==240 and p['final_hold_atoms']==40
    assert expert.native_endpoint(spec)==66703
    assert (66703-5703)//25==2440
    assert p['reference_transport']==place.protocol()['reference_transport']
    old=copy.deepcopy(spec);old['sla_pick_place_v5']['version']=place.VERSION
    assert type(place.make_teacher(old,initial()))is place.Teacher
    assert expert.native_protocol(old)==place.protocol()


def test_large_tracking_gap_holds_existing_reference_instead_of_advancing_goal():
    t=teacher(place.PacedTeacher);sample=initial();ref=sample['tcp'].copy();ref[1,3]+=.006
    program,proposal=t.decide(sample,ref,np.array([0,0,0,1,0,0,0.]))
    assert proposal['expert_pacing_hold'] and t.stage=='TRANSFER'
    np.testing.assert_allclose(program.decoded_H40[:,:3],np.tile(sample['tcp'][:3,3],(40,1)),rtol=0,atol=0)
    np.testing.assert_allclose(program.reference_path[:,:3],np.tile(ref[:3,3],(40,1)),rtol=0,atol=1e-15)
    np.testing.assert_allclose(t.goal_center-initial()['object_pose'][:3,3],[0,.15,0],rtol=0,atol=1e-15)
    assert program.geometry_scale==program.time_stretch_factor==1
    assert t.pacing_holds==1 and proposal['student_metadata']is False


def test_within_gap_budget_preserves_original_requested_motion_exactly():
    a=teacher(place.Teacher);b=teacher(place.PacedTeacher);sample=initial();ref=sample['tcp'].copy()
    old,_=a.decide(sample,ref,np.array([0,0,0,1,0,0,0.]))
    new,proposal=b.decide(sample,ref,np.array([0,0,0,1,0,0,0.]))
    np.testing.assert_array_equal(old.decoded_H40,new.decoded_H40)
    np.testing.assert_array_equal(old.reference_path,new.reference_path)
    assert not proposal['expert_pacing_hold'] and b.pacing_holds==0


def test_pacing_never_hides_lost_grasp_or_invents_transfer_baseline():
    sample=initial();sample['native_grasp']=False;ref=sample['tcp'].copy();ref[1,3]+=.01
    with pytest.raises(ValueError):teacher(place.PacedTeacher).decide(sample,ref,np.array([0,0,0,1,0,0,0.]))
    t=teacher(place.PacedTeacher);t.reference_baseline=None
    with pytest.raises(ValueError):t.decide(initial(),ref,np.array([0,0,0,1,0,0,0.]))

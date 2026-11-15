import numpy as np
import pytest

from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import ContinuousTeacher, TRANSLATION_TAU
from scripts.pi0292.eval_contract import matrix10


def sample():
    tcp=np.eye(4);tcp[:3,3]=[.25,0,.7]
    cube=np.eye(4);cube[:3,3]=[.35,0,.62]
    return dict(tcp=tcp,object_pose=cube,native_step=5703,native_grasp=False)


def test_real_p10_commit_preserves_reference_velocity_and_acceleration_with_tracking_error():
    s=sample();teacher=ContinuousTeacher(s);reference=s['tcp'].copy();base=[0,0,.4,0,0,0,1]
    for q in range(20):
        program,e=teacher.decide(s,reference,base)
        assert np.allclose(program.reference_path[0,:3],reference[:3,3],atol=1e-14)
        previous=program.reference_path.copy()
        with pytest.raises(ValueError,match='COMMIT_ONLY_REAL'):teacher.commit(program,s['native_step']+249)
        reference=teacher.commit(program,s['native_step']+250)
        assert np.allclose(reference[:3,3],previous[10,:3],atol=1e-14)
        s['native_step']+=250
        # Observed TCP deliberately follows only part of the installed motion.
        s['tcp'][:3,3]+=.7*(matrix10(program.desired_path[10])[:3,3]-s['tcp'][:3,3])
        next_program,ne=teacher.decide(s,reference,base)
        assert np.allclose(ne['initial_velocity_m_s'],teacher.velocity)
        assert np.allclose(ne['initial_acceleration_m_s2'],(teacher.inner_velocity-teacher.velocity)/TRANSLATION_TAU)
        # Replanning uses the new actual observation, without resetting motion.
        assert np.array_equal(next_program.decoded_H40[0,:3],s['tcp'][:3,3])
        assert np.linalg.norm(previous[10,:3]-previous[9,:3])>0
        assert np.linalg.norm(next_program.reference_path[1,:3]-next_program.reference_path[0,:3])<=.0075*.05+1e-12
    assert np.linalg.norm(teacher.velocity)>.005


def test_query_feedback_and_original_four_mm_transition_not_reference_position():
    s=sample();teacher=ContinuousTeacher(s);reference=s['tcp'].copy();reference[0,3]+=.1
    program,e=teacher.decide(s,reference,[0,0,.4,0,0,0,1])
    assert teacher.stage=='PREGRASP'
    goal=np.asarray(e['privileged_goal']);s['tcp'][:3,3]=goal+[.0041,0,0]
    assert teacher.goal(s).shape==(3,)and teacher.stage=='PREGRASP'
    s['tcp'][:3,3]=goal+[.0039,0,0];teacher.goal(s)
    assert teacher.stage=='DESCEND'


def test_bounded_complete_h40_and_continuous_gripper_phase_without_grasp_fabrication():
    s=sample();s['tcp'][:3,3]=s['object_pose'][:3,3]+[-.012,-.0017,.012]
    teacher=ContinuousTeacher(s);teacher.stage='DESCEND';reference=s['tcp'].copy();base=[0,0,.4,0,0,0,1]
    endings=[]
    for q in range(5):
        p,e=teacher.decide(s,reference,base);p.validate_envelope()
        assert teacher.stage=='CLOSE'and not s['native_grasp']
        assert np.linalg.norm(p.decoded_H40[-1,:3]-p.decoded_H40[0,:3])<=.015
        endings.append(p.decoded_H40[10,9]);reference=teacher.commit(p,s['native_step']+250);s['native_step']+=250
    assert endings[0]>endings[1]>endings[2]>endings[3]==endings[4]==0


def test_feedback_changes_direction_after_measured_object_moves():
    s=sample();t=ContinuousTeacher(s);reference=s['tcp'].copy();base=[0,0,.4,0,0,0,1]
    p,e=t.decide(s,reference,base);reference=t.commit(p,5953);s['native_step']=5953
    s['object_pose'][1,3]=.1
    p2,e2=t.decide(s,reference,base)
    assert e2['privileged_goal'][1]>.09
    assert p2.decoded_H40[10,1]>p2.decoded_H40[0,1]

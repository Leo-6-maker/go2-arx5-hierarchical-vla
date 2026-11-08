import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task


def sample(step=34303,height=.46,grasp=False):
    tcp=np.eye(4);tcp[:3,3]=[.35,-.04,.65]
    cube=np.eye(4);cube[:3,3]=[.425,-.096,height]
    return dict(native_step=step,tcp=tcp,object_pose=cube,native_grasp=grasp,object_linear_velocity=np.zeros(3),
        safety_valid=True,contact_valid=True,gripper_state_valid=True,object_present=True)


def test_teacher_full40_keeps_real_native_clock_and_reference_separate():
    s=sample();reference=s['tcp'].copy();reference[:3,3]+=[.01,.02,.03]
    program,proposal=task.Teacher(s).decide(s,reference,np.array([0,0,0,1,0,0,0]))
    assert program.decoded_H40.shape==(40,10)
    assert np.array_equal(program.decoded_H40[0,:3],s['tcp'][:3,3])
    assert np.allclose(program.reference_path[:,:3]-reference[:3,3],program.decoded_H40[:,:3]-s['tcp'][:3,3])
    assert np.array_equal(program.preview(7)[0],program.reference_path[7])
    assert np.array_equal(program.preview(39)[-1],program.reference_path[-1])
    assert program.knot_indices==tuple(range(40)) and program.time_stretch_factor==1
    assert proposal['no_future_achieved_state_labels']is True


def test_closed_gripper_is_preserved_at_new_h40_boundary():
    s=sample();teacher=task.Teacher(s);teacher.stage='CLOSE';teacher.grip=.40
    s['tcp'][:3,3]=teacher.cube[:3,3]+[0,-.0017,.012]
    program,_=teacher.decide(s,s['tcp'],np.array([0,0,0,1,0,0,0]))
    assert program.decoded_H40[0,9]==.40
    assert program.decoded_H40[-1,9]==pytest.approx(.25)
    assert program.desired_path[0,9]<.5


@pytest.mark.parametrize('duration,grasp,safe,expected',[(20,True,True,True),(19,True,True,False),(20,False,True,False),(20,True,False,False)])
def test_score_requires_real_contiguous_grasp_lift_full_second(duration,grasp,safe,expected):
    rows=[sample(step=34303+25*i,height=.481,grasp=grasp)for i in range(duration+1)]
    if not safe:rows[10]['safety_valid']=False
    assert task.score(rows,.46)['task_success']is expected


def test_score_rejects_missing_native_interval():
    rows=[sample(),sample(step=34353)]
    with pytest.raises(ValueError,match='CONTIGUOUS_SCORE_CLOCK'):task.score(rows,.46)


def test_new_prefix_configuration_has_exact_endpoint_without_weakening_legacy():
    from go2arx5_vla.world_model import stable_task_prefix_v1 as prefix
    spec=dict(phase=task.PHASE,kind=task.KIND,task_learning_grasp=task.protocol(),task_command=task.COMMAND,
        live_skill_start_step=task.B,max_ticks=task.E,physics_dt_s=.002,WBC_changes=False)
    assert len(prefix.observation_steps(spec))==28
    assert prefix.cycle(spec,28)['end_step']==task.E==62803
    broken=copy.deepcopy(spec);broken['max_ticks']-=25
    with pytest.raises(ValueError):prefix.observation_steps(broken)

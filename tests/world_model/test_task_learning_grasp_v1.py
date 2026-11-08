import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task


def sample(step=34303,height=.46,grasp=False):
    tcp=np.eye(4);tcp[:3,3]=[.35,-.04,.65]
    cube=np.eye(4);cube[:3,3]=[.425,-.096,height]
    return dict(native_step=step,tcp=tcp,object_pose=cube,native_grasp=grasp,object_linear_velocity=np.zeros(3),
        safety_valid=True,contact_valid=True,gripper_state_valid=True,object_present=True)


def standing_rows(samples):
    initial=dict(qpos=np.array([0,0,.3,1,0,0,0]),feet=np.zeros((4,3)))
    metrics=task.standing_metrics(initial['qpos'],initial['feet'],['FL','FR','RL','RR'],initial)
    return [dict(metrics,native_step=s['native_step'])for s in samples]


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
    assert task.score(rows,.46,standing=standing_rows(rows))['task_success']is expected


def test_score_rejects_missing_native_interval():
    rows=[sample(),sample(step=34353)]
    with pytest.raises(ValueError,match='CONTIGUOUS_SCORE_CLOCK'):task.score(rows,.46,standing=standing_rows(rows))


def test_new_prefix_configuration_has_exact_endpoint_without_weakening_legacy():
    from go2arx5_vla.world_model import stable_task_prefix_v1 as prefix
    spec=dict(phase=task.PHASE,kind=task.KIND,task_learning_grasp=task.protocol(),task_command=task.COMMAND,
        live_skill_start_step=task.CALIBRATION_B,max_ticks=task.E,physics_dt_s=.002,WBC_changes=False)
    assert len(prefix.observation_steps(spec))==40
    assert prefix.cycle(spec,40)['end_step']==task.E==46203
    assert prefix.cycle(spec,-1)['start_step']==4703
    assert prefix.cycle(spec,-1)['end_step']==prefix.observation_steps(spec)[0]==5703
    assert task.start_readiness_contract()['maximum_BASE_linear_speed_m_s']==.001
    broken=copy.deepcopy(spec);broken['max_ticks']-=25
    with pytest.raises(ValueError):prefix.observation_steps(broken)


def test_start_readiness_omission_is_rejected_before_scene_or_native_startup(monkeypatch):
    from go2arx5_vla.world_model import task_learning_grasp_native_v1 as native
    def scene_must_not_run(spec):raise AssertionError('SCENE_READ_BEFORE_MISSING_BINDING_REJECTION')
    monkeypatch.setattr(native,'scene_admission',scene_must_not_run)
    with pytest.raises(ValueError,match='DECLARED_START_READINESS_BEFORE_ANY_NATIVE_STARTUP'):
        native.validate({})


@pytest.mark.parametrize('speed,passes',[(.0008,True),(.001673157058461498,False)])
def test_explicit_post_hold_clock_keeps_original_velocity_threshold(tmp_path,speed,passes):
    from types import SimpleNamespace
    adapter=SimpleNamespace(spec=dict(task_learning_start_readiness=task.start_readiness_contract()),output=tmp_path)
    actual=dict(native_step=task.B,time_s=task.B*.002,qvel=np.r_[speed,np.zeros(31)])
    if passes:task.require_start_readiness(adapter,actual)
    else:
        with pytest.raises(ValueError,match='POST_HOLD_START_READINESS_REJECTED'):
            task.require_start_readiness(adapter,actual)


def test_continuous_capture_progress_survives_second_decision_and_final_seal(tmp_path):
    import json
    from go2arx5_vla.world_model.task_learning_grasp_native_v1 import write_progress
    path=tmp_path/'TASK_LEARNING_CAPTURES.private.json'
    for count in (1,2):write_progress(path,dict(records=list(range(count)),complete=False))
    write_progress(path,dict(records=[0,1],complete=True))
    assert json.loads(path.read_text())==dict(records=[0,1],complete=True)


def test_new_grasp_standing_score_keeps_short_task_pose_bound_as_diagnostic():
    initial=dict(qpos=np.array([0,0,.3,1,0,0,0]),feet=np.zeros((4,3)))
    base=initial['qpos'].copy();base[0]=.003
    metrics=task.standing_metrics(base,initial['feet'],['FL','FR','RL','RR'],initial)
    assert metrics['passes']is True and metrics['old_short_task_pose_bound_passes']is False
    feet=initial['feet'].copy();feet[0,0]=.0051
    assert task.standing_metrics(base,feet,['FL','FR','RL','RR'],initial)['passes']is False
    assert task.standing_metrics(base,initial['feet'],['FL','FR'],initial)['passes']is False


def test_task_preparation_holds_initial_reference_without_source_alignment():
    from types import SimpleNamespace
    from go2arx5_vla.world_model.task_learning_preparation_v1 import HoldProgram,validate_preview
    from go2arx5_vla.world_model.fusion_trajectory_qualification_contract import hold_preview
    initial=np.eye(4);initial[:3,3]=[.25,0,.71]
    actual=initial.copy();actual[0,3]+=.001
    program=HoldProgram(initial,actual)
    adapter=SimpleNamespace(observe=lambda:dict(step=3703),trajectory_initial_tcp=initial,
        base_pose_hold=[0,0,.3,1,0,0,0],trajectory_initial_base=[0,0,.3,1,0,0,0])
    assert np.array_equal(program.preview(15),hold_preview(initial))
    assert np.array_equal(program.desired_at(39)[:3],actual[:3,3])
    validate_preview(adapter,program.preview(0))
    shifted=program.preview(0);shifted[:,0]+=.001
    with pytest.raises(ValueError,match='EXACT_UNCHANGED_INITIAL_REFERENCE'):validate_preview(adapter,shifted)

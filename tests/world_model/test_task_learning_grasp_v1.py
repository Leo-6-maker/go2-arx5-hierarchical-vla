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
    assert np.linalg.norm(program.decoded_H40[-1,:3]-s['tcp'][:3,3])==pytest.approx(.015)


def test_closed_gripper_is_preserved_at_new_h40_boundary():
    s=sample();teacher=task.Teacher(s);teacher.stage='CLOSE';teacher.grip=.40
    s['tcp'][:3,3]=teacher.grasp_goal()
    program,_=teacher.decide(s,s['tcp'],np.array([0,0,0,1,0,0,0]))
    assert program.decoded_H40[0,9]==.40
    assert program.decoded_H40[-1,9]==pytest.approx(0.)
    assert program.desired_path[0,9]<.5


def test_full_two_second_close_keeps_original_finger_speed_bound():
    s=sample();teacher=task.Teacher(s);teacher.stage='CLOSE'
    s['tcp'][:3,3]=teacher.grasp_goal()
    program,_=teacher.decide(s,s['tcp'],np.array([0,0,0,1,0,0,0]))
    assert program.decoded_H40[0,9]==1. and program.decoded_H40[-1,9]==0.
    assert np.max(np.abs(np.diff(program.reference_path[:,9])))*.044/.05<=.04


def test_grasp_offset_uses_frozen_tool_x_and_separate_world_clearance():
    s=sample();s['tcp'][:3,:3]=np.array([[2**-.5,0,2**-.5],[0,1,0],[-2**-.5,0,2**-.5]])
    teacher=task.Teacher(s)
    assert np.allclose(teacher.grasp_goal()-teacher.cube[:3,3],[-.012*2**-.5,-.0017,.012+.012*2**-.5])


def test_hold_control_keeps_the_existing_reference_without_task_motion():
    s=sample();reference=s['tcp'].copy();reference[:3,3]+=[.012,.004,.024]
    control=task.HoldControl(s)
    for shift in (0.,.0002,-.0001):
        s['tcp'][0,3]+=shift
        program,evidence=control.decide(s,reference,np.array([0,0,.3,1,0,0,0]))
        assert np.allclose(program.reference_path[:,:3],reference[:3,3],atol=1e-15,rtol=0)
        assert np.all(program.decoded_H40[:,9]==1.)
        assert evidence['diagnostic_control']==task.HOLD_CONTROL and evidence['expert_training_eligible']is False


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
    assert len(prefix.observation_steps(spec))==15
    assert prefix.cycle(spec,15)['end_step']==task.E==21203
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


def test_task_local_hold_reaches_continuation_with_real_clock_and_full_consumption_fixture(tmp_path,monkeypatch):
    """Lifecycle fixture only; no simulator, site admission or Teacher claim."""
    import json
    from go2arx5_vla.world_model import task_learning_grasp_native_v1 as native
    from go2arx5_vla.world_model import fusion_trajectory_qualification_native as loop
    from test_fusion_trajectory_qualification import FakeAdapter,setup_flow,reference
    library=tmp_path/'trace-library';library.write_bytes(b'fixture')
    spec=dict(phase=task.PHASE,kind=task.KIND,task_learning_grasp=task.protocol(),task_command=task.COMMAND,
        live_skill_start_step=task.CALIBRATION_B,max_ticks=task.E,physics_dt_s=.002,WBC_changes=False,
        trace_library=str(library))
    episode=dict(schema='TASK_LEARNING_LOCAL_PREPARATION_EPISODE_V4',protocol=task.protocol(),
        source_pose_alignment=False,end_native_step=4703,query_pins=dict(role='NO_MODEL_QUERY'))
    class Adapter(FakeAdapter):
        def install_reference(self,action):
            native.validate_adapter_install(self,action)
            self.active_reference=dict(reference(round(self.step*.002*1e9)/1e9),action_EEF10=action.tolist())
            self.installs.append((self.step,action.copy()));return self.active_reference
    adapter=Adapter(spec,tmp_path);adapter.trace_sealed=False;setup_flow(monkeypatch,adapter)
    monkeypatch.setattr(native,'episode_config',lambda s:(episode,{}))
    monkeypatch.setattr(native,'validate',lambda s:None)
    monkeypatch.setattr(loop,'actual_trace_binding',lambda a:dict(fixture_only=True))
    class AtContinuation(Exception):pass
    def reached(a,report,samples,windows):
        assert a.step==4703 and len(windows)==88 and samples[-1]['native_step']==4703
        assert report['task_local_preparation']['original_steady_prefix_qualified']is True
        assert report['task_local_preparation']['source_pose_alignment']is False
        assert report['all_40_knots_consumed']is True
        assert len(a.reference_history)==88 and len(a.grip_packets)==88
        assert all(np.array_equal(action,a.installs[0][1])for _,action in a.installs)
        assert json.loads((tmp_path/'TRAJECTORY_QUALIFICATION_PREFIX.private.json').read_text())['native_step']==3703
        raise AtContinuation
    monkeypatch.setattr(native,'continue_native',reached)
    with pytest.raises(AtContinuation):native.run(adapter)

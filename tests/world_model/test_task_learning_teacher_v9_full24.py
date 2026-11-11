import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_teacher_v8 as v8
from go2arx5_vla.world_model import task_learning_teacher_e1_v9 as small
from go2arx5_vla.world_model import task_learning_teacher_v9_full24 as full
from go2arx5_vla.world_model import task_learning_teacher_v9_full24_qualification as q
from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as legacy
from go2arx5_vla.world_model.storage import digest

BASE=np.array([0.,0.,.3,1.,0.,0.,0.])


def sample(step=task.B):
    cube=np.eye(4);cube[:3,3]=[.26,-.014,.62];tcp=np.eye(4)
    tcp[:3,3]=cube[:3,3]+[-.012,-.0017,v8.GRASP_Z]
    return dict(native_step=step,time_s=.002*step,tcp=tcp,object_pose=cube,native_grasp=False,
        gripper_fraction=.9545,left_object_contact=False,right_object_contact=False,
        opposing_contact=False,relative_motion_valid=True,safety_valid=True,contact_valid=True,
        gripper_state_valid=True,object_present=True,object_linear_velocity=np.zeros(3))


def consumed(s,g):
    raw=np.r_[s['tcp'][:3,3],[1.,0.,0.,0.,1.,0.],g];installed=raw.copy();installed[9]=(.002+.04*g)/.044
    return dict(native_end_step=s['native_step'],policy_EEF10=raw,installed_consumed_EEF10=installed,solver_gripper_confirmed=True)


def witness(s):
    return dict(native_step=s['native_step'],supported=True,native_grasp=s['native_grasp'],
        actual_mechanical_grip_fraction=s['gripper_fraction'],
        left_object_contact=s['left_object_contact'],right_object_contact=s['right_object_contact'],
        opposing_contact=s['opposing_contact'],relative_motion_valid=s['relative_motion_valid'],
        actual_goal_tool_Y_offset_m=.0024,pad_normal_force_N={side:.050985796667277775
            if s[side+'_object_contact']else 0. for side in ('left','right')},
        actual_TCP_position_m=s['tcp'][:3,3].tolist(),actual_cube_position_m=s['object_pose'][:3,3].tolist(),
        previous_complete_H40_solver_gripper_consumption=True)


@pytest.mark.parametrize('kind',full.TYPES)
@pytest.mark.parametrize('sign',[-1,1])
def test_wrapper_cannot_change_qualified_delegate_actions_at_intervention_entry_or_recovery(monkeypatch,tmp_path,kind,sign):
    s=sample();state={'grip':1.}
    monkeypatch.setattr(v8,'command_context',lambda sample,*args:consumed(sample,state['grip']))
    monkeypatch.setattr(v8,'native_support',lambda *args:True)
    monkeypatch.setattr(small,'current_witness',lambda adapter,sample,*args:witness(sample))
    fields={prefix+'_recovery_condition':kind for prefix in('teacher_full24','teacher_v8','teacher_v9')}
    fields.update({prefix+'_offset_sign':sign for prefix in('teacher_full24','teacher_v8','teacher_v9')})
    wrapped_dir=tmp_path/'wrapped';direct_dir=tmp_path/'direct';wrapped_dir.mkdir();direct_dir.mkdir()
    wrapped=full.NativeQualificationTeacher(SimpleNamespace(spec=fields,output=wrapped_dir),s,lambda:(None,None))
    factory=v8.NativeQualificationTeacher if kind=='UNDER_CLOSE'else small.NativeQualificationTeacher
    direct=factory(SimpleNamespace(spec=fields,output=direct_dir),s,lambda:(None,None))
    wrapped.delegate.oracle.stage=direct.oracle.stage='CLOSE'
    def compare(current):
        a,proof=wrapped.decide(current,current['tcp'],BASE);b,original=direct.decide(current,current['tcp'],BASE)
        assert np.array_equal(a.decoded_H40,b.decoded_H40)and np.array_equal(a.reference_path,b.reference_path)
        assert proof['Teacher_version']==full.VERSION and not proof['delegated_command_changed']
        assert proof['qualified_delegate_Teacher_version']==(v8.VERSION if kind=='UNDER_CLOSE'else small.VERSION)
        return a
    compare(s)
    after=copy.deepcopy(s);after.update(native_step=task.B+1000,time_s=.002*(task.B+1000),gripper_fraction=.54545)
    after['tcp'][1,3]+=.0024
    if kind=='SUPPORTED_SINGLE_PAD':after['left_object_contact']=True
    state['grip']=full.INTERVENTIONS[kind][1];compare(after)
    assert wrapped.entry['E1_condition_established']
    recovered=copy.deepcopy(after);recovered.update(native_step=task.B+2000,time_s=.002*(task.B+2000),
        native_grasp=True,left_object_contact=True,right_object_contact=True,opposing_contact=True)
    state['grip']=.1;compare(recovered)
    assert wrapped.recovered['E2_actual_recovered']and wrapped.native_grasp_after_correction


def test_underclose_uses_original_mechanical_opening_and_side_single_keep_original_gates():
    s=sample();w=witness(s);w['actual_mechanical_grip_fraction']=.499999
    assert not full.entry_condition('UNDER_CLOSE',w)
    w['actual_mechanical_grip_fraction']=.5;assert full.entry_condition('UNDER_CLOSE',w)
    w['native_grasp']=True;assert not full.entry_condition('UNDER_CLOSE',w)
    w['native_grasp']=False;w['supported']=False;assert not full.entry_condition('UNDER_CLOSE',w)
    w['supported']=True;assert full.entry_condition('SUPPORTED_SIDE_OFFSET',w)
    assert not full.entry_condition('SUPPORTED_SINGLE_PAD',w)
    w['left_object_contact']=True;w['pad_normal_force_N']['left']=.050985796667277775
    assert full.entry_condition('SUPPORTED_SINGLE_PAD',w)


@pytest.mark.parametrize('kind',full.TYPES)
def test_full24_has_no_action_when_actual_object_is_unsupported_and_not_grasped(monkeypatch,tmp_path,kind):
    import json
    s=sample()
    monkeypatch.setattr(v8,'command_context',lambda sample,*args:consumed(sample,1.))
    monkeypatch.setattr(v8,'native_support',lambda *args:False)
    monkeypatch.setattr(small,'current_witness',lambda adapter,sample,*args:dict(witness(sample),supported=False))
    spec={prefix+'_recovery_condition':kind for prefix in('teacher_full24','teacher_v8','teacher_v9')}
    spec.update({prefix+'_offset_sign':1 for prefix in('teacher_full24','teacher_v8','teacher_v9')})
    teacher=full.NativeQualificationTeacher(SimpleNamespace(spec=spec,output=tmp_path),s,lambda:(None,None))
    with pytest.raises(full.RecoveryUnavailable,match='UNSUPPORTED_NOT_GRASPED_NO_AIR_OPENING'):
        teacher.decide(s,s['tcp'],BASE)
    event=json.loads((tmp_path/'TEACHER_FULL24_RECOVERY_EXIT.private.json').read_bytes())
    assert event['Teacher_version']==full.VERSION and event['recovery_condition']==kind
    assert not event['intervention_issued']and event['E1']is None and event['E2']is None
    assert not(tmp_path/'TEACHER_FULL24_E2_RECOVERED.private.json').exists()


def test_typed_unavailable_preserves_failure_and_cannot_admit_another_source_role_or_phase():
    row=dict(recovery_condition='UNDER_CLOSE')
    native=dict(status='NATIVE_STAGE_FAILED_PRESERVED',shutdown_clean=True,
        owned_process_exits=[dict(name=n,exit_code=0)for n in('sim','solver','controller','planner')],
        failure='ValueError: TEACHER_V9_FULL24_RECOVERY_UNAVAILABLE:UNSUPPORTED_NOT_GRASPED_NO_AIR_OPENING')
    args=dict(native=native,worker=dict(exit_code=1),lease=dict(status='CLOSED'),
        readiness=dict(passes=True,native_step=task.B),score={},
        spec=dict(task_learning_role=full.ROLE,phase=task.PHASE,source_sha='candidate'),
        expected_source='candidate',attempted=True,recovery_exit=dict(Teacher_version=full.VERSION,
            recovery_condition='UNDER_CLOSE',reason='UNSUPPORTED_NOT_GRASPED_NO_AIR_OPENING'))
    result=q.classify(row,**args)
    assert result['technical_valid']and not result['task_success']
    assert not result['E1_condition_established']and result['E2_actual_recovered']is None
    for field,value in(('task_learning_role',v8.ROLE),('phase','unrelated'),('source_sha','old')):
        changed=copy.deepcopy(args);changed['spec'][field]=value
        assert not q.classify(row,**changed)['technical_valid']


def panel(monkeypatch):
    from go2arx5_vla.world_model import task_learning_collection_v1 as collection
    monkeypatch.setattr(collection,'qualified_base',lambda v:None)
    monkeypatch.setattr(q,'qualified_small4',lambda v:None)
    prereg=Path(__file__).resolve().parents[2]/'experiments/wm02/task_learning_teacher_v9_recovery24_v1.json'
    return dict(schema=q.SCHEMA,task_learning_role=full.ROLE,protocol=full.contract(),source_sha='a'*40,
        full24_controller_sha256=digest(full.__file__),unchanged_V8_source_sha256=digest(v8.__file__),
        unchanged_small4_source_sha256=digest(small.__file__),preserved_V7_fingerprint=legacy.FROZEN_TEACHER,
        preregistration=dict(path=str(prereg),sha256=digest(prereg)),base_Teacher_qualification={},small4_qualification={},
        scene_producers={l:{}for l in q.RECOVERY_OFFSETS},assignments=[dict(native_run=f'full24-{i}',layout=l,restart=r,
            recovery_condition=k,offset_sign=1 if r==1 else -1,Teacher_mode=full.VERSION,parent=q.scene_id(l),
            split='TRAIN',command=task.COMMAND,purpose=q.PURPOSE,model_sha256='b'*64)
            for i,(l,r,k)in enumerate(q.assignments())])


def test_full24_population_and_original22_total7_per_type_thresholds(monkeypatch):
    p=panel(monkeypatch);assert len(q.validate(p))==24
    values=[dict(r,attempted=True,technical_valid=True,hard_safety_stop=False,E1_condition_established=True,
        E2_actual_recovered=True,E3_original_task_success=True,genuine_recovery_success=True)for r in p['assignments']]
    assert q.summarize(p,values)['passes']
    assert not q.summarize(p,values[:23])['passes']
    bad=copy.deepcopy(values);same=[r for r in bad if r['recovery_condition']=='UNDER_CLOSE']
    same[0]['genuine_recovery_success']=same[1]['genuine_recovery_success']=False
    assert q.summarize(p,bad)['genuine_recovery_successes']==22 and not q.summarize(p,bad)['passes']
    bad=copy.deepcopy(values);bad[0]['genuine_recovery_success']=bad[1]['genuine_recovery_success']=False
    assert q.summarize(p,bad)['passes']
    bad[2]['hard_safety_stop']=True;assert not q.summarize(p,bad)['passes']
    with pytest.raises(ValueError):q.summarize(p,values+[values[0]])


def test_unestablished_condition_is_not_a_recovery_failure_denominator(monkeypatch):
    p=panel(monkeypatch);values=[dict(r,attempted=True,technical_valid=True,hard_safety_stop=False,
        E1_condition_established=False,E2_actual_recovered=None,E3_original_task_success=False,
        genuine_recovery_success=False)for r in p['assignments']]
    result=q.summarize(p,values)
    assert result['complete']and not result['passes']and all(v['E2_evaluable']==0 and v['P_E2_given_E1']is None for v in result['per_type'].values())
    from go2arx5_vla.world_model.task_learning_teacher_v8_qualification import validate as original
    with pytest.raises(ValueError):original(p)


def test_static_or_invented_custody_cannot_admit_full24(monkeypatch):
    monkeypatch.setattr(q,'pinned',lambda v:dict(schema='STATIC_ONLY',original_references={}))
    with pytest.raises((ValueError,KeyError)):q.qualified_small4(dict(path='not-qualified',sha256='c'*64))


def test_original_controllers_fingerprint_clock_and_score_remain_unchanged():
    assert digest(v8.__file__)==q.V8_SOURCE_SHA and digest(small.__file__)==q.SMALL_SOURCE_SHA
    assert legacy.teacher_fingerprint()==legacy.FROZEN_TEACHER
    original=task.protocol(task.QUALIFICATION_ROLE);new=task.protocol(full.ROLE)
    for k in('start_step','prefix_atoms','policy_queries','final_hold_atoms','physics_dt_s','standing','required_object_lift_m','required_hold_s'):
        assert new[k]==original[k]
    from go2arx5_vla.world_model.task_learning_grasp_native_v1 import native_result_schema
    assert native_result_schema(dict(task_learning_role=full.ROLE))==q.NATIVE_SCHEMA

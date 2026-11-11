import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_teacher_e1_v9 as v9
from go2arx5_vla.world_model import task_learning_teacher_e1_v9_qualification as q
from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as old
from go2arx5_vla.world_model import task_learning_teacher_v8 as v8
from go2arx5_vla.world_model.storage import digest

BASE=np.array([0.,0.,.3,1.,0.,0.,0.])


def sample(step=task.B):
    cube=np.eye(4);cube[:3,3]=[.26,-.014,.62];tcp=np.eye(4)
    tcp[:3,3]=cube[:3,3]+[-.012,-.0017,v8.GRASP_Z]
    return dict(native_step=step,time_s=step*.002,contact_time_s=(step-1)*.002,
        tcp=tcp,object_pose=cube,native_grasp=False,gripper_fraction=.9545,
        left_object_contact=False,right_object_contact=False,opposing_contact=False,relative_motion_valid=True,
        safety_valid=True,contact_valid=True,gripper_state_valid=True,object_present=True,
        object_linear_velocity=np.zeros(3),finger_q=np.array([.042,.042]))


def consumed(s,g):
    raw=np.r_[s['tcp'][:3,3],[1.,0.,0.,0.,1.,0.],g];installed=raw.copy();installed[9]=(.002+.04*g)/.044
    return dict(native_end_step=s['native_step'],policy_EEF10=raw,installed_consumed_EEF10=installed,
        solver_gripper_confirmed=True)


def witness(s,lateral=.0024):
    return dict(native_step=s['native_step'],time_s=s['time_s'],supported=True,native_grasp=s['native_grasp'],
        left_object_contact=s['left_object_contact'],right_object_contact=s['right_object_contact'],
        opposing_contact=s['opposing_contact'],relative_motion_valid=s['relative_motion_valid'],
        actual_goal_tool_Y_offset_m=lateral,
        pad_normal_force_N={side:.06 if s[side+'_object_contact']else 0. for side in ('left','right')},
        signed_pad_cube_gap_m={'left':-.0002,'right':.004},
        actual_TCP_position_m=s['tcp'][:3,3].tolist(),actual_cube_position_m=s['object_pose'][:3,3].tolist(),
        previous_complete_H40_solver_gripper_consumption=True)


def test_geometry_overlap_and_nominal_labels_never_establish_actual_single_pad():
    w=witness(sample());assert w['signed_pad_cube_gap_m']['left']<0
    assert not v9.entry_condition('SUPPORTED_SINGLE_PAD',w)
    w['pad_normal_force_N']['left']=.049999
    assert not v9.entry_condition('SUPPORTED_SINGLE_PAD',w)
    w['pad_normal_force_N']['left']=.05
    with pytest.raises(ValueError,match='CONTACT_FLAGS_MATCH'):v9.entry_condition('SUPPORTED_SINGLE_PAD',w)
    w['left_object_contact']=True;assert v9.entry_condition('SUPPORTED_SINGLE_PAD',w)
    w['right_object_contact']=True;w['pad_normal_force_N']['right']=.05
    assert not v9.entry_condition('SUPPORTED_SINGLE_PAD',w)


def test_side_entry_keeps_measured_two_mm_and_actual_support_no_grasp():
    w=witness(sample(),.00199999);assert not v9.entry_condition('SUPPORTED_SIDE_OFFSET',w)
    w['actual_goal_tool_Y_offset_m']=-.002;assert v9.entry_condition('SUPPORTED_SIDE_OFFSET',w)
    w['supported']=False;assert not v9.entry_condition('SUPPORTED_SIDE_OFFSET',w)
    w['supported']=True;w['native_grasp']=True;assert not v9.entry_condition('SUPPORTED_SIDE_OFFSET',w)


@pytest.mark.parametrize('kind',v9.TYPES)
@pytest.mark.parametrize('sign',[-1,1])
def test_new_interventions_preserve_full40_and_command_grip_history(monkeypatch,tmp_path,kind,sign):
    s=sample();state={'g':1.}
    monkeypatch.setattr(v8,'command_context',lambda sample,*args:consumed(sample,state['g']))
    monkeypatch.setattr(v9,'current_witness',lambda adapter,sample,*args:witness(sample))
    adapter=SimpleNamespace(output=tmp_path,spec=dict(teacher_v9_recovery_condition=kind,teacher_v9_offset_sign=sign))
    teacher=v9.NativeQualificationTeacher(adapter,s,lambda:(None,None));teacher.oracle.stage='CLOSE'
    program,proof=teacher.decide(s,s['tcp'],BASE)
    assert program.decoded_H40.shape==(40,10)and program.decoded_H40[0,9]==1.
    assert program.decoded_H40[-1,9]==v9.INTERVENTIONS[kind][1]
    original=np.asarray(proof['unexecuted_Teacher_intent_H40'])
    assert np.allclose(program.decoded_H40[-1,:3]-original[-1,:3],[0,sign*.0045,0])
    assert proof['label_role']=='INTERVENTION_NOT_EXPERT'
    assert proof['physical_envelope_metrics']['maximum_finger_speed_m_s']<=.03
    assert proof['Teacher_version']==v9.VERSION and not proof['expert_training_eligible']
    after=copy.deepcopy(s);after.update(native_step=task.B+1000,time_s=(task.B+1000)*.002)
    state['g']=v9.INTERVENTIONS[kind][1]
    if kind=='SUPPORTED_SINGLE_PAD':
        # Negative FK gap is intentionally insufficient without actual force.
        with pytest.raises(v9.RecoveryUnavailable,match='CONDITION_NOT_ESTABLISHED'):
            teacher.decide(after,after['tcp'],BASE)
        import json
        assert json.loads((tmp_path/'TEACHER_V9_E1_ENTRY.private.json').read_bytes())['E1_condition_established']is False
        assert not(tmp_path/'TEACHER_V9_E2_RECOVERED.private.json').exists()
    else:
        teacher.decide(after,after['tcp'],BASE);assert teacher.entry_observed


def test_actual_one_pad_entry_reuses_release_then_later_native_grasp(monkeypatch,tmp_path):
    s=sample();state={'g':1.}
    monkeypatch.setattr(v8,'command_context',lambda sample,*args:consumed(sample,state['g']))
    monkeypatch.setattr(v9,'current_witness',lambda adapter,sample,*args:witness(sample))
    adapter=SimpleNamespace(output=tmp_path,spec=dict(teacher_v9_recovery_condition='SUPPORTED_SINGLE_PAD',teacher_v9_offset_sign=1))
    teacher=v9.NativeQualificationTeacher(adapter,s,lambda:(None,None));teacher.oracle.stage='CLOSE'
    teacher.decide(s,s['tcp'],BASE)
    after=copy.deepcopy(s);after.update(native_step=task.B+1000,time_s=(task.B+1000)*.002,left_object_contact=True)
    after['tcp'][1,3]+=.0024;state['g']=.5
    program,proof=teacher.decide(after,after['tcp'],BASE)
    assert teacher.entry_observed and proof['recovery_substage']=='SUPPORTED_RELEASE'
    assert program.decoded_H40[-1,9]==.8 and not teacher.native_grasp_after_correction
    recovered=copy.deepcopy(after);recovered.update(native_step=task.B+2000,time_s=(task.B+2000)*.002,
        native_grasp=True,right_object_contact=True,opposing_contact=True)
    state['g']=.8;teacher.decide(recovered,recovered['tcp'],BASE)
    assert teacher.native_grasp_after_correction and(tmp_path/'TEACHER_V9_E2_RECOVERED.private.json').exists()


def make_panel(monkeypatch):
    from go2arx5_vla.world_model import task_learning_collection_v1 as collection
    monkeypatch.setattr(collection,'qualified_base',lambda value:None)
    prereg=Path(__file__).resolve().parents[2]/'experiments/wm02/task_learning_teacher_e1_v9_candidates.json'
    rows=[dict(native_run=f'v9-{i}',layout=l,restart=r,recovery_condition=k,offset_sign=sign,Teacher_mode=v9.VERSION,
        purpose=q.PURPOSE,split='TRAIN',command=task.COMMAND,parent=q.scene_id(l),model_sha256='a'*64)
        for i,(l,r,k,sign)in enumerate(q.ORDER)]
    return dict(schema=q.SCHEMA,task_learning_role=v9.ROLE,protocol=v9.contract(),source_sha='b'*40,
        V9_implementation_SHA256=digest(v9.__file__),unchanged_V8_Oracle_SHA256=digest(v8.__file__),
        preserved_V7_fingerprint=old.FROZEN_TEACHER,preregistration=dict(path=str(prereg),sha256=digest(prereg)),
        base_Teacher_qualification={},scene_producers={l:{}for l in q.LAYOUTS},assignments=rows)


def test_fixed4_and_E1_E2_conditional_denominators_never_promote_unestablished(monkeypatch):
    p=make_panel(monkeypatch);assert len(q.validate(p))==4
    outcomes=[dict(row,attempted=True,technical_valid=True,hard_safety_stop=False,E1_condition_established=False,
        E2_actual_recovered=None,E3_original_task_success=False,genuine_recovery_success=False)for row in p['assignments']]
    result=q.summarize(p,outcomes)
    assert result['complete']and not result['E1_passes']and not result['passes']
    assert all(v['E2_evaluable']==0 and v['P_E2_given_E1']is None for v in result['per_type'].values())
    assert q.summarize(p,outcomes[:2])['requested']==4 and not q.summarize(p,outcomes[:2])['complete']
    with pytest.raises(ValueError):q.summarize(p,outcomes+[outcomes[0]])
    bad=copy.deepcopy(p);bad['assignments'][0]['offset_sign']=-1
    with pytest.raises(ValueError):q.validate(bad)


def test_completed_task_without_real_E1_E2_cannot_be_recovery_success():
    row=dict(recovery_condition='SUPPORTED_SINGLE_PAD');native=dict(status='NATIVE_STAGE_PASSED',
        actual_target_integrations=task.E,reference_qualification=dict(schema=q.NATIVE_SCHEMA,
            excluded_intervention_decisions=1,expert_policy_decisions=14),shutdown_clean=True,
        owned_process_exits=[dict(name=n,exit_code=0)for n in ('sim','solver','controller','planner')])
    kwargs=dict(native=native,worker=dict(exit_code=0),lease=dict(status='CLOSED'),
        readiness=dict(passes=True,native_step=task.B),score=dict(task_success=True),
        spec=dict(task_learning_role=v9.ROLE,source_sha='a'*40,phase=task.PHASE),
        expected_source='a'*40,attempted=True)
    result=q.classify(row,**kwargs)
    assert result['E3_original_task_success']and not result['task_success']and result['E2_actual_recovered']is None
    s=sample(task.B+1000);s['left_object_contact']=True
    entry=dict(witness(s),Teacher_version=v9.VERSION,recovery_condition='SUPPORTED_SINGLE_PAD',
        E1_condition_established=True,intervention_native_start_step=task.B)
    s=sample(task.B+2000);s.update(left_object_contact=True,right_object_contact=True,native_grasp=True,opposing_contact=True)
    recovered=dict(witness(s),Teacher_version=v9.VERSION,recovery_condition='SUPPORTED_SINGLE_PAD',
        E2_actual_recovered=True,E1_entry_native_step=task.B+1000)
    result=q.classify(row,entry_witness=entry,recovered_witness=recovered,**kwargs)
    assert result['task_success']and result['E1_condition_established']and result['E2_actual_recovered']


def test_original_V7_V8_and_full_clock_score_are_unchanged():
    assert old.teacher_fingerprint()==old.FROZEN_TEACHER and digest(v8.__file__)==q.FROZEN_V8_SHA
    original=task.protocol(task.QUALIFICATION_ROLE);new=task.protocol(v9.ROLE)
    for k in('start_step','prefix_atoms','policy_queries','final_hold_atoms','physics_dt_s','standing','required_object_lift_m','required_hold_s'):
        assert new[k]==original[k]
    from go2arx5_vla.world_model.task_learning_grasp_native_v1 import native_result_schema
    assert native_result_schema(dict(task_learning_role=v9.ROLE))==q.NATIVE_SCHEMA

import copy
import pytest

from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_teacher_v8 as v8
from go2arx5_vla.world_model import task_learning_teacher_v8_qualification as q
from go2arx5_vla.world_model import task_learning_collection_v1 as collection
from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as legacy
from go2arx5_vla.world_model.storage import digest


def panel(monkeypatch):
    monkeypatch.setattr(collection,'qualified_base',lambda _:None)
    rows=[dict(native_run=f'v8-{i:02d}',layout=layout,restart=restart,recovery_condition=kind,
        Teacher_mode=v8.VERSION,offset_sign=1 if restart==1 else -1,purpose=q.PURPOSE,split='TRAIN',
        command=task.COMMAND,parent=q.scene_id(layout),model_sha256='a'*64)
        for i,(layout,restart,kind)in enumerate(q.assignments())]
    return dict(schema=q.SCHEMA,task_learning_role=v8.ROLE,protocol=v8.contract(),source_sha='b'*40,
        V8_implementation_SHA256=digest(v8.__file__),preserved_V7_fingerprint=legacy.FROZEN_TEACHER,
        base_Teacher_qualification={},scene_producers={k:dict(path='/sealed/'+k,sha256='a'*64)for k in q.RECOVERY_OFFSETS},assignments=rows)


def test_new_role_keeps_exact_native_horizon_and_old_V7_protocol():
    old=task.protocol(task.QUALIFICATION_ROLE);new=task.protocol(v8.ROLE)
    assert old['task_revision']=='V7_TOOL_X_GRASP_OFFSET_WITH_ORIGINAL_REFERENCE_FLANGE_CLEARANCE'
    assert new['task_revision']==v8.VERSION
    assert new['start_step']==old['start_step']==5703 and new['prefix_atoms']==40 and new['policy_queries']==15
    assert legacy.teacher_fingerprint()==legacy.FROZEN_TEACHER


def test_partial_or_per_type_failures_cannot_pass_qualification(monkeypatch):
    p=panel(monkeypatch);assert len(q.validate(p))==24
    outcomes=[dict(r,attempted=True,reset_ready=True,genuine_recovery_success=True,hard_safety_stop=False,
        technical_valid=True,qualification_condition_not_established=False)for r in p['assignments']]
    partial=q.summarize(p,outcomes[:23]);assert partial['requested']==24 and not partial['complete']and not partial['passes']
    failed=copy.deepcopy(outcomes)
    under=[r for r in failed if r['recovery_condition']=='UNDER_CLOSE']
    under[0]['genuine_recovery_success']=under[1]['genuine_recovery_success']=False
    s=q.summarize(p,failed);assert s['genuine_recovery_successes']==22 and not s['passes']
    failed=copy.deepcopy(outcomes);failed[0]['genuine_recovery_success']=False;failed[1]['genuine_recovery_success']=False
    assert q.summarize(p,failed)['passes']is True


def test_duplicate_mode_or_retry_is_not_a24_request_panel(monkeypatch):
    p=panel(monkeypatch);p['assignments'][1]=copy.deepcopy(p['assignments'][0])
    with pytest.raises(ValueError,match='FIXED24'):q.validate(p)


def test_full_task_alone_does_not_prove_genuine_recovery(monkeypatch):
    row=panel(monkeypatch)['assignments'][0]
    evidence=dict(Teacher_version=v8.VERSION,recovery_condition=row['recovery_condition'],
        intervention_issued=True,actual_recovery_entry_observed=True,native_grasp_after_correction=True)
    native=dict(status='NATIVE_STAGE_PASSED',actual_target_integrations=task.E,shutdown_clean=True,
        owned_process_exits=[dict(name=n,exit_code=0)for n in ('sim','solver','controller','planner')],
        reference_qualification=dict(schema=q.NATIVE_SCHEMA,teacher_v8_recovery_evidence=evidence,
            excluded_intervention_decisions=1,expert_policy_decisions=14))
    args=dict(native=native,worker=dict(exit_code=0),lease=dict(status='CLOSED'),
        readiness=dict(passes=True,native_step=task.B),score=dict(task_success=True),
        spec=dict(task_learning_role=v8.ROLE,phase=task.PHASE,source_sha='candidate'),
        expected_source='candidate',attempted=True)
    assert q.classify(row,**args)['genuine_recovery_success']is True
    for key in ('intervention_issued','actual_recovery_entry_observed','native_grasp_after_correction'):
        evidence[key]=False
        out=q.classify(row,**args)
        assert out['original_task_success']is True and out['genuine_recovery_success']is False
        evidence[key]=True


def test_typed_recovery_exit_does_not_override_wrong_role_or_phase(monkeypatch):
    row=panel(monkeypatch)['assignments'][0]
    native=dict(status='NATIVE_STAGE_FAILED_PRESERVED',shutdown_clean=True,
        owned_process_exits=[dict(name=n,exit_code=0)for n in ('sim','solver','controller','planner')],
        failure='ValueError: TEACHER_V8_RECOVERY_UNAVAILABLE:DECLARED_RECOVERY_CONDITION_NOT_ESTABLISHED')
    args=dict(native=native,worker=dict(exit_code=1),lease=dict(status='CLOSED'),
        readiness=dict(passes=True,native_step=task.B),score={},
        spec=dict(task_learning_role=v8.ROLE,phase=task.PHASE,source_sha='candidate'),
        expected_source='candidate',attempted=True,recovery_exit=dict(Teacher_version=v8.VERSION,
            recovery_condition=row['recovery_condition'],reason='DECLARED_RECOVERY_CONDITION_NOT_ESTABLISHED'))
    out=q.classify(row,**args)
    assert out['technical_valid']is True and out['task_success']is False and out['qualification_condition_not_established']is True
    for field,value in [('task_learning_role',task.QUALIFICATION_ROLE),('phase','unrelated'),('source_sha','old')]:
        changed=copy.deepcopy(args);changed['spec'][field]=value
        assert q.classify(row,**changed)['technical_valid']is False


@pytest.mark.parametrize('changed',['builder','model'])
def test_archived_builder_custody_reuses_original_scene_without_old_code_execution(tmp_path,monkeypatch,changed):
    from go2arx5_vla.world_model import task_learning_student_registration_v2 as registration
    from go2arx5_vla.world_model.task_learning_scene_v1 import MEMBER
    source=tmp_path/'archived';builder=source/MEMBER;builder.parent.mkdir(parents=True)
    builder.write_text('archived builder provenance; never execute')
    model=tmp_path/'SCENE.private.mjb';model.write_bytes(b'sealed original recovery scene')
    reference=dict(path='/sealed/static',sha256='a'*64)
    proof=dict(parent='TASK_LEARNING_UNDER_HAND_RECOVERY01',builder_sha256=digest(builder),compiled=dict(scene_sha256=digest(model)))
    original=dict(data_role='TRAIN',task_learning_role=task.RECOVERY_QUALIFICATION_ROLE,
        task_learning_scene_reference=reference,scene_id=proof['parent'],data_parent=proof['parent'],
        source_root=str(source),source_files={MEMBER:proof['builder_sha256']},model_path=str(model))
    monkeypatch.setattr(registration,'closed_base',lambda pin:original)
    monkeypatch.setattr(q,'pinned',lambda pin:proof)
    assert q.archived_scene_spec({'path':'original-success'},reference,model)==original
    (builder if changed=='builder'else model).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='ARCHIVED_BUILDER_CUSTODY'):q.archived_scene_spec({'path':'original-success'},reference,model)

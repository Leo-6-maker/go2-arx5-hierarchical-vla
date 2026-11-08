import json
import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import task_learning_scene_v1 as scene
from go2arx5_vla.world_model import task_learning_teacher_panel_v1 as panel


def plan():
    rows=[dict(native_run=f'q-{layout}-{restart}',layout=layout,restart=restart,
        parent=scene.scene_id(layout),split='TRAIN',command=task.COMMAND,
        purpose='TEACHER_QUALIFICATION_NOT_BC',model_sha256='a'*64)
        for restart in(1,2)for layout in scene.QUALIFICATION_OFFSETS]
    return dict(schema=panel.SCHEMA,protocol=task.protocol(task.QUALIFICATION_ROLE),
        Teacher_implementation_source_sections_sha256=panel.FROZEN_TEACHER,assignments=rows)


def outcomes(p,ready=20,successes=20):
    return [dict(native_run=r['native_run'],attempted=True,reset_ready=i<ready,
        task_success=i<successes,technical_valid=True,hard_safety_stop=False)
        for i,r in enumerate(p['assignments'])]


def test_role_change_does_not_change_frozen_teacher_or_physical_contract():
    assert panel.teacher_fingerprint()==panel.FROZEN_TEACHER
    a,b=task.protocol(),task.protocol(task.QUALIFICATION_ROLE)
    assert a.pop('stage')==task.CONSTRUCTION_ROLE
    assert b.pop('stage')==task.QUALIFICATION_ROLE
    assert a==b


@pytest.mark.parametrize('ready,successes,passes',[(20,20,True),(18,17,True),(18,16,False),(17,17,False)])
def test_gate_keeps_reset_conditional_and_requested_denominators_separate(ready,successes,passes):
    p=plan();v=panel.summarize(p,outcomes(p,ready,successes))
    assert v['passes']is passes
    assert v['P_reset_ready']==ready/20
    assert v['P_success_per_requested_episode']==successes/20
    assert v['P_success_given_reset_ready']==successes/ready
    assert v['qualified_training_demonstrations']==0


def test_missing_request_or_hard_stop_or_technical_failure_never_passes():
    p=plan();rows=outcomes(p)
    assert not panel.summarize(p,rows[:-1])['passes']
    rows[3]['hard_safety_stop']=True
    assert not panel.summarize(p,rows)['passes']
    rows[3]['hard_safety_stop']=False;rows[3]['technical_valid']=False
    assert not panel.summarize(p,rows)['passes']


def test_duplicate_restart_or_retried_result_is_rejected():
    p=plan();p['assignments'][-1]['restart']=1
    with pytest.raises(ValueError):panel.validate(p)
    p=plan();rows=outcomes(p)
    with pytest.raises(ValueError):panel.summarize(p,rows+[rows[0]])


def test_construction_and_path_traversal_cannot_enter_qualification_panel():
    for field,value in [('purpose','TASK_CONSTRUCTION_NOT_TEACHER_QUALIFICATION_OR_BC'),('native_run','../other-run')]:
        p=plan();p['assignments'][0][field]=value
        with pytest.raises(ValueError):panel.validate(p)


def test_old_construction_success_is_not_a_qualification_outcome():
    row=plan()['assignments'][0]
    n=dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,owned_process_exits=[dict(exit_code=0)],
        actual_target_integrations=task.E,reference_qualification=dict(schema='TASK_LEARNING_GRASP_CONSTRUCTION_NATIVE_ACTUAL_V1'))
    args=dict(native=n,worker=dict(exit_code=0),lease=dict(status='CLOSED'),
        readiness=dict(passes=True,native_step=task.B),score=dict(task_success=True),
        spec=dict(phase=task.PHASE,source_sha='candidate'),expected_source='candidate',attempted=True)
    actual=panel.classify(row,**args)
    assert actual['task_success']is False and actual['technical_valid']is False
    args['spec']['task_learning_role']=task.QUALIFICATION_ROLE
    args['native']['reference_qualification']['schema']='TASK_LEARNING_TEACHER_QUALIFICATION_NATIVE_ACTUAL_V1'
    assert panel.classify(row,**args)['task_success']is True


def test_safe_reference_rejection_is_preserved_as_teacher_failure():
    row=plan()['assignments'][0]
    v=panel.classify(row,native=dict(status='NATIVE_STAGE_FAILED_PRESERVED',shutdown_clean=True,
        owned_process_exits=[dict(exit_code=0)],failure='ValueError: ORIGINAL_FULL_NATIVE_CONTACT_REJECT'),
        worker=dict(exit_code=1),lease=dict(status='CLOSED'),readiness=dict(passes=True,native_step=task.B),
        score={},spec=dict(task_learning_role=task.QUALIFICATION_ROLE,phase=task.PHASE,source_sha='candidate'),
        expected_source='candidate',attempted=True)
    assert v['technical_valid']is True and v['reset_ready']is True and v['task_success']is False


def test_unregistered_panel_cannot_launch_a_worker(tmp_path,monkeypatch):
    from scripts.wm02.task_learning_teacher_qualification_v1 import run
    import subprocess
    (tmp_path/'control').mkdir();(tmp_path/'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED')))
    f=tmp_path/'INPUT.private.json';f.write_text('{}')
    def forbidden(*args,**kwargs):raise AssertionError('WORKER_LAUNCHED_WITHOUT_ACTIVE_REGISTERED_PANEL')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    with pytest.raises(PermissionError,match='FRESH_ACTUAL_FINITE_TEACHER_PANEL'):run(f)
